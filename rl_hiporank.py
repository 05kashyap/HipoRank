import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from tqdm import tqdm
import random
from collections import namedtuple
from rouge import Rouge

from hipo_rank.dataset_iterators.billsum import BillsumDataset
from hipo_rank.dataset_iterators.pubmed import PubmedDataset
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.similarities.cos import CosSimilarity
from hipo_rank.directions.edge import EdgeBased
from hipo_rank.scorers.add import AddScorer
from hipo_rank.summarizers.default import DefaultSummarizer

# For storing transitions in memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class SentenceSelector(nn.Module):
    def __init__(self, input_dim):
        super(SentenceSelector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class RLHipoRankSummarizer:
    def __init__(self, 
                 input_dim=768,  # Dimension of BERT embeddings
                 num_words=200, 
                 gamma=0.99,
                 lr=0.001,
                 batch_size=32,
                 memory_size=10000,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 cuda=True):
        
        self.num_words = num_words
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        
        # Policy network - decides which sentence to include in the summary
        self.policy_net = SentenceSelector(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()
        
        # Experience replay memory
        self.memory = ReplayMemory(memory_size)
        
        # For computing ROUGE rewards
        self.rouge = Rouge()
        
    def get_state_features(self, doc, embeddings, similarities, sentence_scores):
        """
        Create state representation for each sentence combining:
        - Sentence embedding
        - Position in document/section
        - HipoRank score
        - Similarity to other selected sentences
        """
        features = []
        flat_sentences = []
        
        # Flatten document sentences
        for sect_idx, section in enumerate(doc.sections):
            for local_idx, sentence in enumerate(section.sentences):
                flat_sentences.append((sect_idx, local_idx, sentence))
        
        # Get embeddings for each sentence
        sent_embeddings = []
        for sect_idx, section_emb in enumerate(embeddings.sentence):
            for local_idx, sent_emb in enumerate(section_emb.embeddings):
                sent_embeddings.append(sent_emb)
        
        # Create features for each sentence
        for i, (sect_idx, local_idx, _) in enumerate(flat_sentences):
            # Extract HipoRank score for this sentence
            hipo_score = 0
            for score, s_idx, l_idx, global_idx in sentence_scores:
                if s_idx == sect_idx and l_idx == local_idx:
                    hipo_score = score
                    break
            
            # Position features
            section_position = sect_idx / len(doc.sections)
            sentence_position = local_idx / len(doc.sections[sect_idx].sentences)
            
            # Normalize HipoRank score
            norm_score = hipo_score / max(s[0] for s in sentence_scores) if sentence_scores else 0
            
            # Combine features
            feature_vector = np.concatenate([
                sent_embeddings[i],  # Sentence embedding
                [section_position],   # Position of section in document
                [sentence_position],  # Position of sentence in section
                [norm_score]          # Normalized HipoRank score
            ])
            
            features.append(feature_vector)
            
        return features, flat_sentences
    
    def select_action(self, state, training=True):
        """Select whether to include a sentence based on policy network"""
        if training and random.random() < self.epsilon:
            # Random action during exploration
            return random.randint(0, 1)
        else:
            # Use policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor([state]).to(self.device)
                action_prob = self.policy_net(state_tensor).item()
                return 1 if action_prob >= 0.5 else 0
    
    def compute_reward(self, selected_sentences, reference):
        """Compute ROUGE score as reward"""
        if not selected_sentences:
            return 0.0
            
        # Join sentences into a summary
        summary = " ".join(selected_sentences)
        reference_text = " ".join(reference)
        
        # Calculate ROUGE scores
        try:
            scores = self.rouge.get_scores(summary, reference_text)[0]
            # Use average of ROUGE-1, ROUGE-2 and ROUGE-L F1 scores
            reward = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
            return reward
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return 0.0
    
    def update_model(self):
        """Train the policy network using experiences from memory"""
        if len(self.memory) < self.batch_size:
            return
            
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.FloatTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        
        # Get current predicted values
        current_output = self.policy_net(state_batch)
        
        # Compute loss between predicted actions and actual actions, weighted by rewards
        loss = self.loss_fn(current_output, action_batch) * reward_batch
        loss = loss.mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train_episode(self, doc, embeddings, similarities, sentence_scores):
        """Train on a single document"""
        # Get state features for all sentences
        states, flat_sentences = self.get_state_features(doc, embeddings, similarities, sentence_scores)
        
        # Initialize summary
        selected_indices = []
        selected_sentences = []
        total_words = 0
        episode_rewards = []
        
        # For each sentence, decide whether to include it
        for i, (state, (sect_idx, local_idx, sentence)) in enumerate(zip(states, flat_sentences)):
            # Select action (1 = include, 0 = exclude)
            action = self.select_action(state, training=True)
            
            if action == 1:
                # Include sentence in summary
                sentence_words = len(sentence.split())
                
                # Check if adding would exceed word limit
                if total_words + sentence_words <= self.num_words:
                    selected_indices.append(i)
                    selected_sentences.append(sentence)
                    total_words += sentence_words
            
            # Compute reward (ROUGE score) for current summary state
            reward = self.compute_reward(selected_sentences, doc.reference)
            episode_rewards.append(reward)
            
            # Store experience in memory
            self.memory.push(
                state,
                action,
                reward,
                None,  # Next state (not used in this implementation)
                i == len(states) - 1  # done flag
            )
            
            # Update policy network
            loss = self.update_model()
        
        # Return final summary and average reward
        avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        summary = [(sentence, 1.0, sect_idx, local_idx, i) 
                  for i, (sect_idx, local_idx, sentence) in enumerate(flat_sentences) 
                  if i in selected_indices]
        
        return summary, avg_reward
    
    def get_summary(self, doc, sorted_scores):
        """Generate a summary using the trained policy network"""
        # Get embeddings and similarities for the document
        embedder = BertEmbedder(
            bert_config_path="models/pacssum_models/bert_config.json",
            bert_model_path="models/pacssum_models/pytorch_model_finetuned.bin",
            bert_tokenizer="bert-base-uncased",
            cuda=torch.cuda.is_available()
        )
        similarity = CosSimilarity()
        direction = EdgeBased()
        
        # Process document
        embeddings = embedder.get_embeddings(doc)
        similarities = similarity.get_similarities(embeddings)
        directed_sims = direction.update_directions(similarities)
        
        # Get state features for all sentences
        states, flat_sentences = self.get_state_features(doc, embeddings, similarities, sorted_scores)
        
        # Initialize summary
        selected_indices = []
        selected_sentences = []
        total_words = 0
        
        # For each sentence, decide whether to include it
        for i, (state, (sect_idx, local_idx, sentence)) in enumerate(zip(states, flat_sentences)):
            # Select action (1 = include, 0 = exclude) without exploration
            action = self.select_action(state, training=False)
            
            if action == 1:
                # Include sentence in summary
                sentence_words = len(sentence.split())
                
                # Check if adding would exceed word limit
                if total_words + sentence_words <= self.num_words:
                    selected_indices.append(i)
                    selected_sentences.append(sentence)
                    total_words += sentence_words
        
        # Create summary in the format expected by HipoRank
        summary = [(sentence, 1.0, sect_idx, local_idx, i) 
                  for i, (sect_idx, local_idx, sentence) in enumerate(flat_sentences) 
                  if i in selected_indices]
        
        return summary

def train_rl_summarizer(dataset_name="billsum", num_epochs=5):
    """Train the RL summarizer on a dataset"""
    # Setup
    output_dir = Path("rl_summary_output")
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    if dataset_name == "pubmed":
        dataset = PubmedDataset(file_path="data/pubmed-dataset/pubmed-dataset/test.txt")
    else:
        dataset = BillsumDataset(split="train")
    
    # Initialize components
    embedder = BertEmbedder(
        bert_config_path="models/pacssum_models/bert_config.json",
        bert_model_path="models/pacssum_models/pytorch_model_finetuned.bin",
        bert_tokenizer="bert-base-uncased",
        cuda=torch.cuda.is_available()
    )
    similarity = CosSimilarity()
    direction = EdgeBased()
    scorer = AddScorer()
    
    # Detect feature dimension from first document
    doc = next(iter(dataset))
    embeddings = embedder.get_embeddings(doc)
    similarities = similarity.get_similarities(embeddings)
    directed_sims = direction.update_directions(similarities)
    scores = scorer.get_scores(directed_sims)
    
    # Create a temporary summarizer to get feature dimensions
    temp_summarizer = RLHipoRankSummarizer(num_words=200, cuda=torch.cuda.is_available())
    features, _ = temp_summarizer.get_state_features(doc, embeddings, directed_sims, scores)
    input_dim = features[0].shape[0] if features else 771  # Use first feature vector's dimension
    print(f"Detected input dimension: {input_dim}")
    
    # Create the actual summarizer with the correct dimension
    rl_summarizer = RLHipoRankSummarizer(input_dim=input_dim, num_words=200, cuda=torch.cuda.is_available())
    
    # Process documents and train
    docs = list(dataset)[:50]  # Limit to 50 docs for faster training
    all_rewards = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_rewards = []
        
        for doc_idx, doc in enumerate(tqdm(docs, desc="Training on documents")):
            # Get HipoRank features
            embeddings = embedder.get_embeddings(doc)
            similarities = similarity.get_similarities(embeddings)
            directed_sims = direction.update_directions(similarities)
            scores = scorer.get_scores(directed_sims)
            
            # Train RL summarizer on this document
            _, reward = rl_summarizer.train_episode(doc, embeddings, directed_sims, scores)
            epoch_rewards.append(reward)
            
        avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
        all_rewards.append(avg_epoch_reward)
        print(f"Epoch {epoch+1} average reward: {avg_epoch_reward:.4f}")
    
    # Save the trained model
    torch.save(rl_summarizer.policy_net.state_dict(), output_dir / "rl_summarizer_model.pt")
    
    # Plot training progression
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(all_rewards)
        plt.title("Training Rewards")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.savefig(output_dir / "training_rewards.png")
    except:
        pass
    
    return rl_summarizer

def evaluate_rl_summarizer(rl_summarizer, dataset_name="billsum"):
    """Evaluate the RL summarizer on a dataset"""
    # Setup
    output_dir = Path("rl_summary_output")
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    if dataset_name == "pubmed":
        dataset = PubmedDataset(file_path="data/pubmed-dataset/pubmed-dataset/test.txt")
    else:
        dataset = BillsumDataset(split="test")
    
    # Initialize components
    embedder = BertEmbedder(
        bert_config_path="models/pacssum_models/bert_config.json",
        bert_model_path="models/pacssum_models/pytorch_model_finetuned.bin",
        bert_tokenizer="bert-base-uncased",
        cuda=torch.cuda.is_available()
    )
    similarity = CosSimilarity()
    direction = EdgeBased()
    scorer = AddScorer()
    
    # Process documents and generate summaries
    docs = list(dataset)
    results = []
    
    print("Generating summaries with RL-HipoRank...")
    for doc in tqdm(docs):
        # Get HipoRank features
        embeddings = embedder.get_embeddings(doc)
        similarities = similarity.get_similarities(embeddings)
        directed_sims = direction.update_directions(similarities)
        scores = scorer.get_scores(directed_sims)
        
        # Generate summary
        summary = rl_summarizer.get_summary(doc, scores)
        
        results.append({
            "document_id": getattr(doc, "id", "unknown"),
            "summary": summary
        })
    
    # Save results
    with open(output_dir / "rl_summaries.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Evaluate ROUGE scores
    generated_summaries = []
    reference_summaries = []
    
    for doc in docs:
        doc_id = getattr(doc, "id", "unknown")
        
        # Find the generated summary for this document
        summary = None
        for result in results:
            if result["document_id"] == doc_id:
                summary = result["summary"]
                break
        
        if summary:
            # Extract just the sentences (first element of each tuple)
            summary_sentences = [item[0].strip() for item in summary if item[0].strip()]
            
            # Get the reference summary from the document
            reference_summary = doc.reference
            
            # Add to the lists for evaluation
            generated_summaries.append(summary_sentences)
            reference_summaries.append([reference_summary])
    
    # Create text versions of summaries and references
    generated_texts = [' '.join(summary) for summary in generated_summaries]
    reference_texts = [' '.join(ref[0]) for ref in reference_summaries]
    
    # Initialize Rouge
    rouge = Rouge()
    
    # Calculate ROUGE scores
    rouge_results = {}
    for i in range(len(generated_texts)):
        if generated_texts[i] and reference_texts[i]:
            try:
                scores = rouge.get_scores(generated_texts[i], reference_texts[i])[0]
                for metric, values in scores.items():
                    if metric not in rouge_results:
                        rouge_results[metric] = {}
                    for k, v in values.items():
                        key = f"{metric}-{k}"
                        if key not in rouge_results:
                            rouge_results[key] = []
                        rouge_results[key].append(v)
            except Exception as e:
                print(f"Error calculating ROUGE for document {i}: {e}")
    
    # Average the scores
    final_scores = {}
    for metric, values in rouge_results.items():
        final_scores[metric] = sum(values) / len(values) if values else 0
    
    # Save ROUGE results
    with open(output_dir / "rl_rouge_results.json", "w") as f:
        json.dump(final_scores, f, indent=2)
    
    # Print a summary of the ROUGE results
    print("\nRL-HipoRank ROUGE Evaluation Results:")
    for metric, score in final_scores.items():
        print(f"{metric}: {score:.4f}")
    
    return final_scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate RL-enhanced HipoRank")
    parser.add_argument("--dataset", type=str, default="billsum", help="Dataset to use (pubmed or billsum)")
    parser.add_argument("--train", action="store_true", help="Train the RL model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the RL model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.train:
        print(f"Training RL-HipoRank on {args.dataset} dataset")
        rl_summarizer = train_rl_summarizer(dataset_name=args.dataset, num_epochs=args.epochs)
        
        if args.eval:
            print(f"Evaluating trained RL-HipoRank on {args.dataset} dataset")
            evaluate_rl_summarizer(rl_summarizer, dataset_name=args.dataset)
    elif args.eval:
        print(f"Evaluating RL-HipoRank on {args.dataset} dataset")
        # Load a pre-trained model if available, otherwise train a new one
        output_dir = Path("rl_summary_output")
        model_path = output_dir / "rl_summarizer_model.pt"
        
        if model_path.exists():
            rl_summarizer = RLHipoRankSummarizer(num_words=200, cuda=torch.cuda.is_available())
            rl_summarizer.policy_net.load_state_dict(torch.load(model_path))
            print("Loaded pre-trained model")
        else:
            print("No pre-trained model found, training a new one")
            rl_summarizer = train_rl_summarizer(dataset_name=args.dataset, num_epochs=args.epochs)
        
        evaluate_rl_summarizer(rl_summarizer, dataset_name=args.dataset)
    else:
        print("Please specify --train or --eval")