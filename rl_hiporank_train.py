import os
from pathlib import Path
import argparse
import torch
import numpy as np
import random
import json
from tqdm import tqdm

from rl.agents import RLHipoRankAgent
from rl.environment import HipoRankEnvironment, get_valid_actions, global_to_local_idx
from rl.states import create_state
from rl.rewards import calculate_reward

from hipo_rank.dataset_iterators.pubmed import PubmedDataset
from hipo_rank.dataset_iterators.billsum import BillsumDataset
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.similarities.cos import CosSimilarity
from hipo_rank.directions.edge import EdgeBased
from hipo_rank.scorers.add import AddScorer
from hipo_rank.summarizers.default import DefaultSummarizer

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_rl_hiporank(documents, embedder, similarity, direction, scorer, 
                      num_episodes=1000, max_summary_sentences=10, max_words=200,
                      checkpoint_dir="checkpoints", log_interval=50):
    """Train the RL-HipoRank model"""
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    
    # Filter documents that are too long for BERT (512 token limit)
    filtered_documents = []
    for doc in documents:
        total_tokens = 0
        for section in doc.sections:
            for sentence in section.sentences:
                total_tokens += len(sentence.split())
        if total_tokens <= 500:  # Leave some margin for special tokens
            filtered_documents.append(doc)
    
    if len(filtered_documents) == 0:
        raise ValueError("No documents found with acceptable length for BERT processing")
    
    print(f"Filtered documents: {len(filtered_documents)} out of {len(documents)} (kept documents with ≤ 500 tokens)")
    documents = filtered_documents
    
    # Initialize agent with appropriate state size
    # This is a rough estimate - actual size depends on embedding dimension and document length
    sample_doc = documents[0]
    sample_embeddings = embedder.get_embeddings(sample_doc)
    sample_similarities = similarity.get_similarities(sample_embeddings)
    sample_directed_sims = direction.update_directions(sample_similarities)
    
    # Create a sample state to determine the state dimension
    sample_state = create_state(sample_doc, sample_embeddings, sample_directed_sims)
    print(f"State shape: {len(sample_state)}, first few values: {sample_state[:10]}")
    state_size = len(sample_state)
    
    # Count total sentences in the document (across all sections)
    total_sentences = 0
    for section in sample_doc.sections:
        total_sentences += len(section.sentences)
    
    # Fix: Set a reasonable action space size based on observed documents
    action_size = 100  # This should accommodate most documents
    print(f"Document has {total_sentences} total sentences across {len(sample_doc.sections)} sections")
    print(f"Initializing agent with state_size={state_size}, action_size={action_size}")
    
    agent = RLHipoRankAgent(state_size=state_size, action_size=action_size)
    # Training metrics
    episode_rewards = []
    best_avg_reward = float('-inf')
    training_stats = {'actor_loss': [], 'critic_loss': [], 'entropy': [], 'mean_reward': []}
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        # Randomly select a document, with error handling for processing issues
        valid_doc = False
        retries = 0
        doc = None
        
        while not valid_doc and retries < 5:
            try:
                doc = random.choice(documents)
                # Get HipoRank features
                embeddings = embedder.get_embeddings(doc)
                similarities = similarity.get_similarities(embeddings)
                directed_sims = direction.update_directions(similarities)
                valid_doc = True
            except Exception as e:
                print(f"Error processing document in episode {episode}, retrying... ({e})")
                retries += 1
        
        if not valid_doc:
            print(f"Skipping episode {episode} due to document processing errors")
            continue
        
        # Create environment
        try:
            env = HipoRankEnvironment(doc, embeddings, similarities, directed_sims, 
                                    max_summary_sentences, max_words)
        except Exception as e:
            print(f"Error creating environment in episode {episode}: {e}")
            continue
        
        # Initialize episode
        state = env.reset()
        states, actions, rewards, valid_action_masks = [], [], [], []
        episode_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Get valid actions directly from environment
            valid_actions = env.available_actions
            if not valid_actions:
                break
            
            # Create valid action mask - FIX: Ensure the mask is the correct size
            valid_mask = torch.zeros(action_size)
            # Ensure all indices are within bounds
            valid_actions_in_range = [a for a in valid_actions if a < action_size]
            if len(valid_actions_in_range) == 0:
                print(f"Warning: No valid actions within range. Original actions: {valid_actions}")
                break
            
            valid_mask[valid_actions_in_range] = 1.0
            valid_action_masks.append(valid_mask.tolist())
                
            # Get action from agent with epsilon-greedy exploration
            if random.random() < max(0.1, 1.0 - episode / (num_episodes * 0.75)):  # Decay epsilon
                action = random.choice(valid_actions_in_range)
            else:
                action = agent.get_action(state, valid_actions_in_range)
                
            if action is None:
                break
                
            actions.append(action)
            
            # Take step in environment
            next_state, reward, done = env.step(action)
            
            # Store experience
            states.append(state)
            rewards.append(reward)
            episode_reward += reward
            
            # Move to next state
            state = next_state
            
        # Skip update if episode was too short
        if len(states) < 2:
            continue
            
        # Update agent after episode
        update_stats = agent.train(states, actions, rewards, valid_action_masks)
        
        # Track progress
        episode_rewards.append(episode_reward)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            
            # Update training stats
            for key, value in update_stats.items():
                if key in training_stats:
                    training_stats[key].append(value)
            training_stats['mean_reward'].append(avg_reward)
            
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.4f}, "
                  f"Actor Loss: {update_stats['actor_loss']:.4f}, "
                  f"Critic Loss: {update_stats['critic_loss']:.4f}, "
                  f"Entropy: {update_stats['entropy']:.4f}")
            
            # Save checkpoint if improved
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(checkpoint_path / "best_model.pt")
                print(f"New best model saved with avg reward: {best_avg_reward:.4f}")
    
    # Save final model
    agent.save_model(checkpoint_path / "final_model.pt")
    
    # Save training statistics
    with open(checkpoint_path / "training_stats.json", "w") as f:
        json.dump(training_stats, f)
    
    return agent, episode_rewards

def evaluate_rl_agent(agent, documents, embedder, similarity, direction, scorer, num_samples=100, max_words=200):
    """Evaluate the trained RL agent on documents"""
    # Filter documents that are too long for BERT
    filtered_documents = [doc for doc in documents if sum(len(sent.split()) for sect in doc.sections for sent in sect.sentences) <= 500]
    if not filtered_documents:
        print("Warning: No documents with acceptable length for evaluation")
        return []
        
    total_documents = min(len(filtered_documents), num_samples)
    test_documents = random.sample(filtered_documents, total_documents)
    
    results = []
    
    for doc in tqdm(test_documents, desc="Evaluating"):
        try:
            # Get HipoRank features
            embeddings = embedder.get_embeddings(doc)
            similarities = similarity.get_similarities(embeddings)
            directed_sims = direction.update_directions(similarities)
            
            # Create environment
            env = HipoRankEnvironment(doc, embeddings, similarities, directed_sims, max_words=max_words)
            
            # Generate summary
            state = env.reset()
            done = False
            
            while not done:
                valid_actions = get_valid_actions(doc, env.current_summary, max_words)
                # Ensure valid actions are within agent's action space
                valid_actions = [a for a in valid_actions if a < agent.action_size]
                if not valid_actions:
                    break
                    
                action = agent.get_action(state, valid_actions)
                state, reward, done = env.step(action)
            
            # Format summary
            summary = []
            for idx in env.current_summary:
                section_idx, local_idx = global_to_local_idx(doc, idx)
                if section_idx is not None and local_idx is not None:
                    sentence = doc.sections[section_idx].sentences[local_idx]
                    summary.append([sentence, 1.0])  # Format: [sentence, score]
            
            results.append({
                "document_id": getattr(doc, "id", "unknown"),
                "summary": summary
            })
        except Exception as e:
            print(f"Error evaluating document: {e}")
            continue
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL-HipoRank summarization model")
    parser.add_argument("--dataset", type=str, default="billsum", choices=["pubmed", "billsum"],
                        help="Dataset to use (pubmed or billsum)")
    parser.add_argument("--episodes", type=int, default=1000, 
                        help="Number of training episodes")
    parser.add_argument("--max-words", type=int, default=200,
                        help="Maximum number of words in summary")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="rl_summary_output",
                        help="Directory to save results")
    parser.add_argument("--checkpoint-dir", type=str, default="rl_checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate trained model")
    parser.add_argument("--max-sentences", type=int, default=10,
                        help="Maximum number of sentences in summary")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Interval for logging training progress")
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Setup based on dataset choice
        if args.dataset == "pubmed":
            dataset_path = "data/pubmed-dataset/pubmed-dataset/test.txt"
            dataset = PubmedDataset(file_path=dataset_path)
        else:
            dataset = BillsumDataset(split="train")
         
        # Initialize HipoRank components with try/except for better error messages
        embedder = BertEmbedder(
            bert_config_path="models/pacssum_models/bert_config.json",
            bert_model_path="models/pacssum_models/pytorch_model_finetuned.bin",
            bert_tokenizer="bert-base-uncased",
            cuda=torch.cuda.is_available()
        )
        
        # Setting max_length parameter to handle BERT's 512 token limit warning
        if hasattr(embedder, 'tokenizer'):
            embedder.tokenizer.model_max_length = 512
            print(f"Set BERT tokenizer max_length to 512")
        
        similarity = CosSimilarity()
        direction = EdgeBased()
        scorer = AddScorer()
        
        print(f"Loading dataset: {args.dataset}")
        print(f"Cuda available: {torch.cuda.is_available()}")
        
        # Load all documents
        documents = list(dataset)
        print(f"Total documents: {len(documents)}")
        
        # Train RL-HipoRank agent
        print("Starting RL-HipoRank training...")
        agent, rewards = train_rl_hiporank(
            documents, 
            embedder, 
            similarity, 
            direction, 
            scorer,
            num_episodes=args.episodes,
            max_summary_sentences=args.max_sentences,
            max_words=args.max_words,
            checkpoint_dir=args.checkpoint_dir,
            log_interval=args.log_interval
        )
        
        # Save training rewards with more detailed metrics
        episode_data = {
            "rewards": rewards,
            "moving_avg_10": [np.mean(rewards[max(0, i-9):i+1]) for i in range(len(rewards))],
            "moving_avg_50": [np.mean(rewards[max(0, i-49):i+1]) for i in range(len(rewards))],
            "moving_avg_100": [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
        }
        
        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(episode_data, f)
        
        # Plot training curve
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(rewards, alpha=0.3, label='Episode Rewards')
            plt.plot(episode_data['moving_avg_50'], label='50-Episode Moving Avg')
            plt.plot(episode_data['moving_avg_100'], label='100-Episode Moving Avg')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('RL-HipoRank Training Progress')
            plt.legend()
            plt.savefig(output_dir / "training_curve.png")
        except ImportError:
            print("Matplotlib not available for plotting")
        
        # Evaluate if requested
        if args.eval:
            print("Evaluating trained model...")
            results = evaluate_rl_agent(
                agent, 
                documents, 
                embedder, 
                similarity, 
                direction, 
                scorer, 
                num_samples=100,
                max_words=args.max_words
            )
            
            # Save evaluation results
            with open(output_dir / "rl_summaries.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Evaluation results saved to {output_dir / 'rl_summaries.json'}")
    
    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()