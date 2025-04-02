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
    
    action_size = total_sentences  # Max possible action space
    print(f"Document has {total_sentences} total sentences across {len(sample_doc.sections)} sections")
    print(f"Initializing agent with state_size={state_size}, action_size={action_size}")
    
    agent = RLHipoRankAgent(state_size=state_size, action_size=action_size)
    # Training metrics
    episode_rewards = []
    best_avg_reward = float('-inf')
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        doc = random.choice(documents)
        
        # Get HipoRank features
        embeddings = embedder.get_embeddings(doc)
        similarities = similarity.get_similarities(embeddings)
        directed_sims = direction.update_directions(similarities)
        
        # Create environment
        env = HipoRankEnvironment(doc, embeddings, similarities, directed_sims, 
                                 max_summary_sentences, max_words)
        
        # Initialize episode
        state = env.reset()
        states, actions, rewards = [], [], []
        episode_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Get valid actions directly from environment
            valid_actions = env.available_actions
            if not valid_actions:
                break
                
            # Get action from agent
            action = agent.get_action(state, valid_actions)
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
            
        # Update agent after episode
        agent.train(states, actions, rewards)
        
        # Track progress
        episode_rewards.append(episode_reward)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.4f}")
            
            # Save checkpoint if improved
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(checkpoint_path / "best_model.pt")
                print(f"New best model saved with avg reward: {best_avg_reward:.4f}")
    
    # Save final model
    agent.save_model(checkpoint_path / "final_model.pt")
    
    return agent, episode_rewards

def evaluate_rl_agent(agent, documents, embedder, similarity, direction, scorer, num_samples=100, max_words=200):
    """Evaluate the trained RL agent on documents"""
    total_documents = min(len(documents), num_samples)
    test_documents = random.sample(documents, total_documents)
    
    results = []
    
    for doc in tqdm(test_documents, desc="Evaluating"):
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
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup based on dataset choice
    if args.dataset == "pubmed":
        dataset_path = "data/pubmed-dataset/pubmed-dataset/test.txt"
        dataset = PubmedDataset(file_path=dataset_path)
    else:
        dataset = BillsumDataset(split="train")
     
    # Initialize HipoRank components
    embedder = BertEmbedder(
        bert_config_path="models/pacssum_models/bert_config.json",
        bert_model_path="models/pacssum_models/pytorch_model_finetuned.bin",
        bert_tokenizer="bert-base-uncased",
        cuda=torch.cuda.is_available()
    )
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
        max_words=args.max_words,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save training rewards
    with open(output_dir / "training_rewards.json", "w") as f:
        json.dump({"rewards": rewards}, f)
    
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

if __name__ == "__main__":
    main()