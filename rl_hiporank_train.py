import os
from pathlib import Path
import argparse
import torch
import numpy as np
import random
import json
from tqdm import tqdm

from rl.agents import RLHipoRankAgent
from rl.environment import HipoRankEnvironment, get_valid_actions, global_to_local_idx, get_all_sentences
from rl.states import create_state
from rl.rewards import calculate_reward
from rl.transformer_utils import TransformerSentenceEncoder  # Import transformer utils

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
                      checkpoint_dir="checkpoints", log_interval=50,
                      transformer_model="distilbert-base-uncased"):
    """Train the RL-HipoRank model with transformer guidance"""
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
    
    # Initialize transformer encoder for enhanced representations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_encoder = None
    
    try:
        print(f"Initializing transformer encoder with model: {transformer_model}")
        transformer_encoder = TransformerSentenceEncoder(
            model_name=transformer_model,
            max_length=512,
            device=device
        )
        print("Transformer encoder initialized successfully")
    except Exception as e:
        print(f"Failed to initialize transformer encoder: {e}. Continuing without transformer guidance.")
    
    # Initialize agent with updated state size based on our enhanced state representation in states.py
    # The state size is now fixed at 5 + 4*23 = 97 (see updated create_state function)
    state_size = 5 + (4 * 23)  # Updated fixed size with transformer features
    action_size = 100  # Fixed action space size
    
    print(f"Initializing agent with state_size={state_size}, action_size={action_size}")
    
    agent = RLHipoRankAgent(state_size=state_size, action_size=action_size, 
                          learning_rate=0.0005)  # Lower learning rate for stability
    
    # Training metrics
    episode_rewards = []
    best_avg_reward = float('-inf')
    training_stats = {'actor_loss': [], 'critic_loss': [], 'entropy': [], 'mean_reward': []}
    
    # Add early stopping to prevent wasting time on non-improving runs
    no_improvement_count = 0
    best_reward_window = float('-inf')
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        # Randomly select a document, with error handling for processing issues
        valid_doc = False
        retries = 0
        doc = None
        
        while not valid_doc and retries < 5:
            try:
                doc = random.choice(documents)
                # Get HipoRank features - with simplified approach
                embeddings = embedder.get_embeddings(doc)
                similarities = similarity.get_similarities(embeddings)
                directed_sims = direction.update_directions(similarities)
                valid_doc = True
            except Exception as e:
                retries += 1
                print(f"Error processing document: {e}. Retrying...")
        
        if not valid_doc:
            print(f"Skipping episode {episode} due to document processing errors")
            continue
        
        # Create environment with transformer encoder
        try:
            env = HipoRankEnvironment(
                doc, embeddings, similarities, directed_sims, 
                max_summary_sentences, max_words,
                transformer_encoder=transformer_encoder
            )
        except Exception as e:
            print(f"Error creating environment in episode {episode}: {e}")
            continue
        
        # Initialize episode
        state = env.reset()
        states, actions, rewards, valid_action_masks = [], [], [], []
        next_states = []  # Store next states for off-policy learning
        dones = []  # Store done flags
        episode_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Get valid actions directly from environment
            valid_actions = env.available_actions
            if not valid_actions:
                break
            
            # Create valid action mask - ensure the mask is the correct size
            valid_mask = torch.zeros(action_size)
            valid_actions_in_range = [a for a in valid_actions if a < action_size]
            if len(valid_actions_in_range) == 0:
                break
            
            valid_mask[valid_actions_in_range] = 1.0
            valid_action_masks.append(valid_mask.tolist())
                
            # Get transformer importance scores if available
            transformer_scores = env.get_transformer_scores() if hasattr(env, "get_transformer_scores") else None
                
            # Get action from agent with decreasing epsilon for exploration and transformer guidance
            epsilon = max(0.05, 0.5 - episode / (num_episodes * 0.3))  # Faster decay
            if random.random() < epsilon:
                # Exploration: randomly select from valid actions
                action = random.choice(valid_actions_in_range)
            else:
                # Exploitation: use agent's policy with transformer guidance
                action = agent.get_action(state, valid_actions_in_range, transformer_scores)
                
            if action is None:
                break
                
            actions.append(action)
            states.append(state)  # Store state before taking action
            
            # Take step in environment
            next_state, reward, done = env.step(action)
            
            # Store experience
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            episode_reward += reward
            
            # Store experience in replay buffer for off-policy learning
            if hasattr(agent, "store_experience"):
                agent.store_experience(state, action, reward, next_state, done, valid_mask.tolist())
            
            # Move to next state
            state = next_state
        
        # Skip update if episode was too short
        if len(states) < 2:
            continue
            
        # Update agent after episode with gradient clipping
        update_stats = agent.train(states, actions, rewards, valid_action_masks)
        
        # Optionally perform additional updates from replay buffer
        if hasattr(agent, "update_from_replay") and episode > 100 and episode % 5 == 0:
            replay_stats = agent.update_from_replay()
            if replay_stats:
                # Incorporate replay stats into update_stats for logging
                for key, value in replay_stats.items():
                    if key in update_stats:
                        update_stats[key] = (update_stats[key] + value) / 2
        
        # Track progress
        episode_rewards.append(episode_reward)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            window_size = min(log_interval, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-window_size:])
            
            # Update learning rate schedulers based on performance
            agent.update_lr_schedulers(avg_reward)
            
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
                no_improvement_count = 0  # Reset counter
            else:
                no_improvement_count += 1
            
            # Early stopping check (after enough episodes)
            if episode > 10000 and no_improvement_count >= 1000:
                print(f"No improvement for {no_improvement_count * log_interval} episodes. Early stopping.")
                break
                
            # Check for reward window improvement (for long-term trends)
            if episode > 500:  # Only after enough episodes
                window_reward = np.mean(episode_rewards[-500:])
                if window_reward > best_reward_window:
                    best_reward_window = window_reward
                    agent.save_model(checkpoint_path / "trend_model.pt")
    
    # Save final model
    agent.save_model(checkpoint_path / "final_model.pt")
    
    # Save training statistics
    with open(checkpoint_path / "training_stats.json", "w") as f:
        json.dump(training_stats, f)
    
    return agent, episode_rewards

def evaluate_rl_agent(agent, documents, embedder, similarity, direction, scorer,
                     num_samples=100, max_words=200, transformer_model="distilbert-base-uncased"):
    """Evaluate the trained RL agent on documents with transformer enhancement"""
    # Filter documents that are too long for BERT
    filtered_documents = [doc for doc in documents if sum(len(sent.split()) for sect in doc.sections for sent in sect.sentences) <= 500]
    if not filtered_documents:
        print("Warning: No documents with acceptable length for evaluation")
        return []
        
    total_documents = min(len(filtered_documents), num_samples)
    test_documents = random.sample(filtered_documents, total_documents)
    
    # Initialize transformer encoder if possible
    transformer_encoder = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer_encoder = TransformerSentenceEncoder(
            model_name=transformer_model,
            max_length=512,
            device=device
        )
        print("Using transformer guidance for evaluation")
    except Exception as e:
        print(f"Failed to initialize transformer for evaluation: {e}")
    
    results = []
    
    for doc in tqdm(test_documents, desc="Evaluating"):
        try:
            # Get HipoRank features
            embeddings = embedder.get_embeddings(doc)
            similarities = similarity.get_similarities(embeddings)
            directed_sims = direction.update_directions(similarities)
            
            # Create environment with transformer integration
            env = HipoRankEnvironment(
                doc, embeddings, similarities, directed_sims, 
                max_words=max_words,
                transformer_encoder=transformer_encoder
            )
            
            # Generate summary
            state = env.reset()
            done = False
            
            while not done:
                valid_actions = get_valid_actions(doc, env.current_summary, max_words)
                # Ensure valid actions are within agent's action space
                valid_actions = [a for a in valid_actions if a < agent.action_size]
                if not valid_actions:
                    break
                
                # Get transformer scores for hybrid action selection
                transformer_scores = env.get_transformer_scores() if hasattr(env, "get_transformer_scores") else None
                
                # Use hybrid action selection for better summaries
                action = agent.get_action(state, valid_actions, transformer_scores)
                state, reward, done = env.step(action)
            
            # Format summary with transformer-based quality scores
            summary = []
            current_summary_sentences = []
            
            for idx in env.current_summary:
                section_idx, local_idx = global_to_local_idx(doc, idx)
                if section_idx is not None and local_idx is not None:
                    sentence = doc.sections[section_idx].sentences[local_idx]
                    current_summary_sentences.append(sentence)
                    
                    # Get sentence importance from transformer if available
                    importance_score = 1.0  # Default score
                    if transformer_encoder is not None and hasattr(env, "transformer_importance_scores"):
                        if env.transformer_importance_scores is not None and idx < len(env.transformer_importance_scores):
                            importance_score = float(env.transformer_importance_scores[idx])
                    
                    summary.append([sentence, importance_score])
            
            # Add semantic quality score if transformer is available
            semantic_quality = 0.0
            if transformer_encoder is not None and current_summary_sentences:
                try:
                    all_sentences = get_all_sentences(doc)
                    semantic_quality = transformer_encoder.calculate_summary_quality(
                        all_sentences, current_summary_sentences
                    )
                except Exception as e:
                    print(f"Error calculating semantic quality: {e}")
            
            results.append({
                "document_id": getattr(doc, "id", "unknown"),
                "summary": summary,
                "semantic_quality": semantic_quality
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
    parser.add_argument("--transformer-model", type=str, default="distilbert-base-uncased",
                        help="Transformer model to use for sentence encoding")
    parser.add_argument("--no-transformer", action="store_true",
                        help="Disable transformer guidance (use base RL only)")
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set transformer model or None if disabled
    transformer_model = None if args.no_transformer else args.transformer_model
    
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
        
        # Train RL-HipoRank agent with transformer enhancements
        print("Starting RL-HipoRank training with transformer guidance...")
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
            log_interval=args.log_interval,
            transformer_model=transformer_model
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
            plt.title('Transformer-Enhanced RL-HipoRank Training Progress')
            plt.legend()
            plt.savefig(output_dir / "training_curve.png")
        except ImportError:
            print("Matplotlib not available for plotting")
        
        # Evaluate if requested
        if args.eval:
            print("Evaluating trained model with transformer guidance...")
            results = evaluate_rl_agent(
                agent, 
                documents, 
                embedder, 
                similarity, 
                direction, 
                scorer, 
                num_samples=100,
                max_words=args.max_words,
                transformer_model=transformer_model
            )
            
            # Save evaluation results
            with open(output_dir / "rl_summaries.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Evaluation results saved to {output_dir / 'rl_summaries.json'}")
            
            # Generate evaluation metrics summary if transformer was used
            if transformer_model is not None:
                semantic_qualities = [result["semantic_quality"] for result in results if "semantic_quality" in result]
                if semantic_qualities:
                    metrics_summary = {
                        "avg_semantic_quality": sum(semantic_qualities) / len(semantic_qualities),
                        "min_semantic_quality": min(semantic_qualities),
                        "max_semantic_quality": max(semantic_qualities),
                        "num_summaries": len(results)
                    }
                    
                    with open(output_dir / "evaluation_metrics.json", "w") as f:
                        json.dump(metrics_summary, f, indent=2)
                    
                    print(f"Average semantic quality: {metrics_summary['avg_semantic_quality']:.4f}")
    
    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()