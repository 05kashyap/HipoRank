import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def plot_training_rewards(rewards_file, output_path=None, window_size=10):
    """
    Plot training rewards from the JSON file saved during RL-HipoRank training
    with advanced analysis of learning rate and convergence.
    """
    # Load the rewards
    with open(rewards_file, 'r') as f:
        data = json.load(f)
    
    rewards = data['rewards']
    episodes = np.arange(1, len(rewards) + 1)
    
    # Calculate moving average for smoother visualization
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    smooth_rewards = moving_average(rewards, window_size)
    smooth_episodes = np.arange(window_size, len(rewards) + 1)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Set up subplots: main rewards and learning rate analysis
    plt.subplot(2, 1, 1)
    
    # Raw rewards (light color)
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Smoothed rewards (darker color)
    plt.plot(smooth_episodes, smooth_rewards, linewidth=2, color='darkblue', 
             label=f'Moving Average (window={window_size})')
    
    # Add trend line
    z = np.polyfit(episodes, rewards, 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), linestyle='--', color='red', 
             label=f'Trend Line (slope={z[0]:.4f})')
    
    # Annotate
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.ylabel('Reward')
    plt.title('RL-HipoRank Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    last_n = min(50, len(rewards))
    recent_mean = np.mean(rewards[-last_n:])
    
    stats_text = (f"Mean Reward: {mean_reward:.4f}\n"
                 f"Max Reward: {max_reward:.4f}\n"
                 f"Min Reward: {min_reward:.4f}\n"
                 f"Last {last_n} Avg: {recent_mean:.4f}")
    
    plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Learning rate analysis in the second subplot
    plt.subplot(2, 1, 2)
    
    # Calculate improvement rate
    window = max(50, window_size)
    improvement_rates = []
    x_vals = []
    
    for i in range(window, len(rewards), window//2):
        prev_window = rewards[i-window:i]
        if i + window < len(rewards):
            next_window = rewards[i:i+window]
            improvement = np.mean(next_window) - np.mean(prev_window)
            improvement_rates.append(improvement)
            x_vals.append(i)
    
    plt.bar(x_vals, improvement_rates, width=window//2, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Improvement Rate')
    plt.title('Learning Improvement Rate (Higher is Better)')
    
    # Detect plateau
    convergence_threshold = 0.01
    plateau_episodes = []
    
    for i in range(len(improvement_rates)-3):
        if all(abs(rate) < convergence_threshold for rate in improvement_rates[i:i+3]):
            plateau_episodes.append(x_vals[i])
            break
    
    if plateau_episodes:
        plt.axvline(x=plateau_episodes[0], color='green', linestyle='--', 
                   label=f'Potential Convergence at Episode {plateau_episodes[0]}')
        plt.legend()
        
        # Add to main plot
        plt.subplot(2, 1, 1)
        plt.axvline(x=plateau_episodes[0], color='green', linestyle='--')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
        # Save analysis summary
        summary_path = Path(output_path).with_suffix('.txt')
        with open(summary_path, 'w') as f:
            f.write(f"RL-HipoRank Training Analysis\n")
            f.write(f"==========================\n\n")
            f.write(f"Total Episodes: {len(rewards)}\n")
            f.write(f"Mean Reward: {mean_reward:.4f}\n")
            f.write(f"Max Reward: {max_reward:.4f}\n")
            f.write(f"Min Reward: {min_reward:.4f}\n")
            f.write(f"Last {last_n} Episodes Average: {recent_mean:.4f}\n")
            f.write(f"Overall Trend Slope: {z[0]:.4f}\n\n")
            
            if plateau_episodes:
                f.write(f"Training likely converged around episode {plateau_episodes[0]}\n")
                f.write(f"Could have stopped training {len(rewards) - plateau_episodes[0]} episodes earlier\n")
            else:
                f.write("No clear convergence detected - more training might be beneficial\n")
                
        print(f"Analysis summary saved to {summary_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RL-HipoRank training rewards with advanced analysis")
    parser.add_argument("--rewards-file", type=str, 
                        default="results/rl_hiporank/training_rewards_500_epochs.json",
                        help="Path to the training_rewards.json file")
    parser.add_argument("--output", type=str, default="results/rl_hiporank/reward_plot_analysis.png",
                        help="Output file path (PNG)")
    parser.add_argument("--window", type=int, default=20,
                        help="Window size for moving average")
    
    args = parser.parse_args()
    
    plot_training_rewards(args.rewards_file, args.output, args.window)