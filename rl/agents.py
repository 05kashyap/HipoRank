import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, action_size)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)
        return logits

class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        value = self.fc3(x)
        return value

class RLHipoRankAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = PolicyNetwork(state_size, action_size).to(self.device)
        self.critic = ValueNetwork(state_size).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.5, patience=100, verbose=True
        )
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='max', factor=0.5, patience=100, verbose=True
        )

    # Add a method to update learning rate schedulers
    def update_lr_schedulers(self, avg_reward):
        self.actor_scheduler.step(avg_reward)
        self.critic_scheduler.step(avg_reward)
        
    def process_state(self, state):
        """Ensure the state has the correct dimensions"""
        # Convert to numpy array if it's not already
        state_array = np.array(state)
        
        # Check if dimensions match
        if len(state_array) != self.state_size:
            print(f"Warning: State dimension mismatch. Got {len(state_array)}, expected {self.state_size}")
            
            # If state is smaller, pad it
            if len(state_array) < self.state_size:
                padding = np.zeros(self.state_size - len(state_array))
                state_array = np.concatenate([state_array, padding])
            # If state is larger, truncate it
            else:
                state_array = state_array[:self.state_size]
                
        return state_array
        
    def get_action(self, state, valid_actions):
        """Select action based on current state with dynamic action space handling"""
        if not valid_actions:
            return None
            
        # Fix: Convert state to tensor and ensure it's the right shape
        try:
            state_tensor = torch.FloatTensor(self.process_state(state)).to(self.device)
            
            # Get action logits from policy network
            with torch.no_grad():
                action_logits = self.actor(state_tensor)
            
            # Ensure valid actions are within the action space bounds
            valid_actions = [a for a in valid_actions if 0 <= a < self.action_size]
            
            if not valid_actions:
                print("Warning: No valid actions within action space bounds")
                return None
                
            # Create a mask for valid actions
            action_mask = torch.zeros(self.action_size, device=self.device)
            action_mask[valid_actions] = 1.0
            
            # Apply mask by setting logits of invalid actions to a large negative value
            masked_logits = action_logits.clone()
            masked_logits[action_mask == 0] = -1e8  # Use finite negative value to avoid NaN
            
            # Add numerical stability to prevent NaNs in softmax
            masked_logits = masked_logits - masked_logits.max()
            
            # Check for NaN or infinity values and fix them
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                print("Warning: NaN or Inf values in logits, using uniform distribution over valid actions")
                # Use uniform distribution over valid actions
                action = random.choice(valid_actions)
                return action
            
            # Convert to probabilities with increased numerical stability
            action_probs = F.softmax(masked_logits, dim=-1)
            
            # Check for NaN in probabilities
            if torch.isnan(action_probs).any():
                print("Warning: NaN values in probabilities, using uniform distribution")
                action = random.choice(valid_actions)
                return action
            
            # Ensure non-zero probabilities for valid actions (prevent zeros from numerical issues)
            min_prob = 1e-8
            for idx in valid_actions:
                action_probs[idx] = max(min_prob, action_probs[idx])
            
            # Re-normalize probabilities
            action_probs = action_probs / action_probs.sum()
            
            # Force valid action with highest probability if distribution fails
            try:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
            except Exception as e:
                print(f"Error sampling action: {e}, selecting highest probability action")
                action = valid_actions[0]  # Default to first valid action
                max_prob = action_probs[valid_actions[0]]
                
                # Find action with highest probability
                for a in valid_actions:
                    if action_probs[a] > max_prob:
                        max_prob = action_probs[a]
                        action = a
            
            return action
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Return a random valid action as fallback
            return random.choice(valid_actions) if valid_actions else None
        
    def train(self, states, actions, rewards, valid_action_masks):
        """Update networks using collected trajectory with action masks"""
        if len(states) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'mean_advantage': 0, 'mean_return': 0}
        
        try:
            # Process states to ensure correct dimensions
            processed_states = [self.process_state(state) for state in states]
                
            # Convert to tensors
            states_tensor = torch.FloatTensor(processed_states).to(self.device)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            
            # Fix: Ensure valid action masks have correct shape
            fixed_masks = []
            for mask in valid_action_masks:
                if len(mask) < self.action_size:
                    # Extend mask if too small
                    mask = mask + [0.0] * (self.action_size - len(mask))
                elif len(mask) > self.action_size:
                    # Truncate mask if too large
                    mask = mask[:self.action_size]
                fixed_masks.append(mask)
            
            valid_action_masks = torch.FloatTensor(fixed_masks).to(self.device)
            
            # Calculate returns (discounted rewards)
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Normalize returns
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Get values and action logits
            values = self.critic(states_tensor).squeeze()
            action_logits = self.actor(states_tensor)
            
            # Apply action masks
            masked_logits = action_logits.clone()
            # Set invalid action logits to negative infinity to exclude them from softmax
            for i, mask in enumerate(valid_action_masks):
                masked_logits[i][mask == 0] = float('-inf')
            
            # Compute log probabilities and entropy
            log_probs = F.log_softmax(masked_logits, dim=1)
            
            # Fix: Handle potential errors with action indices
            valid_actions = []
            for i, action in enumerate(actions):
                if action >= self.action_size:
                    print(f"Warning: Action {action} out of bounds, using 0 instead")
                    valid_actions.append(0)
                else:
                    valid_actions.append(action)
            
            actions_tensor = torch.LongTensor(valid_actions).unsqueeze(1).to(self.device)
            action_log_probs = log_probs.gather(1, actions_tensor).squeeze()
            
            # Calculate advantage
            advantage = returns - values.detach()
            
            # Actor loss (policy gradient with entropy regularization)
            entropy = -(F.softmax(masked_logits, dim=1) * log_probs).sum(dim=1).mean()
            actor_loss = -(action_log_probs * advantage).mean() - 0.01 * entropy
            
            # Critic loss (MSE)
            critic_loss = F.mse_loss(values, returns)
            
            # Update networks with gradient clipping
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            # Return metrics for logging
            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'entropy': entropy.item(),
                'mean_advantage': advantage.mean().item(),
                'mean_return': returns.mean().item()
            }
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'mean_advantage': 0, 'mean_return': 0}
        
    def save_model(self, path):
        """Save model to path"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size
        }, path)
        
    def load_model(self, path):
        """Load model from path"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.state_size = checkpoint['state_size']
        self.action_size = checkpoint['action_size']