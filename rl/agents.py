import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # Add layer normalization for stability
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.ln1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.ln2(x)
        x = self.dropout2(x)
        logits = self.fc3(x)
        return logits

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # Add layer normalization
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.ln1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.ln2(x)
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
        
        # Networks with adjusted hidden size for larger state
        hidden_size = 196  # Larger hidden size for more complex state space
        self.actor = PolicyNetwork(state_size, action_size, hidden_size).to(self.device)
        self.critic = ValueNetwork(state_size, hidden_size).to(self.device)
        
        # Optimizers with weight decay for regularization
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), 
                                           lr=learning_rate, 
                                           weight_decay=1e-4)  # Use AdamW for better regularization
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), 
                                            lr=learning_rate, 
                                            weight_decay=1e-4)
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.5, patience=100, verbose=True
        )
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='max', factor=0.5, patience=100, verbose=True
        )
        
        # Experience replay buffer for more stable training
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32

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
            # If state is smaller, pad it
            if len(state_array) < self.state_size:
                padding = np.zeros(self.state_size - len(state_array))
                state_array = np.concatenate([state_array, padding])
            # If state is larger, truncate it
            else:
                state_array = state_array[:self.state_size]
                
        # Ensure there are no NaN or infinity values
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
        return state_array
        
    def get_action(self, state, valid_actions, transformer_scores=None):
        """Hybrid action selection combining RL policy with transformer guidance"""
        if not valid_actions:
            return None
            
        # Fix: Convert state to tensor and ensure it's the right shape
        try:
            state_tensor = torch.FloatTensor(self.process_state(state)).unsqueeze(0).to(self.device)
            
            # Get action logits from policy network
            with torch.no_grad():
                action_logits = self.actor(state_tensor).squeeze(0)
            
            # Ensure valid actions are within the action space bounds
            valid_actions = [a for a in valid_actions if 0 <= a < self.action_size]
            
            if not valid_actions:
                return None
                
            # Create a mask for valid actions
            action_mask = torch.zeros(self.action_size, device=self.device)
            action_mask[valid_actions] = 1.0
            
            # Apply mask by setting logits of invalid actions to a large negative value
            masked_logits = action_logits.clone()
            masked_logits[action_mask == 0] = -1e8
            
            # Handle NaN values directly
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                # Set NaN/Inf values to appropriate finite values
                masked_logits = torch.nan_to_num(masked_logits, nan=-1e8, posinf=1e3, neginf=-1e8)
            
            # Apply temperature scaling to improve numerical stability
            masked_logits = masked_logits / 1.0  # Temperature parameter
            
            # NEW: Incorporate transformer importance scores if available
            if transformer_scores is not None and len(transformer_scores) > 0:
                # Create tensor for transformer scores
                transformer_tensor = torch.zeros(self.action_size, device=self.device)
                
                # Fill valid scores
                for action in valid_actions:
                    if action < len(transformer_scores):
                        transformer_tensor[action] = transformer_scores[action]
                
                # Blend RL logits with transformer scores (weighted addition)
                # This creates a hybrid decision process guided by both RL and transformer insights
                alpha = 0.7  # Weight for RL component (adjust as needed)
                combined_logits = alpha * masked_logits + (1 - alpha) * transformer_tensor * 5.0
                # Scale transformer scores for comparable magnitudes
                
                # Apply mask again to ensure invalid actions remain invalid
                combined_logits[action_mask == 0] = -1e8
                
                # Convert to probabilities
                action_probs = F.softmax(combined_logits, dim=-1)
            else:
                # Use original RL logits if no transformer scores
                action_probs = F.softmax(masked_logits, dim=-1)
            
            # Double-check for NaN in probabilities after softmax
            if torch.isnan(action_probs).any():
                # Use uniform distribution for valid actions
                action_probs = torch.zeros_like(action_probs)
                action_probs[valid_actions] = 1.0 / len(valid_actions)
            
            # Sample from the distribution
            try:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
                
                # If action is not in valid_actions due to numerical issues, fall back to argmax
                if action not in valid_actions:
                    valid_probs = action_probs[valid_actions]
                    action = valid_actions[torch.argmax(valid_probs).item()]
            except Exception:
                # If sampling fails, choose the valid action with highest probability
                action = valid_actions[0]
                max_prob = action_probs[valid_actions[0]].item()
                
                for a in valid_actions:
                    if action_probs[a].item() > max_prob:
                        max_prob = action_probs[a].item()
                        action = a
            
            return action
        except Exception as e:
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
            
            # Normalize returns for stability
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Get values and action logits
            values = self.critic(states_tensor).squeeze()
            action_logits = self.actor(states_tensor)
            
            # Apply action masks
            masked_logits = action_logits.clone()
            for i, mask in enumerate(valid_action_masks):
                masked_logits[i][mask == 0] = -1e8
            
            # Replace NaN values if any
            masked_logits = torch.nan_to_num(masked_logits, nan=-1e8, posinf=1e3, neginf=-1e8)
            
            # Compute log probabilities and entropy with numerical stability
            log_probs = F.log_softmax(masked_logits, dim=1)
            
            # Fix action indices
            valid_actions = []
            for i, action in enumerate(actions):
                if action >= self.action_size:
                    valid_actions.append(self.action_size - 1)
                else:
                    valid_actions.append(action)
            
            actions_tensor = torch.LongTensor(valid_actions).unsqueeze(1).to(self.device)
            action_log_probs = log_probs.gather(1, actions_tensor).squeeze()
            
            # Calculate advantage
            advantage = returns - values.detach()
            
            # Normalize advantage for stable training
            if advantage.std() > 0:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            # Actor loss with clamped log probabilities to avoid extreme values
            action_log_probs = torch.clamp(action_log_probs, min=-20, max=0)
            entropy = -(F.softmax(masked_logits, dim=1) * log_probs).sum(dim=1).mean()
            entropy = torch.clamp(entropy, min=-5, max=5)  # Clamp entropy to reasonable values
            
            # Improved actor loss with entropy bonus and advantage scaling
            entropy_coef = 0.01  # Coefficient for entropy bonus
            actor_loss = -(action_log_probs * advantage).mean() - entropy_coef * entropy
            
            # Critic loss with Huber loss for robustness
            critic_loss = F.smooth_l1_loss(values, returns)
            
            # Check for NaN in losses
            if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                print("Warning: NaN in loss calculations, skipping update")
                return {
                    'actor_loss': float('nan'),
                    'critic_loss': float('nan'),
                    'entropy': float('nan'),
                    'mean_advantage': 0,
                    'mean_return': 0
                }
            
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
                'actor_loss': actor_loss.item() if not torch.isnan(actor_loss) else 0,
                'critic_loss': critic_loss.item() if not torch.isnan(critic_loss) else 0,
                'entropy': entropy.item() if not torch.isnan(entropy) else 0,
                'mean_advantage': advantage.mean().item() if not torch.isnan(advantage).any() else 0,
                'mean_return': returns.mean().item() if not torch.isnan(returns).any() else 0
            }
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'mean_advantage': 0, 'mean_return': 0}
    
    def store_experience(self, state, action, reward, next_state, done, valid_action_mask):
        """Store experience in replay buffer for off-policy learning"""
        self.replay_buffer.append((state, action, reward, next_state, done, valid_action_mask))
        
    def update_from_replay(self):
        """Learn from random batch of stored experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough samples
            
        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones, valid_masks = zip(*batch)
        
        # Process states
        processed_states = [self.process_state(state) for state in states]
        processed_next_states = [self.process_state(state) for state in next_states]
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(processed_states).to(self.device)
        next_states_tensor = torch.FloatTensor(processed_next_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor([float(d) for d in dones]).unsqueeze(1).to(self.device)
        
        # Fix masks
        fixed_masks = []
        for mask in valid_masks:
            if len(mask) < self.action_size:
                mask = mask + [0.0] * (self.action_size - len(mask))
            elif len(mask) > self.action_size:
                mask = mask[:self.action_size]
            fixed_masks.append(mask)
        valid_masks_tensor = torch.FloatTensor(fixed_masks).to(self.device)
        
        # Get current Q values
        values = self.critic(states_tensor)
        
        # Compute next state values
        with torch.no_grad():
            next_values = self.critic(next_states_tensor)
            # Calculate target values
            target_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_values
        
        # Calculate critic loss
        critic_loss = F.smooth_l1_loss(values, target_values)
        
        # Get action logits
        action_logits = self.actor(states_tensor)
        
        # Apply action masks
        masked_logits = action_logits.clone()
        for i, mask in enumerate(valid_masks_tensor):
            masked_logits[i][mask == 0] = -1e8
            
        # Get log probabilities
        log_probs = F.log_softmax(masked_logits, dim=1)
        selected_log_probs = log_probs.gather(1, actions_tensor)
        
        # Calculate advantage
        advantage = (target_values - values).detach()
        
        # Actor loss
        actor_loss = -(selected_log_probs * advantage).mean()
        
        # Add entropy bonus
        probs = F.softmax(masked_logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        actor_loss -= 0.01 * entropy
        
        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'mean_advantage': advantage.mean().item()
        }
        
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