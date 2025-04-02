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
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)  # Output logit for each possible action
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
            return None  # No valid actions available
        
        # Process the state to ensure correct dimensions    
        state_processed = self.process_state(state)
        state_tensor = torch.FloatTensor(state_processed).unsqueeze(0).to(self.device)
        
        # Get logits for all actions
        with torch.no_grad():
            action_logits = self.actor(state_tensor).cpu().numpy().flatten()
        
        # Create a mask based on valid actions
        # Only consider actions within the network's output range
        valid_actions_in_range = [a for a in valid_actions if a < len(action_logits)]
        if not valid_actions_in_range:
            print("Warning: All valid actions are outside the network's output range")
            return valid_actions[0]  # Fall back to first valid action
        
        # Filter to only valid actions
        masked_logits = np.ones_like(action_logits) * float('-inf')
        for action in valid_actions_in_range:
            masked_logits[action] = action_logits[action]
        
        # Select action with highest probability
        action = np.argmax(masked_logits)
        return action
        
    def train(self, states, actions, rewards):
        """Update networks using collected trajectory"""
        if len(states) == 0:
            return
            
        # Process states to ensure correct dimensions
        processed_states = [self.process_state(state) for state in states]
            
        # Convert to tensors
        states_tensor = torch.FloatTensor(processed_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        
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
        
        # Calculate advantage
        advantage = returns - values.detach()
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(values, returns)
        
        # Actor loss (policy gradient)
        selected_logits = action_logits.gather(1, actions_tensor).squeeze()
        actor_loss = -(selected_logits * advantage).mean()
        
        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
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