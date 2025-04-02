import torch
import torch.nn as nn
import torch.nn.functional as F

# These models are integrated into agents.py, but we can extend with more specialized networks here

class SentenceEncoder(nn.Module):
    """Additional sentence encoder for richer sentence representations"""
    def __init__(self, input_dim, hidden_dim):
        super(SentenceEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        output, (hidden, _) = self.lstm(x)
        # Concatenate the final forward and backward hidden states
        return torch.cat((hidden[-2], hidden[-1]), dim=1)

class GraphAttention(nn.Module):
    """Graph attention layer for attending to important sentences based on graph structure"""
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, h, adj):
        # h: Node features [N, in_features]
        # adj: Adjacency matrix [N, N]
        
        Wh = torch.mm(h, self.W) # [N, out_features]
        
        # Self-attention mechanism
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Masked attention (zero attention to non-neighbors)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        
        # Apply attention
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(0)
        
        # Create all possible combinations of nodes for attention
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # Combine features for attention computation
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        return all_combinations_matrix.view(N, N, 2 * self.out_features)