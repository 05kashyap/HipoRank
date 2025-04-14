import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import json
from tqdm import tqdm
import random
from collections import namedtuple, deque
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import math

from hipo_rank.dataset_iterators.billsum import BillsumDataset
from hipo_rank.dataset_iterators.pubmed import PubmedDataset
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.similarities.cos import CosSimilarity
from hipo_rank.directions.edge import EdgeBased
from hipo_rank.scorers.add import AddScorer
from hipo_rank.summarizers.default import DefaultSummarizer

# Download NLTK resources if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# For experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class FeatureCache:
    """Cache for document embeddings and TF-IDF features to avoid redundant computation"""
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.embedding_cache = {}
        self.tfidf_cache = {}
        self.similarity_cache = {}
        self.directed_sim_cache = {}
        self.scores_cache = {}
        self.usage_counter = {}
        
    def get_embeddings(self, doc_id, embedder, doc):
        """Get embeddings from cache or compute and store them"""
        if doc_id in self.embedding_cache:
            self.usage_counter[doc_id] = self.usage_counter.get(doc_id, 0) + 1
            return self.embedding_cache[doc_id]
        
        # Compute embeddings
        embeddings = embedder.get_embeddings(doc)
        
        # Store in cache
        self._manage_cache_size()
        self.embedding_cache[doc_id] = embeddings
        self.usage_counter[doc_id] = 1
        
        return embeddings
    
    def get_similarities(self, doc_id, similarity_module, embeddings):
        """Get similarities from cache or compute and store them"""
        cache_key = f"{doc_id}_sim"
        if cache_key in self.similarity_cache:
            self.usage_counter[cache_key] = self.usage_counter.get(cache_key, 0) + 1
            return self.similarity_cache[cache_key]
        
        # Compute similarities
        similarities = similarity_module.get_similarities(embeddings)
        
        # Store in cache
        self._manage_cache_size()
        self.similarity_cache[cache_key] = similarities
        self.usage_counter[cache_key] = 1
        
        return similarities
    
    def get_directed_sims(self, doc_id, direction_module, similarities):
        """Get directed similarities from cache or compute and store them"""
        cache_key = f"{doc_id}_dir"
        if cache_key in self.directed_sim_cache:
            self.usage_counter[cache_key] = self.usage_counter.get(cache_key, 0) + 1
            return self.directed_sim_cache[cache_key]
        
        # Compute directed similarities
        directed_sims = direction_module.update_directions(similarities)
        
        # Store in cache
        self._manage_cache_size()
        self.directed_sim_cache[cache_key] = directed_sims
        self.usage_counter[cache_key] = 1
        
        return directed_sims
    
    def get_scores(self, doc_id, scorer_module, directed_sims):
        """Get scores from cache or compute and store them"""
        cache_key = f"{doc_id}_scores"
        if cache_key in self.scores_cache:
            self.usage_counter[cache_key] = self.usage_counter.get(cache_key, 0) + 1
            return self.scores_cache[cache_key]
        
        # Compute scores
        scores = scorer_module.get_scores(directed_sims)
        
        # Store in cache
        self._manage_cache_size()
        self.scores_cache[cache_key] = scores
        self.usage_counter[cache_key] = 1
        
        return scores
    
    def get_tfidf_features(self, doc_id, sentences, vectorizer=None):
        """Get TF-IDF features from cache or compute and store them"""
        if doc_id in self.tfidf_cache:
            self.usage_counter[doc_id] = self.usage_counter.get(doc_id, 0) + 1
            return self.tfidf_cache[doc_id]
        
        # Initialize vectorizer if not provided
        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            
        # Compute TF-IDF features
        tfidf_features = vectorizer.fit_transform(sentences)
        
        # Store in cache
        self._manage_cache_size()
        self.tfidf_cache[doc_id] = (tfidf_features, vectorizer)
        self.usage_counter[doc_id] = 1
        
        return tfidf_features, vectorizer
    
    def _manage_cache_size(self):
        """Ensure cache doesn't exceed maximum size by removing least used items"""
        all_caches = [
            self.embedding_cache, 
            self.similarity_cache, 
            self.directed_sim_cache,
            self.scores_cache,
            self.tfidf_cache
        ]
        
        total_size = sum(len(cache) for cache in all_caches)
        
        # If we're below max size, no need to remove anything
        if total_size <= self.max_size:
            return
            
        # Get least used items
        items_to_remove = sorted(
            self.usage_counter.items(),
            key=lambda x: x[1]
        )[:total_size - self.max_size]
        
        # Remove items from caches
        for key, _ in items_to_remove:
            if key in self.embedding_cache:
                del self.embedding_cache[key]
            elif key in self.tfidf_cache:
                del self.tfidf_cache[key]
            elif key.endswith("_sim") and key in self.similarity_cache:
                del self.similarity_cache[key]
            elif key.endswith("_dir") and key in self.directed_sim_cache:
                del self.directed_sim_cache[key]
            elif key.endswith("_scores") and key in self.scores_cache:
                del self.scores_cache[key]
                
            del self.usage_counter[key]


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
    
# Add this new class after the existing ReplayMemory class
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # Priority exponent (how much to prioritize)
        self.beta = beta_start  # Importance sampling correction
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 1  # For beta annealing
        self.epsilon = 1e-5  # Small constant to avoid zero priority
        
    def update_beta(self, progress=None):
        """Anneal beta parameter"""
        if progress is None:
            fraction = min(float(self.frame) / self.beta_frames, 1.0)
        else:
            fraction = min(progress, 1.0)
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)
        self.frame += 1
        
    def push(self, *args):
        """Store a new experience with max priority"""
        max_priority = np.max(self.priorities) if len(self.memory) > 0 else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        
        # Set priority for new experience
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of experiences with prioritization"""
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities
            
        # Calculate sampling probabilities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs[:len(self.memory)])
        samples = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        self.update_beta()
        
        return samples, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD error"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            
    def __len__(self):
        return len(self.memory)

class DuelingDQN(nn.Module):
    """Dueling DQN for better policy learning and value estimation"""
    def __init__(self, input_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        # Feature extraction shared network
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Advantage stream - estimates advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Two actions: include/exclude
        )
        
        # Value stream - estimates state value
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Weight initialization for better gradient flow
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        
        # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class UnsupervisedReward:
    """Compute rewards without reference summaries"""
    def __init__(self, reward_weights={
        'coverage': 0.4, 
        'diversity': 0.3,
        'coherence': 0.2, 
        'informativeness': 0.1
    }):
        self.reward_weights = reward_weights
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.document_vectors = None
        
    def fit_document(self, sentences):
        """Calculate document representation for future reward computation"""
        if not sentences:
            return
            
        # Fit TF-IDF on document sentences
        self.tfidf_vectorizer.fit(sentences)
        
        # Get document vectors
        self.document_vectors = self.tfidf_vectorizer.transform(sentences)

    def calculate_coverage(self, selected_sentences):
        """Calculate how well the summary covers important document content"""
        if not selected_sentences or self.document_vectors is None:
            return 0.0
            
        try:
            # Get TF-IDF vectors for selected sentences
            summary_vectors = self.tfidf_vectorizer.transform(selected_sentences)
            
            # Calculate summary centroid
            summary_centroid = summary_vectors.mean(axis=0)
            
            # For each document sentence, calculate similarity to summary centroid
            similarities = cosine_similarity(summary_centroid, self.document_vectors)
            
            # Coverage is the average similarity
            coverage_score = np.mean(similarities)
            
            return float(coverage_score)
        except Exception as e:
            print(f"Error calculating coverage: {e}")
            return 0.0
    
    def calculate_diversity(self, selected_sentences):
        """Measure diversity within the summary (avoid redundancy)"""
        if len(selected_sentences) < 2:
            return 1.0  # Maximum diversity for a single sentence
            
        try:
            # Get TF-IDF vectors for selected sentences
            summary_vectors = self.tfidf_vectorizer.transform(selected_sentences)
            
            # Calculate pairwise similarities between sentences
            pairwise_similarities = cosine_similarity(summary_vectors)
            
            # Set diagonal to zero (ignore self-similarity)
            np.fill_diagonal(pairwise_similarities, 0)
            
            # Average similarity - higher means more redundancy
            avg_similarity = np.mean(pairwise_similarities)
            
            # Convert to diversity score (1 - redundancy)
            diversity_score = 1.0 - avg_similarity
            
            return float(diversity_score)
        except Exception as e:
            print(f"Error calculating diversity: {e}")
            return 0.5  # Default to neutral diversity
    
    def calculate_coherence(self, selected_sentences, flat_sentences):
        """Calculate coherence based on sentence positions and connections"""
        if len(selected_sentences) < 2:
            return 0.5  # Default coherence for a single sentence
            
        # Get the indices of selected sentences
        selected_indices = []
        for i, (_, _, sentence) in enumerate(flat_sentences):
            if sentence in selected_sentences:
                selected_indices.append(i)
        
        # Sort indices to get the order in the document
        selected_indices.sort()
        
        # Calculate gaps between selected sentences
        gaps = [selected_indices[i+1] - selected_indices[i] for i in range(len(selected_indices)-1)]
        
        if not gaps:
            return 0.5
        
        # Normalize gaps by document length
        doc_length = len(flat_sentences)
        norm_gaps = [min(gap / (doc_length * 0.2), 1.0) for gap in gaps]
        
        # Coherence decreases with larger gaps
        coherence_score = 1.0 - np.mean(norm_gaps)
        
        return float(coherence_score)
    
    def calculate_informativeness(self, selected_sentences):
        """Calculate informativeness based on information density"""
        if not selected_sentences:
            return 0.0
            
        # Calculate average content word ratio
        content_ratios = []
        total_length = 0
        
        for sentence in selected_sentences:
            words = nltk.word_tokenize(sentence.lower())
            content_words = [w for w in words if w not in self.stop_words]
            
            if words:
                ratio = len(content_words) / len(words)
                content_ratios.append(ratio)
                total_length += len(words)
        
        # Penalize very short summaries
        length_factor = min(total_length / 100.0, 1.0)
        
        if not content_ratios:
            return 0.0
            
        # Informativeness is average content word ratio
        informativeness_score = np.mean(content_ratios) * length_factor
        
        return float(informativeness_score)
    
    
    # Add these new methods to the UnsupervisedReward class
    def calculate_length_penalty(self, selected_sentences):
        """Penalize summaries that are too short or too long"""
        if not selected_sentences:
            return 0.0
        
        # Calculate total length in words
        total_words = sum(len(s.split()) for s in selected_sentences)
        
        # Target length is around 200 words with some flexibility
        target_min = 150
        target_max = 250
        
        if total_words < target_min:
            # Penalty increases as length gets shorter
            return max(0.0, 1.0 - (target_min - total_words) / target_min)
        elif total_words > target_max:
            # Penalty increases as length gets longer
            return max(0.0, 1.0 - (total_words - target_max) / target_max)
        else:
            # Perfect length gets maximum score
            return 1.0

    def calculate_position_score(self, selected_sentences, flat_sentences):
        """Reward sentences from important document positions (beginning, end, etc.)"""
        if not selected_sentences:
            return 0.0
            
        # Get positions of selected sentences
        selected_indices = []
        for sentence in selected_sentences:
            for i, (_, _, doc_sentence) in enumerate(flat_sentences):
                if sentence == doc_sentence:
                    selected_indices.append(i)
                    break
        
        # Sort indices
        selected_indices.sort()
        
        # Total sentences in document
        doc_len = len(flat_sentences)
        
        # Calculate position scores (higher for first and last sections)
        position_scores = []
        for idx in selected_indices:
            rel_pos = idx / doc_len
            
            # Higher scores for beginning and end of document
            if rel_pos <= 0.2 or rel_pos >= 0.8:
                score = 1.0
            else:
                # Lower but still positive scores for middle sections
                score = 0.5
                
            position_scores.append(score)
        
        return float(np.mean(position_scores)) if position_scores else 0.0

    # Modify the get_reward method to include these new metrics
    def get_reward(self, selected_sentences, flat_sentences):
        """Calculate overall reward combining multiple unsupervised metrics"""
        if not selected_sentences:
            return 0.0
            
        # Calculate individual reward components
        coverage = self.calculate_coverage(selected_sentences)
        diversity = self.calculate_diversity(selected_sentences)
        coherence = self.calculate_coherence(selected_sentences, flat_sentences)
        informativeness = self.calculate_informativeness(selected_sentences)
        length_score = self.calculate_length_penalty(selected_sentences)
        position_score = self.calculate_position_score(selected_sentences, flat_sentences)
        
        # Updated reward weights with new components
        weights = {
            'coverage': 0.3,
            'diversity': 0.25,
            'coherence': 0.15,
            'informativeness': 0.15,
            'length': 0.1,
            'position': 0.05
        }
        
        # Combine using weights
        total_reward = (
            weights['coverage'] * coverage +
            weights['diversity'] * diversity +
            weights['coherence'] * coherence +
            weights['informativeness'] * informativeness +
            weights['length'] * length_score +
            weights['position'] * position_score
        )
        
        return total_reward

class UnsupervisedRLHipoRankSummarizer:
    def __init__(self, 
             input_dim=771,  # Full feature dimension
             num_words=200, 
             gamma=0.99,  # Increased from 0.95
             lr=0.0003,  # Reduced from 0.001 for more stable training
             batch_size=64,  # Increased from 32
             memory_size=50000,  # Increased from 10000
             target_update=5,  # More frequent than 10
             epsilon_start=1.0,
             epsilon_end=0.01,
             epsilon_decay=0.997,  # Slower decay than 0.995
             reward_weights={
                 'coverage': 0.3, 
                 'diversity': 0.25,
                 'coherence': 0.15, 
                 'informativeness': 0.15,
                 'length': 0.1,
                 'position': 0.05
             },
             cuda=True):
        
        self.num_words = num_words
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.update_count = 0
        self.reward_weights = reward_weights
        
        # Setup device
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Policy and target networks (using improved architecture)
        self.policy_net = DuelingDQN(input_dim).to(self.device)
        self.target_net = DuelingDQN(input_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with improved learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)
        
        # Prioritized experience replay
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Unsupervised reward calculator
        self.reward_calculator = UnsupervisedReward(reward_weights)
        
        # For tracking metrics
        self.episode_rewards = []
        self.avg_losses = []
        
    def get_state_features(self, doc, embeddings, similarities, sentence_scores):
        """Create comprehensive state representation for each sentence"""
        features = []
        flat_sentences = []
        all_sentences = []
        
        # Flatten document sentences
        for sect_idx, section in enumerate(doc.sections):
            for local_idx, sentence in enumerate(section.sentences):
                flat_sentences.append((sect_idx, local_idx, sentence))
                all_sentences.append(sentence)
        
        # Fit reward calculator on document sentences
        self.reward_calculator.fit_document(all_sentences)
        
        # Get embeddings for each sentence
        sent_embeddings = []
        for sect_idx, section_emb in enumerate(embeddings.sentence):
            for local_idx, sent_emb in enumerate(section_emb.embeddings):
                sent_embeddings.append(sent_emb)
        
        # Get sentence centrality from HipoRank similarities
        centrality_scores = {}
        for (i, j), sim in np.ndenumerate(similarities.values):
            if i not in centrality_scores:
                centrality_scores[i] = 0
            centrality_scores[i] += sim
        
        # Normalize centrality scores
        max_centrality = max(centrality_scores.values()) if centrality_scores else 1.0
        normalized_centrality = {k: v / max_centrality for k, v in centrality_scores.items()}
        
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
            
            # Document centrality from similarities
            centrality = normalized_centrality.get(i, 0.0)
            
            # Combine features
            feature_vector = np.concatenate([
                sent_embeddings[i],       # Sentence embedding
                [section_position],       # Position of section in document
                [sentence_position],      # Position of sentence in section
                [norm_score],             # Normalized HipoRank score
                [centrality]              # Sentence centrality in document
            ])
            
            features.append(feature_vector)
            
        return features, flat_sentences, all_sentences
    
    def select_action(self, state, training=True):
        """Select action (include/exclude) using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action during exploration
            return random.randint(0, 1)
        else:
            # Use policy network to select best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor([state]).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def update_model(self):
        """Update the model using prioritized experience replay"""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample from memory with priorities
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Move weights to device
        weights = weights.to(self.device)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        
        # Create mask for non-final states
        non_final_mask = torch.BoolTensor([s is not None for s in batch.next_state]).to(self.device)
        non_final_next_states = torch.FloatTensor([s for s in batch.next_state if s is not None]).to(self.device)
        
        # Get current Q values
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q values using Double DQN approach
        next_q_values = torch.zeros(self.batch_size, 1).to(self.device)
        if len(non_final_next_states) > 0:
            with torch.no_grad():
                # Double Q-learning: use policy net for action selection, target net for value estimation
                next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_q_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions)
        
        # Compute target values
        target_values = reward_batch + self.gamma * next_q_values
        
        # Compute TD errors for updating priorities
        td_errors = torch.abs(target_values - q_values).detach().cpu().numpy()
        
        # Compute weighted loss with importance sampling
        loss = F.smooth_l1_loss(q_values, target_values, reduction='none')
        weighted_loss = (loss * weights.unsqueeze(1)).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors.flatten())
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return weighted_loss.item()
    
    # Add this new method to UnsupervisedRLHipoRankSummarizer
    def get_enhanced_state_features(self, doc, embeddings, similarities, sentence_scores, selected_indices=[]):
        """Create more comprehensive state representation with contextual information"""
        base_features, flat_sentences, all_sentences = self.get_state_features(doc, embeddings, similarities, sentence_scores)
        enhanced_features = []
        
        # Calculate informativeness scores for all sentences
        sentence_info_scores = []
        for i, sentence in enumerate(all_sentences):
            words = nltk.word_tokenize(sentence.lower())
            content_words = [w for w in words if w not in self.reward_calculator.stop_words]
            info_score = len(content_words) / max(len(words), 1)
            sentence_info_scores.append(info_score)
        
        # Get TF-IDF representation if needed
        if not hasattr(self, 'tfidf_vectorizer'):
            self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            self.tfidf_vectorizer.fit(all_sentences)
        
        tfidf_matrix = self.tfidf_vectorizer.transform(all_sentences)
        
        # Calculate summary state (what's already been selected)
        summary_state = np.zeros(len(all_sentences))
        for idx in selected_indices:
            summary_state[idx] = 1
        
        # Calculate redundancy with already selected sentences
        redundancy_scores = np.zeros(len(all_sentences))
        if selected_indices:
            selected_vectors = tfidf_matrix[selected_indices]
            if selected_vectors.shape[0] > 0:
                similarities = cosine_similarity(tfidf_matrix, selected_vectors)
                redundancy_scores = np.mean(similarities, axis=1)
        
        # Enhance features with additional information
        for i, base_feature in enumerate(base_features):
            # Add informativeness score
            info_feature = np.array([sentence_info_scores[i]])
            
            # Add redundancy with current summary
            redundancy_feature = np.array([redundancy_scores[i]])
            
            # Add summary state (whether sentences are already selected)
            summary_feature = np.array([summary_state[i]])
            
            # Add global position
            global_position = np.array([i / len(all_sentences)])
            
            # Combine all features
            enhanced_feature = np.concatenate([
                base_feature,
                info_feature,
                redundancy_feature,
                summary_feature,
                global_position
            ])
            
            enhanced_features.append(enhanced_feature)
        
        return enhanced_features, flat_sentences, all_sentences

    def train_episode(self, doc, embeddings, similarities, sentence_scores):
        """Train on a single document with enhanced features and strategies"""
        # Get base state features for all sentences
        states, flat_sentences, all_sentences = self.get_state_features(doc, embeddings, similarities, sentence_scores)
        
        # Initialize summary
        selected_indices = []
        selected_sentences = []
        total_words = 0
        episode_rewards = []
        
        # For each sentence, decide whether to include it
        for i, (_, (sect_idx, local_idx, sentence)) in enumerate(zip(states, flat_sentences)):
            # Get enhanced state representation with context
            enhanced_states, _, _ = self.get_enhanced_state_features(doc, embeddings, similarities, 
                                                                sentence_scores, selected_indices)
            # Current enhanced state
            current_state = enhanced_states[i].copy()
            
            # Select action (1 = include, 0 = exclude)
            action = self.select_action(current_state, training=True)
            
            # Execute action and observe next state
            prev_selected = selected_sentences.copy()
            prev_indices = selected_indices.copy()
            
            if action == 1:
                # Include sentence in summary
                sentence_words = len(sentence.split())
                
                # Check if adding would exceed word limit
                if total_words + sentence_words <= self.num_words:
                    selected_indices.append(i)
                    selected_sentences.append(sentence)
                    total_words += sentence_words
            
            # Compute unsupervised reward based on the current state of the summary
            reward = self.reward_calculator.get_reward(selected_sentences, flat_sentences)
            
            # Add reward shaping: penalize if nothing changes despite an "include" action
            if action == 1 and prev_selected == selected_sentences:
                reward -= 0.05
                
            episode_rewards.append(reward)
            
            # Determine next state
            next_state = enhanced_states[i+1] if i+1 < len(enhanced_states) else None
            
            # Store transition in memory
            self.memory.push(
                current_state,
                action,
                next_state,
                reward,
                i == len(states) - 1  # done flag
            )
            
            # Update the model (more frequently)
            if i % 4 == 0:  # Update every 4 steps instead of every step
                loss = self.update_model()
        
        # Return final summary and average reward
        avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        summary = [(sentence, 1.0, sect_idx, local_idx, i) 
                for i, (sect_idx, local_idx, sentence) in enumerate(flat_sentences) 
                if i in selected_indices]
        
        self.episode_rewards.append(avg_reward)
        return summary, avg_reward
    
    def get_summary(self, doc, sorted_scores):
        """Generate a summary using the trained policy network with a beam search approach"""
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
        states, flat_sentences, all_sentences = self.get_state_features(doc, embeddings, similarities, sorted_scores)
        
        # Implement a beam search for better summary construction
        beam_width = 3
        beam_candidates = [([], [], 0)]  # (selected_indices, selected_sentences, total_words)
        
        # Process sentences in score order rather than document order
        sentence_indices = list(range(len(states)))
        # Sort by score, higher scores first
        sorted_indices = [idx for _, _, _, idx in sorted(sorted_scores, key=lambda x: x[0], reverse=True)]
        
        # Process top scoring sentences first, then the rest in document order
        top_k = min(20, len(sorted_indices))
        processing_order = sorted_indices[:top_k] + [idx for idx in sentence_indices if idx not in sorted_indices[:top_k]]
        
        for sent_idx in processing_order:
            sect_idx, local_idx, sentence = flat_sentences[sent_idx]
            sentence_words = len(sentence.split())
            
            new_candidates = []
            
            # For each beam candidate
            for selected_idxs, selected_sents, total_words in beam_candidates:
                # Get enhanced state with context of already selected sentences
                enhanced_states, _, _ = self.get_enhanced_state_features(doc, embeddings, similarities, 
                                                                    sorted_scores, selected_idxs)
                if sent_idx >= len(enhanced_states):
                    continue
                    
                current_state = enhanced_states[sent_idx]
                
                # Get action probabilities from policy network
                with torch.no_grad():
                    state_tensor = torch.FloatTensor([current_state]).to(self.device)
                    q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                
                # Both include and skip actions should be considered
                for action in [1, 0]:  # Include or skip
                    new_idxs = selected_idxs.copy()
                    new_sents = selected_sents.copy()
                    new_total_words = total_words
                    
                    # If action is to include the sentence and it fits
                    if action == 1 and total_words + sentence_words <= self.num_words:
                        new_idxs.append(sent_idx)
                        new_sents.append(sentence)
                        new_total_words += sentence_words
                        
                        # Calculate a score for this beam
                        action_value = q_values[action]
                        beam_score = self.reward_calculator.get_reward(new_sents, flat_sentences)
                        
                        # Only add if this creates a change (for include action)
                        if selected_idxs != new_idxs:
                            new_candidates.append((new_idxs, new_sents, new_total_words, beam_score))
                    
                    elif action == 0:  # Skip action always valid
                        # Skipped sentences get a default score based on Q-value
                        action_value = q_values[action]
                        beam_score = self.reward_calculator.get_reward(new_sents, flat_sentences)
                        new_candidates.append((new_idxs, new_sents, new_total_words, beam_score))
            
            # Keep top beam_width candidates
            if new_candidates:
                beam_candidates = [(idxs, sents, words) for idxs, sents, words, score in 
                                sorted(new_candidates, key=lambda x: x[3], reverse=True)[:beam_width]]
        
        # Select best beam result
        if beam_candidates:
            selected_indices, selected_sentences, _ = max(beam_candidates, 
                                                        key=lambda x: self.reward_calculator.get_reward(x[1], flat_sentences))
        else:
            selected_indices, selected_sentences = [], []
        
        # If summary is too short, include highest scoring HipoRank sentences
        if len(selected_sentences) < 3 or sum(len(s.split()) for s in selected_sentences) < self.num_words * 0.5:
            sorted_by_score = sorted(sorted_scores, key=lambda x: x[0], reverse=True)
            for score, sect_idx, local_idx, _ in sorted_by_score:
                sentence = doc.sections[sect_idx].sentences[local_idx]
                sent_idx = next((i for i, (s_idx, l_idx, _) in enumerate(flat_sentences) 
                            if s_idx == sect_idx and l_idx == local_idx), None)
                
                if sent_idx is not None and sent_idx not in selected_indices:
                    sentence_words = len(sentence.split())
                    total_words = sum(len(s.split()) for s in selected_sentences)
                    if total_words + sentence_words <= self.num_words:
                        selected_sentences.append(sentence)
                        selected_indices.append(sent_idx)
        
        # Create summary in the format expected by HipoRank
        summary = [(sentence, 1.0, sect_idx, local_idx, i) 
                for i, (sect_idx, local_idx, sentence) in enumerate(flat_sentences) 
                if i in selected_indices]
        
        return summary
def batch_process_features(docs, embedder, similarity, direction, scorer, feature_cache=None):
    """Process features for a batch of documents using caching when available"""
    if feature_cache is None:
        feature_cache = FeatureCache()
    
    results = []
    
    for doc in docs:
        doc_id = getattr(doc, "id", str(id(doc)))  # Use object id if no document id
        
        # Get features from cache or compute them
        embeddings = feature_cache.get_embeddings(doc_id, embedder, doc)
        similarities = feature_cache.get_similarities(doc_id, similarity, embeddings)
        directed_sims = feature_cache.get_directed_sims(doc_id, direction, similarities)
        scores = feature_cache.get_scores(doc_id, scorer, directed_sims)
        
        # Collect flat sentences for further processing
        flat_sentences = []
        all_sentences = []
        for sect_idx, section in enumerate(doc.sections):
            for local_idx, sentence in enumerate(section.sentences):
                flat_sentences.append((sect_idx, local_idx, sentence))
                all_sentences.append(sentence)
        
        # Store TF-IDF features in cache for future use
        if all_sentences:
            _, vectorizer = feature_cache.get_tfidf_features(doc_id, all_sentences)
        
        results.append({
            "doc_id": doc_id,
            "doc": doc,
            "embeddings": embeddings,
            "similarities": similarities,
            "directed_sims": directed_sims,
            "scores": scores,
            "flat_sentences": flat_sentences,
            "all_sentences": all_sentences
        })
    
    return results

def batch_train_rl_summarizer(rl_summarizer, batch_features, batch_size=8):
    """Train RL summarizer on a batch of documents"""
    batch_rewards = []
    batch_losses = []
    
    for i in range(0, len(batch_features), batch_size):
        current_batch = batch_features[i:i+batch_size]
        
        for features in current_batch:
            doc = features["doc"]
            embeddings = features["embeddings"]
            directed_sims = features["directed_sims"]
            scores = features["scores"]
            
            # Train on this document
            _, reward = rl_summarizer.train_episode(doc, embeddings, directed_sims, scores)
            batch_rewards.append(reward)
            
        # Update model once per mini-batch for efficiency
        loss = rl_summarizer.update_model()
        if loss is not None:
            batch_losses.append(loss)
    
    avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
    avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
    
    return avg_reward, avg_loss

# Update the train_unsupervised_rl_summarizer function to use batch processing
def train_unsupervised_rl_summarizer_batched(dataset_name="billsum", num_epochs=20, batch_size=8):
    """Train the unsupervised RL summarizer on a dataset using batched processing"""
    # Setup
    output_dir = Path("unsupervised_rl_output")
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
    
    # Initialize feature cache
    feature_cache = FeatureCache(max_size=1000)
    print("Feature cache initialized")
    print(f"Cache size: {feature_cache.max_size}")
    # Detect feature dimension from first document
    docs = list(dataset)  # Limit to 50 docs for faster training
    if not docs:
        print("No documents found in dataset")
        return None
    
    # Pre-process first document to get feature dimensions
    doc = docs[0]
    doc_id = getattr(doc, "id", str(id(doc)))
    embeddings = feature_cache.get_embeddings(doc_id, embedder, doc)
    similarities = feature_cache.get_similarities(doc_id, similarity, embeddings)
    directed_sims = feature_cache.get_directed_sims(doc_id, direction, similarities)
    scores = feature_cache.get_scores(doc_id, scorer, directed_sims)
    
    # Create a temporary summarizer to get feature dimensions
    temp_summarizer = UnsupervisedRLHipoRankSummarizer(cuda=torch.cuda.is_available())
    features, _, _ = temp_summarizer.get_state_features(doc, embeddings, directed_sims, scores)
    input_dim = features[0].shape[0] if features else 771
    print(f"Detected input dimension: {input_dim}")
    
    # Create the actual summarizer with the correct dimension
    rl_summarizer = UnsupervisedRLHipoRankSummarizer(
        input_dim=input_dim,
        num_words=200,
        gamma=0.99,
        lr=0.0003,
        batch_size=64,
        memory_size=50000,
        epsilon_decay=0.997,
        target_update=5,
        cuda=torch.cuda.is_available()
    )
    
    # Pre-process all documents in batches
    print("Pre-processing documents...")
    all_features = batch_process_features(docs, embedder, similarity, direction, scorer, feature_cache)
    
    # Process documents and train
    all_rewards = []
    all_losses = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Shuffle documents for each epoch
        random.shuffle(all_features)
        
        # Train in batches
        avg_reward, avg_loss = batch_train_rl_summarizer(rl_summarizer, all_features, batch_size)
        
        all_rewards.append(avg_reward)
        if avg_loss is not None:
            all_losses.append(avg_loss)
            
        print(f"Epoch {epoch+1} average reward: {avg_reward:.4f}, average loss: {avg_loss:.6f}")
    
    # Save the trained model
    torch.save({
        'policy_net': rl_summarizer.policy_net.state_dict(),
        'target_net': rl_summarizer.target_net.state_dict(),
        'input_dim': input_dim
    }, output_dir / "unsupervised_rl_model.pt")
    
    # Plot training progression
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(all_rewards)
        plt.title("Training Rewards")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        
        if all_losses:
            plt.subplot(1, 2, 2)
            plt.plot(all_losses)
            plt.title("Training Losses")
            plt.xlabel("Epoch")
            plt.ylabel("Average Loss")
            
        plt.tight_layout()
        plt.savefig(output_dir / "training_progress.png")
    except Exception as e:
        print(f"Error plotting training progress: {e}")
    
    return rl_summarizer

def train_unsupervised_rl_summarizer(dataset_name="billsum", num_epochs=20):
    """Train the unsupervised RL summarizer on a dataset"""
    # Setup
    output_dir = Path("unsupervised_rl_output")
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
    temp_summarizer = UnsupervisedRLHipoRankSummarizer(cuda=torch.cuda.is_available())
    features, _, _ = temp_summarizer.get_state_features(doc, embeddings, directed_sims, scores)
    input_dim = features[0].shape[0] if features else 771
    print(f"Detected input dimension: {input_dim}")
    
    # Create the actual summarizer with the correct dimension
    rl_summarizer = UnsupervisedRLHipoRankSummarizer(
        input_dim=input_dim,
        num_words=200,
        gamma=0.99,
        lr=0.0003,
        batch_size=64,
        memory_size=50000,
        epsilon_decay=0.997,
        target_update=5,
        cuda=torch.cuda.is_available()
    )
    
    # Process documents and train
    docs = list(dataset)[:50]  # Limit to 50 docs for faster training
    all_rewards = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_rewards = []
        
        # Shuffle documents for each epoch
        random.shuffle(docs)
        
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
    torch.save({
        'policy_net': rl_summarizer.policy_net.state_dict(),
        'target_net': rl_summarizer.target_net.state_dict(),
        'input_dim': input_dim
    }, output_dir / "unsupervised_rl_model.pt")
    
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

def evaluate_both_summarizers(dataset_name="billsum"):
    """Evaluate both HipoRank and Unsupervised RL summarizers"""
    # Setup
    output_dir = Path("unsupervised_rl_output")
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
    
    # Load the trained unsupervised RL model
    model_path = output_dir / "unsupervised_rl_model.pt"
    if model_path.exists():
        print(f"Loading unsupervised RL model from {model_path}")
        checkpoint = torch.load(model_path)
        input_dim = checkpoint.get('input_dim', 771)
        rl_summarizer = UnsupervisedRLHipoRankSummarizer(input_dim=input_dim, cuda=torch.cuda.is_available())
        rl_summarizer.policy_net.load_state_dict(checkpoint['policy_net'])
        rl_summarizer.target_net.load_state_dict(checkpoint['target_net'])
    else:
        print("No pre-trained model found, training a new one")
        rl_summarizer = train_unsupervised_rl_summarizer(dataset_name=dataset_name, num_epochs=5)
    
    # Initialize standard HipoRank summarizer
    hiporank_summarizer = DefaultSummarizer(num_words=200)
    
    # Process documents and generate summaries with both methods
    docs = list(dataset)
    rl_results = []
    hiporank_results = []
    
    print("Generating summaries with both models...")
    for doc in tqdm(docs):
        # Get HipoRank features
        embeddings = embedder.get_embeddings(doc)
        similarities = similarity.get_similarities(embeddings)
        directed_sims = direction.update_directions(similarities)
        scores = scorer.get_scores(directed_sims)
        
        # Generate summary with Unsupervised RL
        rl_summary = rl_summarizer.get_summary(doc, scores)
        
        # Generate summary with HipoRank
        hiporank_summary = hiporank_summarizer.get_summary(doc, scores)
        
        doc_id = getattr(doc, "id", "unknown")
        
        # Store RL results
        rl_results.append({
            "document_id": doc_id,
            "summary": rl_summary
        })
        
        # Store HipoRank results
        hiporank_results.append({
            "document_id": doc_id,
            "summary": hiporank_summary
        })
    
    # Save results
    with open(output_dir / "unsupervised_rl_summaries.json", "w") as f:
        json.dump(rl_results, f, indent=2)
    
    with open(output_dir / "hiporank_summaries.json", "w") as f:
        json.dump(hiporank_results, f, indent=2)
    
    # For each method, calculate ROUGE only for comparison
    # (this is for evaluation only, NOT used in training)
    from rouge import Rouge
    rouge = Rouge()
    
    print("\nROUGE scores are computed only for evaluation, not used in training:")
    
    # Calculate ROUGE for RL model
    rl_rouge = calculate_rouge(rl_results, docs, rouge)
    # Calculate ROUGE for HipoRank
    hiporank_rouge = calculate_rouge(hiporank_results, docs, rouge)
    
    # Print comparison
    print("\nPerformance Comparison:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Unsupervised RL':<15} {'HipoRank':<15} {'Difference':<15}")
    print("-" * 50)
    
    for metric in rl_rouge:
        if metric in hiporank_rouge:
            rl_score = rl_rouge[metric]
            hiporank_score = hiporank_rouge[metric]
            improvement = rl_score - hiporank_score
            
            print(f"{metric:<15} {rl_score:.4f} {hiporank_score:.4f} {improvement:+.4f}")
    
    # Calculate and report unsupervised metrics
    print("\nUnsupervised Metrics Comparison:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Unsupervised RL':<15} {'HipoRank':<15} {'Difference':<15}")
    print("-" * 50)
    
    reward_calculator = UnsupervisedReward()
    
    # Calculate metrics for each document
    rl_metrics = {"coverage": [], "diversity": [], "coherence": [], "informativeness": []}
    hiporank_metrics = {"coverage": [], "diversity": [], "coherence": [], "informativeness": []}
    
    for i, doc in enumerate(tqdm(docs, desc="Calculating metrics")):
        # Get sentences from summaries
        rl_sentences = []
        if i < len(rl_results):
            rl_summary = rl_results[i]["summary"]
            rl_sentences = [item[0] for item in rl_summary]
        
        hiporank_sentences = []
        if i < len(hiporank_results):
            hiporank_summary = hiporank_results[i]["summary"]
            hiporank_sentences = [item[0] for item in hiporank_summary if isinstance(item, list) and item]
        
        # Get all document sentences
        flat_sentences = []
        all_sentences = []
        for sect_idx, section in enumerate(doc.sections):
            for local_idx, sentence in enumerate(section.sentences):
                flat_sentences.append((sect_idx, local_idx, sentence))
                all_sentences.append(sentence)
        
        # Fit reward calculator
        reward_calculator.fit_document(all_sentences)
        
        if rl_sentences:
            rl_metrics["coverage"].append(reward_calculator.calculate_coverage(rl_sentences))
            rl_metrics["diversity"].append(reward_calculator.calculate_diversity(rl_sentences))
            rl_metrics["coherence"].append(reward_calculator.calculate_coherence(rl_sentences, flat_sentences))
            rl_metrics["informativeness"].append(reward_calculator.calculate_informativeness(rl_sentences))
            
        if hiporank_sentences:
            hiporank_metrics["coverage"].append(reward_calculator.calculate_coverage(hiporank_sentences))
            hiporank_metrics["diversity"].append(reward_calculator.calculate_diversity(hiporank_sentences))
            hiporank_metrics["coherence"].append(reward_calculator.calculate_coherence(hiporank_sentences, flat_sentences))
            hiporank_metrics["informativeness"].append(reward_calculator.calculate_informativeness(hiporank_sentences))
    
    # Average metrics
    for metric in ["coverage", "diversity", "coherence", "informativeness"]:
        rl_avg = np.mean(rl_metrics[metric]) if rl_metrics[metric] else 0
        hiporank_avg = np.mean(hiporank_metrics[metric]) if hiporank_metrics[metric] else 0
        diff = rl_avg - hiporank_avg
        
        print(f"{metric:<15} {rl_avg:.4f} {hiporank_avg:.4f} {diff:+.4f}")
    
    return rl_rouge, hiporank_rouge

def evaluate_both_summarizers_batched(dataset_name="billsum", batch_size=8):
    """Evaluate both HipoRank and Unsupervised RL summarizers with batched processing"""
    # Setup
    output_dir = Path("unsupervised_rl_output")
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
    
    # Load the trained unsupervised RL model
    model_path = output_dir / "unsupervised_rl_model.pt"
    if model_path.exists():
        print(f"Loading unsupervised RL model from {model_path}")
        checkpoint = torch.load(model_path)
        input_dim = checkpoint.get('input_dim', 771)
        rl_summarizer = UnsupervisedRLHipoRankSummarizer(input_dim=input_dim, cuda=torch.cuda.is_available())
        rl_summarizer.policy_net.load_state_dict(checkpoint['policy_net'])
        rl_summarizer.target_net.load_state_dict(checkpoint['target_net'])
    else:
        print("No pre-trained model found, training a new one")
        rl_summarizer = train_unsupervised_rl_summarizer_batched(dataset_name=dataset_name, num_epochs=5)
    
    # Initialize standard HipoRank summarizer
    hiporank_summarizer = DefaultSummarizer(num_words=200)
    
    # Initialize feature cache
    feature_cache = FeatureCache(max_size=1000)
    
    # Process documents and generate summaries with both methods
    docs = list(dataset)[:50]  # Limit to 50 for faster evaluation if needed
    rl_results = []
    hiporank_results = []
    
    print("Pre-processing documents for evaluation...")
    all_features = batch_process_features(docs, embedder, similarity, direction, scorer, feature_cache)
    
    print("Generating summaries with both models...")
    for batch_idx in range(0, len(all_features), batch_size):
        batch_features = all_features[batch_idx:batch_idx+batch_size]
        
        for features in tqdm(batch_features, desc=f"Batch {batch_idx//batch_size + 1}/{math.ceil(len(all_features)/batch_size)}"):
            doc = features["doc"]
            embeddings = features["embeddings"]
            directed_sims = features["directed_sims"]
            scores = features["scores"]
            
            # Generate summary with Unsupervised RL
            rl_summary = rl_summarizer.get_summary(doc, scores)
            
            # Generate summary with HipoRank
            hiporank_summary = hiporank_summarizer.get_summary(doc, scores)
            
            doc_id = getattr(doc, "id", "unknown")
            
            # Store RL results
            rl_results.append({
                "document_id": doc_id,
                "summary": rl_summary
            })
            
            # Store HipoRank results
            hiporank_results.append({
                "document_id": doc_id,
                "summary": hiporank_summary
            })
    
    # Save results
    with open(output_dir / "unsupervised_rl_summaries.json", "w") as f:
        json.dump(rl_results, f, indent=2)
    
    with open(output_dir / "hiporank_summaries.json", "w") as f:
        json.dump(hiporank_results, f, indent=2)
    
    # For each method, calculate ROUGE for comparison
    from rouge import Rouge
    rouge = Rouge()
    
    print("\nROUGE scores are computed only for evaluation, not used in training:")
    
    # Calculate ROUGE for RL model
    rl_rouge = calculate_rouge(rl_results, docs, rouge)
    # Calculate ROUGE for HipoRank
    hiporank_rouge = calculate_rouge(hiporank_results, docs, rouge)
    
    # Print comparison
    print("\nPerformance Comparison:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Unsupervised RL':<15} {'HipoRank':<15} {'Difference':<15}")
    print("-" * 50)
    
    for metric in rl_rouge:
        if metric in hiporank_rouge:
            rl_score = rl_rouge[metric]
            hiporank_score = hiporank_rouge[metric]
            improvement = rl_score - hiporank_score
            
            print(f"{metric:<15} {rl_score:.4f} {hiporank_score:.4f} {improvement:+.4f}")
    
    # Calculate and report unsupervised metrics in batches
    print("\nUnsupervised Metrics Comparison:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Unsupervised RL':<15} {'HipoRank':<15} {'Difference':<15}")
    print("-" * 50)
    
    reward_calculator = UnsupervisedReward()
    
    # Calculate metrics for each document
    rl_metrics = {"coverage": [], "diversity": [], "coherence": [], "informativeness": []}
    hiporank_metrics = {"coverage": [], "diversity": [], "coherence": [], "informativeness": []}
    
    for i, feature in enumerate(tqdm(all_features, desc="Calculating metrics")):
        doc = feature["doc"]
        flat_sentences = feature["flat_sentences"]
        all_sentences = feature["all_sentences"]
        
        # Get sentences from summaries
        rl_sentences = []
        if i < len(rl_results):
            rl_summary = rl_results[i]["summary"]
            rl_sentences = [item[0] for item in rl_summary]
        
        hiporank_sentences = []
        if i < len(hiporank_results):
            hiporank_summary = hiporank_results[i]["summary"]
            hiporank_sentences = [item[0] for item in hiporank_summary if isinstance(item, list) and item]
        
        # Fit reward calculator using cached document features when possible
        doc_id = getattr(doc, "id", str(id(doc)))
        _, vectorizer = feature_cache.get_tfidf_features(doc_id, all_sentences)
        reward_calculator.tfidf_vectorizer = vectorizer
        reward_calculator.fit_document(all_sentences)
        
        if rl_sentences:
            rl_metrics["coverage"].append(reward_calculator.calculate_coverage(rl_sentences))
            rl_metrics["diversity"].append(reward_calculator.calculate_diversity(rl_sentences))
            rl_metrics["coherence"].append(reward_calculator.calculate_coherence(rl_sentences, flat_sentences))
            rl_metrics["informativeness"].append(reward_calculator.calculate_informativeness(rl_sentences))
            
        if hiporank_sentences:
            hiporank_metrics["coverage"].append(reward_calculator.calculate_coverage(hiporank_sentences))
            hiporank_metrics["diversity"].append(reward_calculator.calculate_diversity(hiporank_sentences))
            hiporank_metrics["coherence"].append(reward_calculator.calculate_coherence(hiporank_sentences, flat_sentences))
            hiporank_metrics["informativeness"].append(reward_calculator.calculate_informativeness(hiporank_sentences))
    
    # Average metrics
    for metric in ["coverage", "diversity", "coherence", "informativeness"]:
        rl_avg = np.mean(rl_metrics[metric]) if rl_metrics[metric] else 0
        hiporank_avg = np.mean(hiporank_metrics[metric]) if hiporank_metrics[metric] else 0
        diff = rl_avg - hiporank_avg
        
        print(f"{metric:<15} {rl_avg:.4f} {hiporank_avg:.4f} {diff:+.4f}")
    
    return rl_rouge, hiporank_rouge

def calculate_rouge(results, docs, rouge_calculator):
    """Calculate ROUGE scores (for evaluation only)"""
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
            # Extract just the sentences (first element of each tuple or list)
            if isinstance(summary[0], tuple):  # RL format
                summary_sentences = [item[0].strip() for item in summary if item[0].strip()]
            else:  # HipoRank format
                summary_sentences = [item[0].strip() for item in summary if isinstance(item, list) and item]
            
            # Get the reference summary from the document
            reference_summary = doc.reference
            
            # Add to the lists for evaluation
            generated_summaries.append(summary_sentences)
            reference_summaries.append([reference_summary])
    
    # Create text versions of summaries and references
    generated_texts = [' '.join(summary) for summary in generated_summaries]
    reference_texts = [' '.join(ref[0]) for ref in reference_summaries]
    
    # Calculate ROUGE scores
    rouge_results = {}
    for i in range(len(generated_texts)):
        if generated_texts[i] and reference_texts[i]:
            try:
                scores = rouge_calculator.get_scores(generated_texts[i], reference_texts[i])[0]
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
    
    return final_scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate Unsupervised RL-enhanced HipoRank")
    parser.add_argument("--dataset", type=str, default="billsum", help="Dataset to use (pubmed or billsum)")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", action="store_true", help="Use batched processing")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    
    args = parser.parse_args()
    
    if args.train:
        print(f"Training Unsupervised RL-HipoRank on {args.dataset} dataset")
        if args.batch:
            print(f"Using batched processing with batch size {args.batch_size}")
            rl_summarizer = train_unsupervised_rl_summarizer_batched(
                dataset_name=args.dataset, 
                num_epochs=args.epochs,
                batch_size=args.batch_size
            )
        else:
            rl_summarizer = train_unsupervised_rl_summarizer(
                dataset_name=args.dataset, 
                num_epochs=args.epochs
            )
        
        if args.eval:
            print(f"Evaluating trained models on {args.dataset} dataset")
            if args.batch:
                evaluate_both_summarizers_batched(dataset_name=args.dataset, batch_size=args.batch_size)
            else:
                evaluate_both_summarizers(dataset_name=args.dataset)
    elif args.eval:
        print(f"Evaluating models on {args.dataset} dataset")
        if args.batch:
            evaluate_both_summarizers_batched(dataset_name=args.dataset, batch_size=args.batch_size)
        else:
            evaluate_both_summarizers(dataset_name=args.dataset)
    else:
        print("Please specify --train or --eval")