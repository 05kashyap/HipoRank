import numpy as np
from .states import create_state

def get_total_sentences(document):
    """Get the total number of sentences in the document"""
    return len(document.sents)

def get_sentence_by_index(document, idx):
    """Get the sentence text by index"""
    return document.sents[idx]

def get_valid_actions(document, current_summary_indices, max_words=200):
    """Return indices of sentences that can still be added to summary"""
    # Count total sentences to determine valid action range
    total_sentences = 0
    for section in document.sections:
        total_sentences += len(section.sentences)
    
    # Filter out sentences already in summary
    valid_indices = [i for i in range(total_sentences) 
                    if i not in current_summary_indices]
    
    # Check word limit
    current_word_count = 0
    for idx in current_summary_indices:
        # Convert global index to section and local index
        section_idx, local_idx = global_to_local_idx(document, idx)
        if section_idx is not None and local_idx is not None:
            sentence = document.sections[section_idx].sentences[local_idx]
            current_word_count += len(sentence.split())
    
    # Further filter based on word limit
    filtered_indices = []
    for idx in valid_indices:
        section_idx, local_idx = global_to_local_idx(document, idx)
        if section_idx is not None and local_idx is not None:
            sentence = document.sections[section_idx].sentences[local_idx]
            if current_word_count + len(sentence.split()) <= max_words:
                filtered_indices.append(idx)
    
    return filtered_indices

def global_to_local_idx(document, global_idx):
    """Convert a global sentence index to section index and local index"""
    count = 0
    for sect_idx, section in enumerate(document.sections):
        if global_idx < count + len(section.sentences):
            local_idx = global_idx - count
            return sect_idx, local_idx
        count += len(section.sentences)
    return None, None  # Invalid index

def get_sentence_by_global_idx(document, global_idx):
    """Get sentence text by global index"""
    section_idx, local_idx = global_to_local_idx(document, global_idx)
    if section_idx is not None and local_idx is not None:
        return document.sections[section_idx].sentences[local_idx]
    return ""

def is_word_limit_reached(document, summary_indices, max_words=200):
    """Check if the word limit for the summary has been reached"""
    word_count = sum(len(get_sentence_by_index(document, i).split()) for i in summary_indices)
    return word_count >= max_words

def get_all_sentences(document):
    """Get all sentences in the document as a flat list"""
    all_sentences = []
    for section in document.sections:
        all_sentences.extend(section.sentences)
    return all_sentences

class HipoRankEnvironment:
    """Environment wrapper for HipoRank-RL with transformer integration"""
    def __init__(self, document, embeddings, similarities, directed_sims, 
                 max_summary_sentences=10, max_words=200, transformer_encoder=None):
        self.document = document
        self.embeddings = embeddings
        self.similarities = similarities
        self.directed_sims = directed_sims
        self.max_summary_sentences = max_summary_sentences
        self.max_words = max_words
        self.transformer_encoder = transformer_encoder  # Add transformer encoder
        
        # Count total sentences
        self.total_sentences = 0
        for section in document.sections:
            self.total_sentences += len(section.sentences)
        
        # Current state
        self.current_summary = []
        self.current_word_count = 0
        
        # Fix: Initialize available actions with proper sentence count
        # Ensure actions don't exceed 100 to match agent's action space
        self.action_size = 100  # Keep consistent with agent
        self.available_actions = list(range(min(self.action_size, self.total_sentences)))
        
        # Initialize transformer features
        self.transformer_features = None
        self.transformer_importance_scores = None
        
        # Extract all sentences from the document
        self.all_sentences = get_all_sentences(document)
        
        # Generate transformer features if transformer encoder is available
        if self.transformer_encoder is not None and self.all_sentences:
            try:
                # Get contextual features from transformer
                self.transformer_features = self.transformer_encoder.get_contextual_sentence_features(self.all_sentences)
                self.transformer_importance_scores = self.transformer_features.get("importance_scores", None)
            except Exception as e:
                print(f"Error generating transformer features: {e}")
                self.transformer_features = None
                self.transformer_importance_scores = None
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_summary = []
        self.current_word_count = 0
        
        # Fix: Reset available actions with proper count and limit
        self.available_actions = list(range(min(self.action_size, self.total_sentences)))
        
        # Create initial state with transformer features
        initial_state = create_state(self.document, self.embeddings, self.directed_sims, 
                                    transformer_features=self.transformer_features)
        return initial_state
        
    def step(self, action):
        """Take action (select sentence) and return new state, reward, done"""
        # Check if action is valid
        if action not in self.available_actions:
            # Invalid action, provide negative reward
            return self.get_current_state(), -1.0, True
            
        # Add sentence to summary
        self.current_summary.append(action)
        
        try:
            self.available_actions.remove(action)
        except ValueError:
            pass  # Action already removed or not in list
        
        # Update word count
        section_idx, local_idx = global_to_local_idx(self.document, action)
        if section_idx is not None and local_idx is not None:
            sentence = self.document.sections[section_idx].sentences[local_idx]
            sentence_word_count = len(sentence.split())
            self.current_word_count += sentence_word_count
        
        # Recalculate valid actions based on current summary and word count
        self.available_actions = get_valid_actions(self.document, self.current_summary, self.max_words)
        # Ensure actions are within bounds for agent compatibility
        self.available_actions = [a for a in self.available_actions if a < self.action_size]
        
        # Get current summary sentences for transformer-based rewards
        current_summary_sentences = []
        for idx in self.current_summary:
            section_idx, local_idx = global_to_local_idx(self.document, idx)
            if section_idx is not None and local_idx is not None:
                current_summary_sentences.append(self.document.sections[section_idx].sentences[local_idx])
        
        # Calculate transformer-based semantic quality if available
        transformer_data = None
        if self.transformer_encoder is not None and current_summary_sentences and self.all_sentences:
            try:
                semantic_score = self.transformer_encoder.calculate_summary_quality(
                    self.all_sentences, current_summary_sentences
                )
                transformer_data = {
                    'importance_scores': self.transformer_importance_scores,
                    'semantic_score': semantic_score
                }
            except Exception as e:
                print(f"Error calculating transformer reward: {e}")
        
        # Calculate reward with transformer guidance if available
        from rl.rewards import calculate_reward
        reward = calculate_reward(self.document, self.current_summary, self.directed_sims, transformer_data)
        
        # Check if summary is complete
        done = (len(self.current_summary) >= self.max_summary_sentences or 
                self.current_word_count >= self.max_words or
                not self.available_actions)
                
        # Get new state
        new_state = self.get_current_state()
        
        return new_state, reward, done
        
    def get_current_state(self):
        """Get current state representation with transformer features"""
        return create_state(self.document, self.embeddings, self.directed_sims, 
                           self.current_summary, self.transformer_features)
    
    def get_transformer_scores(self):
        """Get importance scores from transformer for action selection guidance"""
        if self.transformer_importance_scores is not None:
            return self.transformer_importance_scores
        return None