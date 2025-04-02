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

class HipoRankEnvironment:
    """Environment wrapper for HipoRank-RL"""
    def __init__(self, document, embeddings, similarities, directed_sims, 
                 max_summary_sentences=10, max_words=200):
        self.document = document
        self.embeddings = embeddings
        self.similarities = similarities
        self.directed_sims = directed_sims
        self.max_summary_sentences = max_summary_sentences
        self.max_words = max_words
        
        # Count total sentences
        self.total_sentences = 0
        for section in document.sections:
            self.total_sentences += len(section.sentences)
        
        # Current state
        self.current_summary = []
        self.current_word_count = 0
        self.available_actions = list(range(self.total_sentences))
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_summary = []
        self.current_word_count = 0
        self.available_actions = list(range(self.total_sentences))
        
        initial_state = create_state(self.document, self.embeddings, self.directed_sims)
        return initial_state
        
    def step(self, action):
        """Take action (select sentence) and return new state, reward, done"""
        # Check if action is valid
        if action not in self.available_actions:
            # Invalid action, provide negative reward
            return self.get_current_state(), -1.0, True
            
        # Add sentence to summary
        self.current_summary.append(action)
        self.available_actions.remove(action)
        
        # Update word count
        section_idx, local_idx = global_to_local_idx(self.document, action)
        if section_idx is not None and local_idx is not None:
            sentence = self.document.sections[section_idx].sentences[local_idx]
            sentence_word_count = len(sentence.split())
            self.current_word_count += sentence_word_count
        
        # Calculate reward
        from rl.rewards import calculate_reward
        reward = calculate_reward(self.document, self.current_summary, self.directed_sims)
        
        # Check if summary is complete
        done = (len(self.current_summary) >= self.max_summary_sentences or 
                self.current_word_count >= self.max_words or
                not self.available_actions)
                
        # Get new state
        new_state = self.get_current_state()
        
        return new_state, reward, done
        
    def get_current_state(self):
        """Get current state representation"""
        return create_state(self.document, self.embeddings, self.directed_sims, self.current_summary)