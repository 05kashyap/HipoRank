import numpy as np

def extract_graph_features(similarities):
    """Extract simplified graph features - avoid complex operations"""
    return np.zeros(4)  # Return default zeros instead of complex graph operations

def get_flattened_sentence_embeddings(embeddings):
    """Simplified version to avoid using complex HipoRank embeddings structure"""
    # Return fixed-size zero vector as a placeholder
    return np.zeros(100)  # Smaller fixed size

def create_summary_mask(num_sentences, selected_indices):
    """Create binary mask indicating which sentences are in the summary"""
    mask = np.zeros(num_sentences)
    for idx in selected_indices:
        if 0 <= idx < num_sentences:
            mask[idx] = 1
    return mask

def create_state(document, embeddings, similarities, current_summary_indices=None):
    """Create simplified state representation avoiding complex HipoRank structures"""
    # Count total sentences in the document (simplified)
    total_sentences = 0
    for section in document.sections:
        total_sentences += len(section.sentences)
    
    # Limit to a reasonable size
    max_sentences = min(100, total_sentences)
    
    # Basic features about the document
    doc_features = np.zeros(5)
    doc_features[0] = len(document.sections)  # Number of sections
    doc_features[1] = total_sentences  # Total sentences
    doc_features[2] = total_sentences / max(1, len(document.sections))  # Avg sentences per section
    doc_features[3] = 0.5  # Fixed value to replace complex similarity calcs
    doc_features[4] = 0.5  # Fixed value to replace complex calcs
    
    # Position features - simple array of sentence positions (0-1 range)
    position_features = np.zeros(max_sentences)
    for i in range(min(max_sentences, total_sentences)):
        position_features[i] = i / max(1, total_sentences - 1)
    
    # Create summary mask with fixed size (binary indicators of selected sentences)
    summary_mask = np.zeros(max_sentences)
    
    if current_summary_indices:
        for idx in current_summary_indices:
            if 0 <= idx < max_sentences:
                summary_mask[idx] = 1
    
    # Word count features (approximate sentence lengths)
    word_count_features = np.zeros(max_sentences)
    sent_idx = 0
    for section in document.sections:
        for sentence in section.sentences:
            if sent_idx < max_sentences:
                # Normalize by a typical max sentence length (e.g., 50 words)
                word_count_features[sent_idx] = min(1.0, len(sentence.split()) / 50.0)
                sent_idx += 1
    
    # Combine features - ensure all parts have consistent dimensions
    state = np.concatenate([
        doc_features,  # Basic document features
        position_features,  # Position information
        word_count_features,  # Sentence length information
        summary_mask  # Which sentences are already selected
    ])
    
    return state