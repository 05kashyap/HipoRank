import numpy as np

def extract_graph_features(similarities):
    """Extract graph features from HipoRank similarities structure with error handling"""
    # Initialize features
    features = np.zeros(4)  # [centrality, in_degree, out_degree, clustering]
    feature_count = 0
    
    try:
        # Check if similarities is a HipoRank Similarities object
        if hasattr(similarities, 'sent_to_sent'):
            # Process each sentence similarity matrix
            for sent_sim in similarities.sent_to_sent:
                if hasattr(sent_sim, 'similarities') and sent_sim.similarities.size > 0:
                    sim_matrix = sent_sim.similarities
                    
                    if sim_matrix.ndim == 2 and min(sim_matrix.shape) > 0:
                        # Calculate features
                        centrality = np.mean(np.sum(sim_matrix, axis=1))
                        in_degree = np.mean(np.sum(sim_matrix > 0, axis=0))
                        out_degree = np.mean(np.sum(sim_matrix > 0, axis=1))
                        
                        # Simplified clustering coefficient
                        binary_matrix = (sim_matrix > 0).astype(float)
                        triangle_power = np.matmul(np.matmul(binary_matrix, binary_matrix), binary_matrix)
                        clustering = np.trace(triangle_power) / 6
                        
                        if sim_matrix.shape[0] > 2:
                            n = sim_matrix.shape[0]
                            possible = n * (n-1) * (n-2) / 6
                            clustering = clustering / possible if possible > 0 else 0
                        
                        # Accumulate features
                        features += np.array([centrality, in_degree, out_degree, clustering])
                        feature_count += 1
            
            # Average features if any were calculated
            if feature_count > 0:
                features /= feature_count
                
            return features
        
        # Handle direct numpy array case
        elif isinstance(similarities, np.ndarray) and similarities.ndim == 2:
            sim_matrix = similarities
            
            if min(sim_matrix.shape) > 0:
                centrality = np.mean(np.sum(sim_matrix, axis=1))
                in_degree = np.mean(np.sum(sim_matrix > 0, axis=0))
                out_degree = np.mean(np.sum(sim_matrix > 0, axis=1))
                
                binary_matrix = (sim_matrix > 0).astype(float)
                triangle_power = np.matmul(np.matmul(binary_matrix, binary_matrix), binary_matrix)
                clustering = np.trace(triangle_power) / 6
                
                if sim_matrix.shape[0] > 2:
                    n = sim_matrix.shape[0]
                    possible = n * (n-1) * (n-2) / 6
                    clustering = clustering / possible if possible > 0 else 0
                
                return np.array([centrality, in_degree, out_degree, clustering])
    except Exception as e:
        print(f"Error in extract_graph_features: {e}")
        
    return features  # Return default features

def get_flattened_sentence_embeddings(embeddings):
    """Get fixed-size representation of sentence embeddings by averaging"""
    # Handle HipoRank Embeddings object
    if hasattr(embeddings, 'sentence'):
        # Average all sentence embeddings within each section
        section_avgs = []
        
        for sent_embed in embeddings.sentence:
            if hasattr(sent_embed, 'embeddings') and sent_embed.embeddings.size > 0:
                # Average embeddings for this section
                section_avg = np.mean(sent_embed.embeddings, axis=0)
                section_avgs.append(section_avg)
        
        if not section_avgs:
            return np.zeros(768)  # Default BERT embedding dimension
            
        # Limit number of sections to include
        max_sections = min(5, len(section_avgs))  # Fixed number of sections
        if len(section_avgs) > max_sections:
            section_avgs = section_avgs[:max_sections]
        elif len(section_avgs) < max_sections:
            # Pad with zeros if we have fewer sections
            padding = [np.zeros(section_avgs[0].shape) for _ in range(max_sections - len(section_avgs))]
            section_avgs.extend(padding)
        
        # Stack and flatten section averages
        return np.concatenate(section_avgs)
    
    # Fallback
    return np.zeros(768 * 5)  # Default for 5 sections

def create_summary_mask(num_sentences, selected_indices):
    """Create binary mask indicating which sentences are in the summary"""
    mask = np.zeros(num_sentences)
    for idx in selected_indices:
        if 0 <= idx < num_sentences:
            mask[idx] = 1
    return mask

def create_state(document, embeddings, similarities, current_summary_indices=None):
    """Create state representation combining document graph features and current summary"""
    # Try to extract graph features from HipoRank similarities
    try:
        graph_features = extract_graph_features(similarities)
    except Exception as e:
        print(f"Warning: Error extracting graph features: {e}")
        graph_features = np.zeros(4)  # Default features on error
    
    # Try to get sentence embeddings - limit to a fixed dimension
    try:
        sentence_features = get_flattened_sentence_embeddings(embeddings)
        # Ensure consistent size by trimming or padding
        target_size = 768 * 5  # For example, keep 5 sentence embeddings worth
        if len(sentence_features) > target_size:
            sentence_features = sentence_features[:target_size]
        elif len(sentence_features) < target_size:
            padding = np.zeros(target_size - len(sentence_features))
            sentence_features = np.concatenate([sentence_features, padding])
    except Exception as e:
        print(f"Warning: Error getting sentence embeddings: {e}")
        sentence_features = np.zeros(768 * 5)  # Default embedding size
    
    # Count total sentences in the document
    try:
        total_sentences = 0
        for section in document.sections:
            total_sentences += len(section.sentences)
    except:
        total_sentences = 100  # Fallback if counting fails
    
    # Create summary mask with fixed size
    max_sentences = min(100, total_sentences)  # Upper limit on number of sentences
    summary_mask = np.zeros(max_sentences)
    
    # Fill in mask for selected sentences
    if current_summary_indices:
        for idx in current_summary_indices:
            if 0 <= idx < max_sentences:
                summary_mask[idx] = 1
    
    # Combine features - ensure all parts have consistent dimensions
    state = np.concatenate([
        graph_features,  # [centrality, in_degree, out_degree, clustering]
        sentence_features,
        summary_mask
    ])
    
    return state