import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .environment import global_to_local_idx

def calculate_coverage(document, selected_indices, directed_sims):
    """Calculate how well the selected sentences cover important content"""
    if not selected_indices:
        return 0.0
        
    # Extract sentence-sentence similarities
    sent_sims = []
    for sim_obj in directed_sims.sent_to_sent:
        sent_sims.append(sim_obj.similarities)
    
    if not sent_sims:
        return 0.0
        
    # Get original HipoRank scores as reference for importance
    from hipo_rank.scorers.add import AddScorer
    scorer = AddScorer()
    base_scores = scorer.get_scores(directed_sims)
    
    # Create a mapping from global index to score tuple
    # HipoRank scores are tuples: (score, section_idx, local_idx, global_idx)
    score_map = {score_tuple[3]: score_tuple[0] for score_tuple in base_scores}
    
    # Calculate coverage as weighted average of selected sentence scores
    total_score = 0.0
    valid_indices = 0
    
    for idx in selected_indices:
        if idx in score_map:
            total_score += score_map[idx]
            valid_indices += 1
            
    # Normalize by number of valid selected sentences
    coverage_score = total_score / valid_indices if valid_indices else 0.0
    
    # Normalize to 0-1 range by dividing by maximum possible score
    if base_scores:
        # Get top scores equal to number of selected sentences
        top_scores = sorted([s[0] for s in base_scores], reverse=True)[:len(selected_indices)]
        max_possible = sum(top_scores)
        if max_possible > 0:
            coverage_score /= max_possible
            
    return coverage_score

def get_sentence_embeddings(document, selected_indices, embeddings=None):
    """Get embeddings for selected sentences from HipoRank document structure"""
    if embeddings is None:
        # Default to empty embeddings if none provided
        return np.zeros((len(selected_indices), 768))
        
    # Extract embeddings for selected sentences
    result_embeddings = []
    
    for idx in selected_indices:
        # Convert global index to section and local indices
        from rl.environment import global_to_local_idx
        section_idx, local_idx = global_to_local_idx(document, idx)
        
        if section_idx is not None and local_idx is not None:
            # Find embedding in HipoRank embedding structure
            try:
                if (section_idx < len(embeddings.sentence) and 
                    local_idx < embeddings.sentence[section_idx].embeddings.shape[0]):
                    result_embeddings.append(embeddings.sentence[section_idx].embeddings[local_idx])
                else:
                    # Fallback: use zero vector
                    result_embeddings.append(np.zeros(768))
            except (AttributeError, IndexError):
                # Fallback: use zero vector
                result_embeddings.append(np.zeros(768))
        else:
            # Invalid index, use zero vector
            result_embeddings.append(np.zeros(768))
            
    return np.array(result_embeddings)

def calculate_diversity(document, selected_indices, directed_sims):
    """Calculate diversity (lack of redundancy) in selected sentences"""
    if len(selected_indices) < 2:
        return 1.0  # Perfect diversity with 0-1 sentences
    
    # Convert global indices to section and local indices
    from rl.environment import global_to_local_idx
    section_local_indices = [global_to_local_idx(document, idx) for idx in selected_indices]
    
    # Filter out any invalid indices
    section_local_indices = [x for x in section_local_indices if x[0] is not None and x[1] is not None]
    
    # Group indices by section
    indices_by_section = {}
    for i, (sect_idx, local_idx) in enumerate(section_local_indices):
        if sect_idx not in indices_by_section:
            indices_by_section[sect_idx] = []
        indices_by_section[sect_idx].append((local_idx, selected_indices[i]))
    
    # Calculate similarities within each section
    redundancy_scores = []
    
    for sect_idx, local_indices in indices_by_section.items():
        if sect_idx >= len(directed_sims.sent_to_sent):
            continue
            
        # Get similarity data for this section
        sent_sim = directed_sims.sent_to_sent[sect_idx]
        
        # We need to find similarities between selected sentences within this section
        for i, (idx1, global_idx1) in enumerate(local_indices):
            for j, (idx2, global_idx2) in enumerate(local_indices[i+1:], i+1):
                # Find similarity between these two sentences by searching through pair_indices
                sim_found = False
                try:
                    # Try to find a direct pair
                    for pair_idx, (pi, pj) in enumerate(sent_sim.pair_indices):
                        if (pi == idx1 and pj == idx2) or (pi == idx2 and pj == idx1):
                            redundancy_scores.append(sent_sim.similarities[pair_idx])
                            sim_found = True
                            break
                    
                    # If no direct pair found, use a default low similarity
                    if not sim_found:
                        redundancy_scores.append(0.5)  # Default moderate similarity
                except (AttributeError, IndexError, TypeError):
                    redundancy_scores.append(0.5)  # Default on error
    
    # Average redundancy (higher = more redundant)
    avg_redundancy = sum(redundancy_scores) / len(redundancy_scores) if redundancy_scores else 0.0
    
    # Diversity is inverse of redundancy (1 - redundancy)
    diversity_score = 1.0 - avg_redundancy
    
    return diversity_score

def calculate_coherence(document, selected_indices, directed_sims):
    """Calculate coherence between adjacent selected sentences in summary order"""
    if len(selected_indices) < 2:
        return 1.0  # Perfect coherence with 0-1 sentences
    
    # Sort selected indices to get sentences in order
    ordered_indices = sorted(selected_indices)
    
    # Convert global indices to section and local indices
    from rl.environment import global_to_local_idx
    section_local_indices = [(global_to_local_idx(document, idx), idx) for idx in ordered_indices]
    
    # Filter out any invalid indices
    section_local_indices = [(x, g_idx) for (x, g_idx) in section_local_indices if x[0] is not None and x[1] is not None]
    
    # Calculate coherence between consecutive sentences
    coherence_scores = []
    
    for i in range(len(section_local_indices) - 1):
        (sect_idx1, local_idx1), global_idx1 = section_local_indices[i]
        (sect_idx2, local_idx2), global_idx2 = section_local_indices[i+1]
        
        # Check for similarity within the same section
        if sect_idx1 == sect_idx2 and sect_idx1 < len(directed_sims.sent_to_sent):
            sent_sim = directed_sims.sent_to_sent[sect_idx1]
            
            # Find similarity between these two sentences by searching through pair_indices
            sim_found = False
            try:
                for pair_idx, (pi, pj) in enumerate(sent_sim.pair_indices):
                    if (pi == local_idx1 and pj == local_idx2) or (pi == local_idx2 and pj == local_idx1):
                        coherence_scores.append(sent_sim.similarities[pair_idx])
                        sim_found = True
                        break
                
                if not sim_found:
                    coherence_scores.append(0.5)  # Default moderate similarity
            except (AttributeError, IndexError, TypeError):
                coherence_scores.append(0.5)  # Default on error
        
        # Check for similarity between sections
        elif sect_idx1 < len(directed_sims.sent_to_sect) and sect_idx2 < len(document.sections):
            try:
                # Get sent_to_sect data for the first section
                sent_to_sect = directed_sims.sent_to_sect[sect_idx1]
                
                # Try to find similarity between sentence in sect1 and sect2
                sim_found = False
                for pair_idx, (s_idx, l_idx) in enumerate(sent_to_sect.pair_indices):
                    if s_idx == sect_idx2:
                        coherence_scores.append(sent_to_sect.similarities[pair_idx])
                        sim_found = True
                        break
                
                if not sim_found:
                    coherence_scores.append(0.5)  # Default moderate similarity
            except (AttributeError, IndexError, TypeError):
                coherence_scores.append(0.5)  # Default on error
    
    # Average coherence
    coherence_score = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    
    return coherence_score

def calculate_reward(document, selected_indices, directed_sims):
    """Calculate combined reward based on coverage, diversity, and coherence"""
    if not selected_indices:
        return 0.0
    
    # Calculate component rewards
    coverage = calculate_coverage(document, selected_indices, directed_sims)
    diversity = calculate_diversity(document, selected_indices, directed_sims)
    coherence = calculate_coherence(document, selected_indices, directed_sims)
    
    # Length reward - encourage using most of the available length
    total_words = 0
    for idx in selected_indices:
        section_idx, local_idx = global_to_local_idx(document, idx)
        if section_idx is not None and local_idx is not None:
            sentence = document.sections[section_idx].sentences[local_idx]
            total_words += len(sentence.split())
    
    length_reward = min(1.0, total_words / 200.0)  # Normalized to max 200 words
    
    # Combine scores with weights
    reward = (0.4 * coverage + 
              0.3 * diversity + 
              0.2 * coherence + 
              0.1 * length_reward)
    
    return reward