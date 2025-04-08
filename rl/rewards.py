import numpy as np
from .environment import global_to_local_idx

def calculate_coverage(document, selected_indices, directed_sims):
    """Improved coverage calculation with better importance weighting"""
    if not selected_indices:
        return 0.0
    
    # Enhanced position-based importance with section weighting
    total_score = 0.0
    total_sentences = sum(len(section.sentences) for section in document.sections)
    
    # Weight first sentences in sections more heavily (topic sentences)
    for idx in selected_indices:
        section_idx, local_idx = global_to_local_idx(document, idx)
        if section_idx is not None and local_idx is not None:
            # First sentences in sections get higher weight (topic sentences)
            is_first_in_section = local_idx == 0
            is_in_first_section = section_idx == 0
            
            # Position-based importance 
            position_score = 1.0 - (idx / max(1, total_sentences - 1))
            
            # Combine factors
            importance = position_score * 0.5
            
            # Bonus for topic sentences and first section
            if is_first_in_section:
                importance += 0.3
            if is_in_first_section:
                importance += 0.2
                
            total_score += importance
    
    # Average the scores and normalize to 0-1
    coverage_score = min(1.0, total_score / len(selected_indices))
    return coverage_score

def calculate_diversity(document, selected_indices, directed_sims):
    """Improved diversity - focus on content and section distribution"""
    if len(selected_indices) < 2:
        return 1.0  # Perfect diversity with 0-1 sentences
    
    # Count sentences from each section
    section_counts = {}
    for idx in selected_indices:
        section_idx, _ = global_to_local_idx(document, idx)
        if section_idx is not None:
            section_counts[section_idx] = section_counts.get(section_idx, 0) + 1
    
    # Calculate section diversity (more even distribution = better)
    if section_counts:
        num_sections_used = len(section_counts)
        section_diversity = num_sections_used / min(len(document.sections), len(selected_indices))
    else:
        section_diversity = 0.0
    
    # Calculate position diversity (sentences from different parts of document)
    positions = []
    total_sentences = sum(len(section.sentences) for section in document.sections)
    for idx in selected_indices:
        relative_pos = idx / max(1, total_sentences - 1)
        positions.append(relative_pos)
    
    # Calculate variance of positions (higher variance = better diversity)
    if positions:
        pos_variance = np.var(positions) * 4  # Scale up variance as it's often small
        position_diversity = min(1.0, pos_variance)  # Cap at 1.0
    else:
        position_diversity = 0.0
    
    # Combine diversities
    diversity_score = 0.6 * section_diversity + 0.4 * position_diversity
    return diversity_score

def calculate_coherence(document, selected_indices, directed_sims):
    """Improved coherence - focus on logic flow and section ordering"""
    if len(selected_indices) < 2:
        return 1.0  # Perfect coherence with 0-1 sentences
    
    # Sort selected indices to get chronological order
    ordered_indices = sorted(selected_indices)
    
    # Initialize coherence components
    section_flow_score = 0.0
    sequential_score = 0.0
    
    prev_section_idx = None
    prev_local_idx = None
    
    # Track section transitions
    section_jumps = 0
    backwards_jumps = 0
    
    for idx in ordered_indices:
        section_idx, local_idx = global_to_local_idx(document, idx)
        
        if prev_section_idx is not None:
            # Score section transitions
            section_diff = section_idx - prev_section_idx
            
            if section_diff == 0:  # Same section
                section_flow_score += 1.0
                # Check if sentences are sequential
                if local_idx == prev_local_idx + 1:
                    sequential_score += 1.0
            elif section_diff == 1:  # Next section
                section_flow_score += 0.7
            elif section_diff > 1:  # Forward jump
                section_flow_score += 0.4
                section_jumps += 1
            else:  # Backward jump (negative coherence)
                section_flow_score += 0.1
                backwards_jumps += 1
        
        prev_section_idx = section_idx
        prev_local_idx = local_idx
    
    # Normalize scores
    section_flow = section_flow_score / (len(ordered_indices) - 1) if len(ordered_indices) > 1 else 1.0
    sequential_bonus = sequential_score / (len(ordered_indices) - 1) if len(ordered_indices) > 1 else 1.0
    
    # Penalties for excessive jumps
    jump_penalty = min(1.0, (section_jumps + backwards_jumps * 2) / len(ordered_indices)) if ordered_indices else 0
    
    # Calculate final coherence score
    coherence_score = (0.5 * section_flow + 0.5 * sequential_bonus) * (1.0 - 0.3 * jump_penalty)
    return max(0.1, coherence_score)  # Ensure minimum coherence

def transformer_coverage(document, selected_indices, transformer_importance_scores):
    """Calculate coverage using transformer-based importance scores"""
    if not selected_indices or transformer_importance_scores is None or len(transformer_importance_scores) == 0:
        return 0.0
        
    # Get total importance of all sentences
    total_importance = float(np.sum(transformer_importance_scores))
    if total_importance == 0:
        return 0.0
        
    # Sum importance of selected sentences
    selected_importance = 0.0
    for idx in selected_indices:
        if 0 <= idx < len(transformer_importance_scores):
            selected_importance += float(transformer_importance_scores[idx])
    
    # Calculate coverage as proportion of total importance captured
    coverage_score = min(1.0, selected_importance / total_importance)
    return coverage_score

def transformer_semantic_quality(document, selected_indices, transformer_semantic_score):
    """Use transformer semantic similarity between document and summary as quality score"""
    return transformer_semantic_score if transformer_semantic_score is not None else 0.0

def calculate_reward(document, selected_indices, directed_sims, transformer_data=None):
    """Calculate improved reward with transformer-based components"""
    if not selected_indices:
        return 0.0
    
    # Calculate base component rewards
    coverage = calculate_coverage(document, selected_indices, directed_sims)
    diversity = calculate_diversity(document, selected_indices, directed_sims)
    coherence = calculate_coherence(document, selected_indices, directed_sims)
    
    # Initialize transformer-based components
    transformer_cov = 0.0
    semantic_quality = 0.0
    
    # Add transformer-based rewards if available
    if transformer_data is not None:
        # Extract transformer data
        importance_scores = transformer_data.get('importance_scores')
        semantic_score = transformer_data.get('semantic_score')
        
        # Calculate transformer-based coverage using importance scores
        if importance_scores is not None:
            transformer_cov = transformer_coverage(document, selected_indices, importance_scores)
            
        # Use semantic quality score if available
        if semantic_score is not None:
            semantic_quality = transformer_semantic_quality(document, selected_indices, semantic_score)
    
    # Calculate length reward (encourage suitable length summaries)
    total_words = 0
    for idx in selected_indices:
        section_idx, local_idx = global_to_local_idx(document, idx)
        if section_idx is not None and local_idx is not None:
            sentence = document.sections[section_idx].sentences[local_idx]
            total_words += len(sentence.split())
    
    # Improved length reward curve - peaks at target, falls off on both sides
    target_words = 200
    min_words = 100  # Minimum acceptable summary
    
    # No reward until minimum word count
    if total_words < min_words:
        length_factor = total_words / min_words
    # Full reward at target word count
    elif total_words <= target_words:
        length_factor = 0.5 + 0.5 * (total_words / target_words)
    # Gradually decreasing reward past target (but not too harsh)
    else:
        over_ratio = total_words / target_words
        length_factor = max(0.1, 1.0 - 0.2 * (over_ratio - 1.0))
    
    # Sentence count factor - encourage using appropriate number of sentences
    ideal_sentences = 8  # Typical good summary length
    sentence_count_factor = 1.0 - 0.5 * abs(len(selected_indices) - ideal_sentences) / ideal_sentences
    sentence_count_factor = max(0.5, sentence_count_factor)  # Don't penalize too heavily
    
    # Combine scores with weights - incorporate transformer components if available
    if transformer_data:
        reward = (0.25 * coverage + 
                  0.20 * diversity + 
                  0.15 * coherence + 
                  0.15 * transformer_cov +
                  0.15 * semantic_quality +
                  0.05 * length_factor +
                  0.05 * sentence_count_factor)
    else:
        reward = (0.35 * coverage + 
                  0.25 * diversity + 
                  0.20 * coherence + 
                  0.10 * length_factor +
                  0.10 * sentence_count_factor)
    
    # Add a small random component for exploration (breaks plateaus)
    exploration_noise = np.random.normal(0, 0.02)  # Small Gaussian noise
    reward = max(0.1, min(1.0, reward + exploration_noise))  # Keep in reasonable range
    
    return reward * 5.0  # Scale up rewards to make improvements more noticeable