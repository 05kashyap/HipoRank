import numpy as np
from .environment import global_to_local_idx

def calculate_coverage(document, selected_indices, directed_sims):
    """Simplified coverage calculation - focus on position-based importance"""
    if not selected_indices:
        return 0.0
    
    # Simple position-based importance (sentences at beginning are more important)
    total_score = 0.0
    total_sentences = sum(len(section.sentences) for section in document.sections)
    
    for idx in selected_indices:
        section_idx, local_idx = global_to_local_idx(document, idx)
        if section_idx is not None and local_idx is not None:
            # Position-based importance score
            global_pos = idx / max(1, total_sentences - 1)
            section_pos = section_idx / max(1, len(document.sections) - 1)
            local_pos = local_idx / max(1, len(document.sections[section_idx].sentences) - 1)
            
            # Lower position = higher importance (1 - position)
            importance = 0.6 * (1 - section_pos) + 0.4 * (1 - local_pos)
            total_score += importance
    
    # Average the scores
    coverage_score = total_score / len(selected_indices) if selected_indices else 0.0
    return coverage_score

def calculate_diversity(document, selected_indices, directed_sims):
    """Simplified diversity - based on section and position distribution"""
    if len(selected_indices) < 2:
        return 1.0  # Perfect diversity with 0-1 sentences
    
    # Count sentences from each section
    section_counts = {}
    for idx in selected_indices:
        section_idx, _ = global_to_local_idx(document, idx)
        if section_idx is not None:
            section_counts[section_idx] = section_counts.get(section_idx, 0) + 1
    
    # More even distribution across sections = higher diversity
    max_section_count = max(section_counts.values()) if section_counts else 1
    section_diversity = 1.0 - (max_section_count / len(selected_indices))
    
    # Position diversity - sentences from different positions are more diverse
    positions = []
    total_sentences = sum(len(section.sentences) for section in document.sections)
    for idx in selected_indices:
        relative_pos = idx / max(1, total_sentences - 1)
        positions.append(relative_pos)
    
    # Divide position range (0-1) into 4 quarters and count sentences in each
    quarter_counts = [0, 0, 0, 0]
    for pos in positions:
        quarter = min(3, int(pos * 4))
        quarter_counts[quarter] += 1
    
    # More even distribution across quarters = higher diversity
    max_quarter_count = max(quarter_counts)
    position_diversity = 1.0 - (max_quarter_count / len(selected_indices))
    
    # Combine diversities
    diversity_score = 0.5 * section_diversity + 0.5 * position_diversity
    return diversity_score

def calculate_coherence(document, selected_indices, directed_sims):
    """Simplified coherence - focus on sequential ordering and section consistency"""
    if len(selected_indices) < 2:
        return 1.0  # Perfect coherence with 0-1 sentences
    
    # Sort selected indices to get sentences in order
    ordered_indices = sorted(selected_indices)
    
    # Check for section consistency and sequence gaps
    coherence_score = 0.0
    prev_section_idx = None
    prev_local_idx = None
    
    for idx in ordered_indices:
        section_idx, local_idx = global_to_local_idx(document, idx)
        
        if prev_section_idx is not None:
            # Reward keeping same section (section continuity)
            if section_idx == prev_section_idx:
                coherence_score += 0.5
                
                # Additional reward for sequential sentences within section
                if local_idx == prev_local_idx + 1:
                    coherence_score += 0.5
            
            # Smaller reward for moving to next section
            elif section_idx == prev_section_idx + 1:
                coherence_score += 0.3
        
        prev_section_idx = section_idx
        prev_local_idx = local_idx
    
    # Normalize by maximum possible score
    max_score = (len(ordered_indices) - 1) * 1.0  # Max if all sentences were sequential in same section
    coherence_score = coherence_score / max_score if max_score > 0 else 1.0
    
    return coherence_score

def calculate_reward(document, selected_indices, directed_sims):
    """Calculate combined reward using simplified metrics"""
    if not selected_indices:
        return 0.0
    
    # Calculate component rewards
    coverage = calculate_coverage(document, selected_indices, directed_sims)
    diversity = calculate_diversity(document, selected_indices, directed_sims)
    coherence = calculate_coherence(document, selected_indices, directed_sims)
    
    # Length reward - encourage using most of the available length but penalize exceeding
    total_words = 0
    for idx in selected_indices:
        section_idx, local_idx = global_to_local_idx(document, idx)
        if section_idx is not None and local_idx is not None:
            sentence = document.sections[section_idx].sentences[local_idx]
            total_words += len(sentence.split())
    
    # Length reward that peaks at the target word count
    target_words = 200
    length_ratio = total_words / target_words
    length_reward = max(0.0, 1.0 - abs(1.0 - length_ratio))
    
    # Combine scores with weights
    reward = (0.35 * coverage + 
              0.25 * diversity + 
              0.20 * coherence + 
              0.20 * length_reward)
    
    return reward