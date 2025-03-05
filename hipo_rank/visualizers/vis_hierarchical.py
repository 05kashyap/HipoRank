import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_hierarchical_graph(doc, sims, output_dir, threshold=0.5):
    # Create hierarchical figure
    fig = plt.figure(figsize=(14, 10))
    
    # Setup combined graph
    G = nx.DiGraph()
    
    # Add section nodes 
    for i, section in enumerate(doc.sections[:3]):  # Limit to first 3 sections
        name = f"Section {i}"
        if hasattr(section, 'id') and section.id:
            name = section.id
        G.add_node(f"sect{i}", text=name, type='section')
    
    # Add sentence nodes (first 5 sentences from first 3 sections)
    sentences = []
    total_sent = 0
    for section_idx in range(min(3, len(doc.sections))):
        section = doc.sections[section_idx]
        sent_count = min(5, len(section.sentences))
        for i in range(sent_count):
            node_id = f"s{total_sent}"
            display_text = section.sentences[i][:30] + "..." if len(section.sentences[i]) > 30 else section.sentences[i]
            G.add_node(node_id, text=display_text, type='sentence', section=section_idx)
            sentences.append((node_id, section.sentences[i]))
            
            # Add edge from sentence to its section
            G.add_edge(node_id, f"sect{section_idx}", weight=1.0, type='hierarchy')
            
            total_sent += 1
    
    # Add section-to-section edges
    sect_pairs = sims.sect_to_sect.pair_indices
    sect_sims = sims.sect_to_sect.similarities
    sect_directions = sims.sect_to_sect.directions
    
    for idx, (s1, s2) in enumerate(sect_pairs):
        if s1 >= 3 or s2 >= 3:  # Only first 3 sections
            continue
        sim_value = sect_sims[idx]
        if sim_value > threshold:
            G.add_edge(f"sect{s1}", f"sect{s2}", weight=sim_value, type='section-section')
    
    # Add sentence-to-sentence edges (within first 3 sections)
    for i in range(min(3, len(doc.sections))):
        if i >= len(sims.sent_to_sent):
            continue
        sent_sims = sims.sent_to_sent[i]
        pairs = sent_sims.pair_indices
        similarity_values = sent_sims.similarities
        
        for idx, (s1, s2) in enumerate(pairs):
            if s1 >= len(sentences) or s2 >= len(sentences):
                continue
            sim_value = similarity_values[idx]
            if sim_value > threshold:
                G.add_edge(f"s{s1}", f"s{s2}", weight=sim_value, type='sentence-sentence')
    
    # Create positions with sentences under their sections
    pos = {}
    section_positions = {}
    
    # Position sections at the top
    for i in range(min(3, len(doc.sections))):
        section_positions[f"sect{i}"] = (i * 5, 10)
        pos[f"sect{i}"] = (i * 5, 10)
    
    # Position sentences below their respective sections
    section_sentence_counts = [0] * 3
    section_sentence_counts = [0] * 3
    for node in G.nodes():
        if node.startswith('s'):
            # Add error handling to check if 'section' attribute exists
            if 'section' in G.nodes[node]:
                section_idx = G.nodes[node]['section']
                if section_idx < 3:  # Only first 3 sections
                    x_pos = section_positions[f"sect{section_idx}"][0] - 2 + section_sentence_counts[section_idx]
                    pos[node] = (x_pos, 5)
                    section_sentence_counts[section_idx] += 1
            else:
                # Handle nodes without section attribute - place them at the bottom
                print(f"Warning: Node {node} doesn't have a 'section' attribute")
                # Default position for nodes without section data
                pos[node] = (len(pos) * 2, 0)  # Place at the bottom with increasing x-coordinate
    
    
    # Draw different edge types with different styles/colors
    edges_hierarchy = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'hierarchy']
    edges_sect_sect = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'section-section']
    edges_sent_sent = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'sentence-sentence']
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=edges_hierarchy, width=1, alpha=0.5, edge_color='gray', style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=edges_sect_sect, width=2, alpha=0.7, edge_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=edges_sent_sent, width=1, alpha=0.5, edge_color='blue')
    
    # Draw nodes with different colors for sections and sentences
    section_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'section']
    sentence_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'sentence']
    
    # Create a color map for sentences based on their section
    cmap = plt.cm.get_cmap('tab10', 3)
    sent_colors = [cmap(G.nodes[n]['section']) for n in sentence_nodes]
    
    nx.draw_networkx_nodes(G, pos, nodelist=section_nodes, node_color='gold', node_size=1200, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=sentence_nodes, node_color=sent_colors, node_size=700, alpha=0.6)
    
    # Add labels
    sect_labels = {node: G.nodes[node]['text'] for node in section_nodes}
    sent_labels = {node: node for node in sentence_nodes}  # Just use node IDs for sentences to avoid clutter
    
    nx.draw_networkx_labels(G, pos, labels=sect_labels, font_size=10, font_weight='bold')
    nx.draw_networkx_labels(G, pos, labels=sent_labels, font_size=8)
    
    plt.title("HipoRank Hierarchical Document Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "hierarchical_graph_visualization.png", dpi=300, bbox_inches='tight')
    plt.close(fig)