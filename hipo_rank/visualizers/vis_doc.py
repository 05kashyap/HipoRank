import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


# Create a visualization function
def visualize_document_graph(doc, sims, output_dir,threshold=0.5):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Setup sentence-level graph (first 15 sentences from first 2 sections for clarity)
    G_sent = nx.DiGraph()
    
    # Get sentences from first 2 sections (or fewer if document has fewer sections)
    sentences = []
    section_boundaries = []
    section_count = min(2, len(doc.sections))
    
    total_sent = 0
    for section_idx in range(section_count):
        section = doc.sections[section_idx]
        # Take first 8 sentences from each section (or fewer if section has fewer sentences)
        sent_count = min(8, len(section.sentences))
        for i in range(sent_count):
            # Add node for each sentence
            node_id = f"s{total_sent}"
            # Truncate sentence for display
            display_text = section.sentences[i][:50] + "..." if len(section.sentences[i]) > 50 else section.sentences[i]
            G_sent.add_node(node_id, text=display_text, section=section_idx)
            sentences.append((node_id, section.sentences[i]))
            total_sent += 1
        section_boundaries.append(total_sent)
    
    # Add edges between sentences based on similarity and direction
    for i in range(section_count):
        sent_sims = sims.sent_to_sent[i]
        pairs = sent_sims.pair_indices
        similarity_values = sent_sims.similarities
        directions = sent_sims.directions
        
        # Filter to only include the sentences we're visualizing
        for idx, (s1, s2) in enumerate(pairs):
            if s1 >= len(sentences) or s2 >= len(sentences):
                continue
                
            sim_value = similarity_values[idx]
            if sim_value > threshold:
                direction = directions[idx]
                if direction == "forward":
                    G_sent.add_edge(f"s{s1}", f"s{s2}", weight=sim_value, direction=direction)
                elif direction == "backward":
                    G_sent.add_edge(f"s{s2}", f"s{s1}", weight=sim_value, direction=direction)
                else:  # undirected
                    G_sent.add_edge(f"s{s1}", f"s{s2}", weight=sim_value, direction=direction)
                    G_sent.add_edge(f"s{s2}", f"s{s1}", weight=sim_value, direction=direction)
    
    # Define node colors by section
    colors = ['skyblue', 'lightgreen']
    node_colors = [colors[G_sent.nodes[node]['section']] for node in G_sent.nodes]
    
    # Position the nodes with spring layout
    pos_sent = nx.spring_layout(G_sent, seed=42)
    
    # Draw the sentence graph
    nx.draw(G_sent, pos_sent, with_labels=True, node_color=node_colors, 
            node_size=1000, font_size=8, font_weight="bold", ax=ax1)
            
    # Add edge weights as labels
    edge_weights = nx.get_edge_attributes(G_sent, 'weight')
    edge_labels = {(u, v): f"{d:.2f}" for (u, v), d in edge_weights.items()}
    nx.draw_networkx_edge_labels(G_sent, pos_sent, edge_labels=edge_labels, font_size=7, ax=ax1)
    
    ax1.set_title("Sentence-level Graph (first few sentences)", fontsize=16)
    
    # Setup section-level graph
    G_sect = nx.DiGraph()
    
    # Add nodes for each section
    for i, section in enumerate(doc.sections):
        name = f"Section {i}"
        if hasattr(section, 'id') and section.id:
            name = section.id
        G_sect.add_node(f"sect{i}", text=name)
    
    # Add edges between sections
    sect_pairs = sims.sect_to_sect.pair_indices
    sect_sims = sims.sect_to_sect.similarities
    sect_directions = sims.sect_to_sect.directions
    
    for idx, (s1, s2) in enumerate(sect_pairs):
        if s1 >= len(doc.sections) or s2 >= len(doc.sections):
            continue
            
        sim_value = sect_sims[idx]
        if sim_value > threshold:
            direction = sect_directions[idx]
            if direction == "forward":
                G_sect.add_edge(f"sect{s1}", f"sect{s2}", weight=sim_value)
            elif direction == "backward":
                G_sect.add_edge(f"sect{s2}", f"sect{s1}", weight=sim_value)
            else:  # undirected
                G_sect.add_edge(f"sect{s1}", f"sect{s2}", weight=sim_value)
                G_sect.add_edge(f"sect{s2}", f"sect{s1}", weight=sim_value)
    
    # Position for section graph
    pos_sect = nx.spring_layout(G_sect, seed=42)
    
    # Draw the section graph
    nx.draw(G_sect, pos_sect, with_labels=True, node_color='lightsalmon', 
            node_size=2000, font_size=10, font_weight="bold", ax=ax2)
    
    # Add edge weights as labels
    edge_weights = nx.get_edge_attributes(G_sect, 'weight')
    edge_labels = {(u, v): f"{d:.2f}" for (u, v), d in edge_weights.items()}
    nx.draw_networkx_edge_labels(G_sect, pos_sect, edge_labels=edge_labels, font_size=9, ax=ax2)
    
    ax2.set_title("Section-level Graph", fontsize=16)
    
    # Add a title for the whole figure
    plt.suptitle(f"HipoRank Document Graph Visualization\nDocument ID: {getattr(doc, 'id', 'unknown')}", fontsize=18)
    
    plt.tight_layout()
    plt.savefig(output_dir / "hipo_graph_visualization.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Generate a text file with the sentences for reference
    with open(output_dir / "visualization_sentences.txt", "w") as f:
        f.write(f"Document ID: {getattr(doc, 'id', 'unknown')}\n\n")
        f.write("Sentences in visualization:\n")
        for node_id, text in sentences:
            f.write(f"{node_id}: {text}\n\n")
