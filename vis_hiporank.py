import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from hipo_rank.dataset_iterators.billsum import BillsumDataset
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.similarities.cos import CosSimilarity
from hipo_rank.directions.edge import EdgeBased
from hipo_rank.visualizers.vis_doc import visualize_document_graph
from hipo_rank.visualizers.vis_hierarchical import visualize_hierarchical_graph

# Create output directory
output_dir = Path("visualization_output")
output_dir.mkdir(exist_ok=True)

# Load a single document from the dataset
dataset = BillsumDataset(split="test")
docs = list(dataset)[:1]  # Just take the first document

# Initialize the pipeline components
embedder = BertEmbedder(
    bert_config_path="models/pacssum_models/bert_config.json",
    bert_model_path="models/pacssum_models/pytorch_model_finetuned.bin",
    bert_tokenizer="bert-base-uncased",
    cuda=torch.cuda.is_available()
)
similarity = CosSimilarity()
direction = EdgeBased()

# Process the document
print("Embedding document...")
doc = docs[0]
embeddings = embedder.get_embeddings(doc)

print("Calculating similarities...")
sims = similarity.get_similarities(embeddings)

print("Applying direction strategy...")
directed_sims = direction.update_directions(sims)

# Generate the visualization with a similarity threshold
# visualize_document_graph(doc, directed_sims, output_dir,threshold=0.4)

# print(f"Visualization saved to {output_dir / 'hipo_graph_visualization.png'}")
# print(f"Sentence reference saved to {output_dir / 'visualization_sentences.txt'}")

visualize_hierarchical_graph(doc, directed_sims, output_dir,threshold=0.4)
print(f"Hierarchical visualization saved to {output_dir / 'hierarchical_graph_visualization.png'}")