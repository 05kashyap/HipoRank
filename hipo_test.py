from pathlib import Path
import json
from tqdm import tqdm
import torch
# Import HipoRank components
from hipo_rank.dataset_iterators.pubmed import PubmedDataset
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.similarities.cos import CosSimilarity
from hipo_rank.directions.edge import EdgeBased
from hipo_rank.scorers.add import AddScorer
from hipo_rank.summarizers.default import DefaultSummarizer
from hipo_rank.evaluators.rouge import evaluate_rouge

# Set up parameters
dataset_path = "data/pubmed-dataset/pubmed-dataset/test.txt"  # Change this to your input file
output_dir = Path("summary_output")
output_dir.mkdir(exist_ok=True)

print(f"Cuda avail: {torch.cuda.is_available()}")

# Initialize components
dataset = PubmedDataset(file_path=dataset_path)
embedder = BertEmbedder(
    bert_config_path="models/pacssum_models/bert_config.json",
    bert_model_path="models/pacssum_models/pytorch_model_finetuned.bin",
    bert_tokenizer="bert-base-uncased",
    cuda=torch.cuda.is_available()
)
similarity = CosSimilarity()
direction = EdgeBased()
scorer = AddScorer()
summarizer = DefaultSummarizer(num_words=200)  # Adjust summary length as needed

# Process documents and generate summaries
docs = list(dataset)
results = []

print("Embedding documents...")
embeddings = [embedder.get_embeddings(doc) for doc in tqdm(docs)]

print("Calculating similarities...")
similarities = [similarity.get_similarities(e) for e in embeddings]

print("Applying direction strategy...")
directed_sims = [direction.update_directions(s) for s in similarities]

print("Generating summaries...")
for sim, doc in zip(directed_sims, docs):
    scores = scorer.get_scores(sim)
    summary = summarizer.get_summary(doc, scores)
    
    results.append({
        "document_id": getattr(doc, "id", "unknown"),
        "summary": summary
    })
    
    # Print the summary text
    print("\n===== SUMMARY =====")
    for sent, _ in summary:
        print(sent)

# Save results
with open(output_dir / "summaries.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSummaries saved to {output_dir / 'summaries.json'}")