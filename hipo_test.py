from pathlib import Path
import json
from tqdm import tqdm
import torch
from hipo_rank.dataset_iterators.pubmed import PubmedDataset
from hipo_rank.dataset_iterators.ilc import ILCDataset  # Import the new dataset
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.similarities.cos import CosSimilarity
from hipo_rank.directions.edge import EdgeBased
from hipo_rank.scorers.add import AddScorer
from hipo_rank.summarizers.default import DefaultSummarizer
from hipo_rank.evaluators.rouge import evaluate_rouge

# Set up parameters
pubmed_dataset_path = "data/pubmed-dataset/pubmed-dataset/test.txt"
output_dir = Path("summary_output")
output_dir.mkdir(exist_ok=True)

print(f"Cuda avail: {torch.cuda.is_available()}")

# Initialize components
embedder = BertEmbedder(
    bert_config_path="models/pacssum_models/bert_config.json",
    bert_model_path="models/pacssum_models/pytorch_model_finetuned.bin",
    bert_tokenizer="bert-base-uncased",
    cuda=torch.cuda.is_available()
)
similarity = CosSimilarity()
direction = EdgeBased()
scorer = AddScorer()

# Process PubMed dataset
def process_dataset(dataset, dataset_name, num_words=200):
    summarizer = DefaultSummarizer(num_words=num_words)
    docs = list(dataset)
    results = []
    references = []
    summaries = []

    print(f"Processing {dataset_name} dataset...")
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
            "document_id": getattr(doc.meta, "id", "unknown") if doc.meta else "unknown",
            "summary": summary
        })
        
        # For ROUGE evaluation
        summaries.append([s[0] for s in summary])
        references.append([doc.reference])

    # Save results
    with open(output_dir / f"{dataset_name}_summaries.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Evaluate with ROUGE
    rouge_result = evaluate_rouge(summaries, references)
    with open(output_dir / f"{dataset_name}_rouge_results.json", "w") as f:
        json.dump(rouge_result, f, indent=2)
    
    print(f"\nSummaries saved to {output_dir / f'{dataset_name}_summaries.json'}")
    print(f"ROUGE results saved to {output_dir / f'{dataset_name}_rouge_results.json'}")

# Process PubMed dataset
# pubmed_dataset = PubmedDataset(file_path=pubmed_dataset_path)
# process_dataset(pubmed_dataset, "pubmed", num_words=200)

# Process ILC dataset
ilc_dataset = ILCDataset(split="train")  # You can change to "test" or "validation" if needed
process_dataset(ilc_dataset, "ilc", num_words=150)  # Adjust summary length as needed