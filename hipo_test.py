from pathlib import Path
import json
from tqdm import tqdm
import torch
# Import HipoRank components
from hipo_rank.dataset_iterators.pubmed import PubmedDataset
from hipo_rank.dataset_iterators.billsum import BillsumDataset
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.similarities.cos import CosSimilarity
from hipo_rank.directions.edge import EdgeBased
from hipo_rank.scorers.add import AddScorer
from hipo_rank.summarizers.default import DefaultSummarizer
from hipo_rank.evaluators.rouge import evaluate_rouge
from rouge import Rouge 
import re
import argparse

parser = argparse.ArgumentParser(description="Run HipoRank on a dataset and generate summaries")
parser.add_argument("--dataset", type=str, default="billsum", help="Dataset to use (pubmed or billsum)")
parser.add_argument("--eval", action="store_true", help="Evaluate generated summaries with ROUGE")

args = parser.parse_args()

# Set up parameters
dataset_path = "data/pubmed-dataset/pubmed-dataset/test.txt"  # Change this to your input file
output_dir = Path("summary_output")
output_dir.mkdir(exist_ok=True)

print(f"Cuda avail: {torch.cuda.is_available()}")

# Initialize components
# dataset = PubmedDataset(file_path=dataset_path)
dataset = BillsumDataset(split="test")

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

# Save results
with open(output_dir / "summaries.json", "w") as f:
    json.dump(results, f, indent=2)


print(f"\nSummaries saved to {output_dir / 'summaries.json'}")

if args.eval:
    print("Evaluating ROUGE scores...")

    # Load the generated summaries from file to ensure we evaluate exactly what was saved
    with open(output_dir / "summaries.json", "r") as f:
        results = json.load(f)

    # Create a mapping from document ID to generated summary
    id_to_summary = {item["document_id"]: item["summary"] for item in results}

    # Prepare generated summaries and reference summaries for evaluation
    generated_summaries = []
    reference_summaries = []

    for doc in docs:
        doc_id = getattr(doc, "id", "unknown")
        
        if doc_id in id_to_summary:
            # Get the generated summary which is a list of lists [sentence, score, ...]
            summary_list = id_to_summary[doc_id]
            
            # Extract just the sentences (first element of each sublist)
            summary_sentences = []
            for item in summary_list:
                if isinstance(item, list) and len(item) > 0:
                    # Clean the sentence text
                    sentence = item[0].strip()
                    if sentence:
                        summary_sentences.append(sentence)
            
            # Get the reference summary from the document
            reference_summary = doc.reference
            
            # Add to the lists for evaluation
            generated_summaries.append(summary_sentences)
            reference_summaries.append([reference_summary])  # Format needed by evaluate_rouge

    print(f"Evaluating ROUGE for {len(generated_summaries)} document summaries...")

    # Create simple text versions of summaries and references for the rouge package
    generated_texts = [' '.join(summary) for summary in generated_summaries]
    reference_texts = [' '.join(ref[0]) for ref in reference_summaries]

    # Initialize Rouge
    rouge = Rouge()

    # Calculate ROUGE scores
    rouge_results = {}
    for i in range(len(generated_texts)):
        if generated_texts[i] and reference_texts[i]:
            try:
                scores = rouge.get_scores(generated_texts[i], reference_texts[i])[0]
                for metric, values in scores.items():
                    if metric not in rouge_results:
                        rouge_results[metric] = {}
                    for k, v in values.items():
                        key = f"{metric}-{k}"
                        if key not in rouge_results:
                            rouge_results[key] = []
                        rouge_results[key].append(v)
            except Exception as e:
                print(f"Error calculating ROUGE for document {i}: {e}")

    # Average the scores
    final_scores = {}
    for metric, values in rouge_results.items():
        final_scores[metric] = sum(values) / len(values) if values else 0

    # Save ROUGE results
    with open(output_dir / "rouge_results.json", "w") as f:
        json.dump(final_scores, f, indent=2)

    # Print a summary of the ROUGE results
    print("\nROUGE Evaluation Results:")
    for metric, score in final_scores.items():
        print(f"{metric}: {score:.4f}")

    print(f"\nDetailed ROUGE evaluation results saved to {output_dir / 'rouge_results.json'}")