import os
from pathlib import Path
import argparse
import torch
import json
from tqdm import tqdm
from rouge import Rouge

from rl.agents import RLHipoRankAgent
from rl.environment import HipoRankEnvironment, get_valid_actions, global_to_local_idx
from rl.states import create_state

from hipo_rank.dataset_iterators.pubmed import PubmedDataset
from hipo_rank.dataset_iterators.billsum import BillsumDataset
from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.similarities.cos import CosSimilarity
from hipo_rank.directions.edge import EdgeBased
from hipo_rank.scorers.add import AddScorer

def load_agent(checkpoint_path, state_size, action_size=100):
    """Load a trained RL-HipoRank agent"""
    agent = RLHipoRankAgent(state_size=state_size, action_size=action_size)
    agent.load_model(checkpoint_path)
    return agent

def generate_summaries(agent, documents, embedder, similarity, direction, max_words=200):
    """Generate summaries using the trained RL agent"""
    results = []
    
    for doc in tqdm(documents, desc="Generating Summaries"):
        # Get HipoRank features
        embeddings = embedder.get_embeddings(doc)
        similarities = similarity.get_similarities(embeddings)
        directed_sims = direction.update_directions(similarities)
        
        # Count total sentences in the document (for action_size)
        total_sentences = 0
        for section in doc.sections:
            total_sentences += len(section.sentences)
            
        # Create environment
        env = HipoRankEnvironment(doc, embeddings, similarities, directed_sims, max_words=max_words)
        
        # Generate summary
        state = env.reset()
        done = False
        
        while not done:
            valid_actions = env.available_actions
            if not valid_actions:
                break
                
            action = agent.get_action(state, valid_actions)
            if action is None:
                break
                
            state, reward, done = env.step(action)
        
        # Format summary
        summary = []
        for idx in env.current_summary:
            section_idx, local_idx = global_to_local_idx(doc, idx)
            if section_idx is not None and local_idx is not None:
                sentence = doc.sections[section_idx].sentences[local_idx]
                summary.append([sentence, 1.0])  # Format: [sentence, score]
        
        results.append({
            "document_id": getattr(doc, "id", "unknown"),
            "summary": summary
        })
    
    return results

def evaluate_rouge(generated_summaries, documents):
    """Evaluate generated summaries using ROUGE"""
    # Create mapping of document IDs to generated summaries
    id_to_summary = {item["document_id"]: item["summary"] for item in generated_summaries}
    
    # Prepare generated summaries and reference summaries
    gen_texts = []
    ref_texts = []
    
    for doc in documents:
        doc_id = getattr(doc, "id", "unknown")
        
        if doc_id in id_to_summary:
            # Extract just the sentences from the summary
            summary_sentences = []
            for item in id_to_summary[doc_id]:
                if isinstance(item, list) and len(item) > 0:
                    sentence = item[0].strip()
                    if sentence:
                        summary_sentences.append(sentence)
            
            # Get reference summary
            reference_summary = doc.reference
            
            # Add to evaluation lists
            gen_text = ' '.join(summary_sentences)
            ref_text = ' '.join(reference_summary)
            
            if gen_text and ref_text:
                gen_texts.append(gen_text)
                ref_texts.append(ref_text)
    
    # Calculate ROUGE scores
    rouge = Rouge()
    scores = {}
    
    for i in range(len(gen_texts)):
        try:
            doc_scores = rouge.get_scores(gen_texts[i], ref_texts[i])[0]
            for metric, values in doc_scores.items():
                for k, v in values.items():
                    key = f"{metric}-{k}"
                    if key not in scores:
                        scores[key] = []
                    scores[key].append(v)
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
    
    # Average the scores
    final_scores = {}
    for metric, values in scores.items():
        final_scores[metric] = sum(values) / len(values) if values else 0
    
    return final_scores

def main():
    parser = argparse.ArgumentParser(description="Test RL-HipoRank summarization model")
    parser.add_argument("--dataset", type=str, default="billsum", choices=["pubmed", "billsum"],
                        help="Dataset to use (pubmed or billsum)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="rl_summary_output",
                        help="Directory to save results")
    parser.add_argument("--max-words", type=int, default=200,
                        help="Maximum number of words in summary")
    parser.add_argument("--num-docs", type=int, default=10,
                        help="Number of documents to test on")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup dataset
    if args.dataset == "pubmed":
        dataset_path = "data/pubmed-dataset/pubmed-dataset/test.txt"
        dataset = PubmedDataset(file_path=dataset_path)
    else:
        dataset = BillsumDataset(split="test")
    
    # Initialize HipoRank components
    embedder = BertEmbedder(
        bert_config_path="models/pacssum_models/bert_config.json",
        bert_model_path="models/pacssum_models/pytorch_model_finetuned.bin",
        bert_tokenizer="bert-base-uncased",
        cuda=torch.cuda.is_available()
    )
    similarity = CosSimilarity()
    direction = EdgeBased()
    
    print(f"Loading dataset: {args.dataset}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    
    # Load documents
    all_documents = list(dataset)
    documents = all_documents[:args.num_docs]  # Limit to specified number of docs
    print(f"Testing on {len(documents)} documents")
    
    # Create a sample state to determine state size
    sample_doc = documents[0]
    sample_embeddings = embedder.get_embeddings(sample_doc)
    sample_similarities = similarity.get_similarities(sample_embeddings)
    sample_directed_sims = direction.update_directions(sample_similarities)
    sample_state = create_state(sample_doc, sample_embeddings, sample_directed_sims)
    state_size = len(sample_state)
    
    # Count total sentences for action space size
    total_sentences = 0
    for section in sample_doc.sections:
        total_sentences += len(section.sentences)
    action_size = total_sentences
    
    # Load trained agent
    print(f"Loading model from {args.model_path}")
    agent = load_agent(args.model_path, state_size, action_size)
    
    # Generate summaries
    print("Generating summaries...")
    summaries = generate_summaries(agent, documents, embedder, similarity, direction, max_words=args.max_words)
    
    # Save summaries
    with open(output_dir / "rl_test_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)
    
    # Evaluate with ROUGE
    print("Evaluating with ROUGE...")
    rouge_scores = evaluate_rouge(summaries, documents)
    
    # Save ROUGE results
    with open(output_dir / "rl_rouge_results.json", "w") as f:
        json.dump(rouge_scores, f, indent=2)
    
    # Print ROUGE scores
    print("\nROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"{metric}: {score:.4f}")
    
    print(f"\nSummaries saved to {output_dir / 'rl_test_summaries.json'}")
    print(f"ROUGE results saved to {output_dir / 'rl_rouge_results.json'}")

if __name__ == "__main__":
    main()