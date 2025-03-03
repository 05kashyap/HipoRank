import json
from pathlib import Path
from hipo_rank.dataset_iterators.billsum import BillsumDataset
from tqdm import tqdm

def create_debug_file(split="test", num_docs=5):
    """Create a JSON file with document structure details for debugging"""
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading {split} split of BillSum dataset...")
    dataset = BillsumDataset(split=split)
    
    # Process a subset of documents for debugging
    docs = list(dataset)[:num_docs]
    debug_data = []
    
    print(f"Processing {len(docs)} documents for debugging...")
    for i, doc in enumerate(tqdm(docs)):
        # Document metadata
        doc_info = {
            "document_id": getattr(doc, "id", f"doc_{i}"),
            "title": dataset.docs[i].title,
            "num_sections": len(doc.sections),
            "sections": []
        }
        
        # Process each section
        for section in doc.sections:
            section_info = {
                "section_id": section.id,
                "num_sentences": len(section.sentences),
                "sentences": section.sentences[:10]  # Show first 10 sentences only to keep output manageable
            }
            doc_info["sections"].append(section_info)
        
        # Include reference summary for comparison
        doc_info["reference_summary"] = doc.reference
        
        # Original raw text (first 500 chars)
        doc_info["original_text_sample"] = dataset.docs[i].text[:500] + "..."
        
        debug_data.append(doc_info)
    
    # Save debug information
    output_file = output_dir / f"billsum_{split}_debug.json"
    with open(output_file, "w") as f:
        json.dump(debug_data, f, indent=2)
    
    print(f"Debug information saved to {output_file}")

if __name__ == "__main__":
    create_debug_file("test", num_docs=3)