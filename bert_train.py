import torch
import numpy as np
import json
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification, 
    AdamW,
    get_linear_schedule_with_warmup
)
from rouge import Rouge
import sys
import os

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hipo_rank.dataset_iterators.billsum import BillsumDataset
from hipo_rank import Document

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Configuration
OUTPUT_DIR = Path("bert_extractive_output")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 0
MAX_SEQ_LENGTH = 512
SUMMARY_LENGTH = 200  # Number of words for generated summaries

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SentenceLabelingDataset(Dataset):
    def __init__(self, documents, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.inputs = []
        self.labels = []
        self.doc_indices = []  # Track which document each sentence belongs to
        
        print("Preparing dataset...")
        for doc_idx, doc in enumerate(tqdm(documents)):
            self._process_document(doc, doc_idx)
    
    def _process_document(self, doc, doc_idx):
        # Get all sentences from the document
        all_sentences = []
        section_indices = []  # Track which section each sentence belongs to
        local_indices = []    # Track local index within section
        
        for sect_idx, section in enumerate(doc.sections):
            for local_idx, sentence in enumerate(section.sentences):
                all_sentences.append(sentence)
                section_indices.append(sect_idx)
                local_indices.append(local_idx)
        
        # Find which sentences appear in the reference summary
        reference_text = " ".join(doc.reference)
        labels = []
        
        # Enhanced matching algorithm
        rouge_calc = Rouge()
        
        for sentence in all_sentences:
            if len(sentence.split()) < 3:  # Skip very short sentences
                labels.append(0)
                continue
                
            # Try direct string matching first (exact or substring)
            if sentence in reference_text or reference_text.find(sentence) != -1:
                labels.append(1)
                continue
                
            # If not direct match, try ROUGE similarity
            try:
                scores = rouge_calc.get_scores(sentence, reference_text)[0]
                score = (scores["rouge-1"]["f"] + scores["rouge-2"]["f"] + scores["rouge-l"]["f"]) / 3
                
                # Threshold for inclusion in summary
                if score > 0.4:  # Adjust threshold as needed
                    labels.append(1)
                else:
                    labels.append(0)
            except Exception:
                # Fall back to 0 if scoring fails
                labels.append(0)
        
        # Encode each sentence for BERT
        for i, sentence in enumerate(all_sentences):
            encoded = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            self.inputs.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'section_idx': section_indices[i],
                'local_idx': local_indices[i],
                'text': sentence,
                'doc_idx': doc_idx
            })
            self.labels.append(labels[i])
            self.doc_indices.append(doc_idx)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'],
            'attention_mask': self.inputs[idx]['attention_mask'],
            'section_idx': self.inputs[idx]['section_idx'],
            'local_idx': self.inputs[idx]['local_idx'],
            'text': self.inputs[idx]['text'],
            'doc_idx': self.inputs[idx]['doc_idx'],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# The rest of the functions remain mostly the same

def train_model(model, train_dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)  # Changed from 'labels' to 'label'
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_dataloader)

def evaluate_model(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_doc_indices = []
    all_section_indices = []
    all_local_indices = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)  # Changed from 'labels' to 'label'
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Get probability of class 1 (sentence should be in summary)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(probs.cpu().numpy())  # Store probabilities instead of binary predictions
            all_labels.extend(labels.cpu().numpy())
            all_doc_indices.extend(batch['doc_idx'].cpu().numpy())
            all_section_indices.extend(batch['section_idx'].cpu().numpy())
            all_local_indices.extend(batch['local_idx'].cpu().numpy())
    
    # Calculate accuracy
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    accuracy = (np.array(binary_preds) == np.array(all_labels)).mean()
    
    return total_loss / len(eval_dataloader), accuracy, all_preds, all_doc_indices, all_section_indices, all_local_indices

def generate_summaries(docs, predictions_by_doc):
    summaries = []
    
    for doc_idx, doc in enumerate(docs):
        if doc_idx not in predictions_by_doc:
            summaries.append([])  # Empty summary if no predictions
            continue
        
        predictions = predictions_by_doc[doc_idx]
        sentences_with_scores = []
        
        # Collect sentences with their scores (probability of being in summary)
        for section_idx, section in enumerate(doc.sections):
            for local_idx, sentence in enumerate(section.sentences):
                key = (section_idx, local_idx)
                if key in predictions:
                    score = predictions[key]
                    sentences_with_scores.append((sentence, score, section_idx, local_idx))
        
        # Sort by score (descending)
        sentences_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select sentences until we reach word limit
        selected_sentences = []
        word_count = 0
        
        for sentence, score, section_idx, local_idx in sentences_with_scores:
            words = sentence.split()
            sentence_word_count = len(words)
            
            if word_count + sentence_word_count <= SUMMARY_LENGTH:
                selected_sentences.append((sentence, score, section_idx, local_idx))
                word_count += sentence_word_count
            
            if word_count >= SUMMARY_LENGTH:
                break
        
        # Re-sort selected sentences by their position in the document
        selected_sentences.sort(key=lambda x: (x[2], x[3]))
        
        # Extract just the sentences
        summary = [item[0] for item in selected_sentences]
        summaries.append(summary)
    
    return summaries

def evaluate_rouge(predicted_summaries, reference_summaries):
    rouge = Rouge()
    
    # Make sure we have content to evaluate
    valid_pairs = []
    for pred, ref in zip(predicted_summaries, reference_summaries):
        if pred and ref:
            valid_pairs.append((pred, ref[0]))
    
    if not valid_pairs:
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
    
    pred_texts = [' '.join(summary) for summary, _ in valid_pairs]
    ref_texts = [' '.join(summary) for _, summary in valid_pairs]
    
    scores = rouge.get_scores(pred_texts, ref_texts, avg=True)
    return scores

def main():
    # Load dataset
    print("Loading BillSum dataset...")
    # train_dataset = BillsumDataset(split="train")
    test_dataset = BillsumDataset(split="test")
    
    train_docs = list(test_dataset)[:5]
    test_docs = list(test_dataset)[:5]
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_data = SentenceLabelingDataset(train_docs, tokenizer, MAX_SEQ_LENGTH)
    
    # Split training data into train and validation
    train_indices, val_indices = train_test_split(
        range(len(train_data)), 
        test_size=0.1, 
        random_state=seed
    )
    
    # Create data loaders
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_dataloader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler
    )
    
    val_dataloader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        sampler=val_sampler
    )
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        avg_train_loss = train_model(model, train_dataloader, optimizer, scheduler, device)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Evaluate on validation set
        avg_val_loss, val_accuracy, *_ = evaluate_model(model, val_dataloader, device)
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save model checkpoint
        model_path = OUTPUT_DIR / f"model_epoch_{epoch+1}"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved to {model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    
    # Process test data
    test_data = SentenceLabelingDataset(test_docs, tokenizer, MAX_SEQ_LENGTH)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    _, test_accuracy, test_predictions, doc_indices, section_indices, local_indices = evaluate_model(model, test_dataloader, device)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Organize predictions by document
    predictions_by_doc = {}
    
    for i in range(len(test_predictions)):
        doc_idx = doc_indices[i]
        section_idx = section_indices[i]
        local_idx = local_indices[i]
        
        if doc_idx not in predictions_by_doc:
            predictions_by_doc[doc_idx] = {}
        
        predictions_by_doc[doc_idx][(section_idx, local_idx)] = test_predictions[i]
    
    # Generate summaries
    predicted_summaries = generate_summaries(test_docs, predictions_by_doc)
    reference_summaries = [[doc.reference] for doc in test_docs]
    
    # Evaluate ROUGE
    rouge_scores = evaluate_rouge(predicted_summaries, reference_summaries)
    
    print("\nROUGE Evaluation Results:")
    for metric, scores in rouge_scores.items():
        for score_type, value in scores.items():
            print(f"{metric}-{score_type}: {value:.4f}")
    
    # Save results
    with open(OUTPUT_DIR / "rouge_results.json", "w") as f:
        json.dump(rouge_scores, f, indent=2)
    
    # Save some example summaries
    examples = []
    for i in range(min(10, len(predicted_summaries))):
        examples.append({
            "predicted_summary": predicted_summaries[i],
            "reference_summary": reference_summaries[i][0]
        })
    
    with open(OUTPUT_DIR / "example_summaries.json", "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()