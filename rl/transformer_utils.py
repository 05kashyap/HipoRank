import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple

class TransformerSentenceEncoder:
    """Uses a pre-trained transformer model to encode sentences into rich representations"""
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=512, device=None):
        """Initialize transformer encoder with specified model"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading transformer model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode
        self.max_length = max_length
        
    def encode_sentences(self, sentences: List[str]) -> torch.Tensor:
        """Encode a list of sentences into embeddings using the transformer model"""
        # Handle empty input
        if not sentences:
            return torch.zeros((0, self.model.config.hidden_size), device=self.device)
            
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            
            # Use [CLS] token embedding as sentence representation
            # (first token of last hidden state)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings
        
    def calculate_sentence_similarities(self, sentences: List[str]) -> torch.Tensor:
        """Calculate pairwise cosine similarities between sentences"""
        # Encode sentences
        embeddings = self.encode_sentences(sentences)
        
        if len(embeddings) == 0:
            return torch.zeros((0, 0), device=self.device)
            
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        # Calculate pairwise cosine similarities
        similarities = torch.mm(embeddings_norm, embeddings_norm.transpose(0, 1))
        
        return similarities
    
    def get_sentence_importance(self, sentences: List[str]) -> np.ndarray:
        """Calculate importance scores for sentences based on centrality"""
        # Get similarities
        similarities = self.calculate_sentence_similarities(sentences)
        
        if len(similarities) == 0:
            return np.array([])
        
        # Convert to numpy for easier manipulation
        sim_matrix = similarities.cpu().numpy()
        
        # Calculate centrality scores (average similarity to other sentences)
        centrality = np.mean(sim_matrix, axis=1)
        
        # Normalize to [0, 1] range
        if len(centrality) > 0 and np.max(centrality) > np.min(centrality):
            centrality = (centrality - np.min(centrality)) / (np.max(centrality) - np.min(centrality))
            
        return centrality

    def get_document_representation(self, sentences: List[str]) -> torch.Tensor:
        """Get a document-level representation by averaging sentence embeddings"""
        embeddings = self.encode_sentences(sentences)
        if len(embeddings) == 0:
            return torch.zeros(self.model.config.hidden_size, device=self.device)
        return torch.mean(embeddings, dim=0)
    
    def calculate_summary_quality(self, document_sentences: List[str], summary_sentences: List[str]) -> float:
        """Calculate the quality of a summary based on semantic similarity to the document"""
        if not summary_sentences or not document_sentences:
            return 0.0
            
        # Get document and summary representations
        doc_embedding = self.get_document_representation(document_sentences)
        summary_embedding = self.get_document_representation(summary_sentences)
        
        # Calculate cosine similarity
        doc_norm = doc_embedding / doc_embedding.norm()
        summary_norm = summary_embedding / summary_embedding.norm()
        
        similarity = torch.dot(doc_norm, summary_norm).item()
        
        return max(0.0, min(1.0, similarity))  # Clip to [0, 1]

    def get_contextual_sentence_features(self, sentences: List[str]) -> Dict[str, np.ndarray]:
        """Extract contextual features from sentences using transformer embeddings"""
        if not sentences:
            return {
                "sentence_embeddings": np.array([]),
                "importance_scores": np.array([]),
                "position_scores": np.array([])
            }
        
        # Get sentence embeddings
        embeddings = self.encode_sentences(sentences).cpu().numpy()
        
        # Calculate importance based on centrality
        importance_scores = self.get_sentence_importance(sentences)
        
        # Calculate position bias (favor introductory and concluding sentences)
        n = len(sentences)
        position_scores = np.zeros(n)
        for i in range(n):
            # Higher scores for beginning and end of document
            relative_pos = i / max(1, n - 1)
            if relative_pos <= 0.2:  # First 20% of document
                position_scores[i] = 1.0 - relative_pos * 2  # Linearly decreasing
            elif relative_pos >= 0.8:  # Last 20% of document
                position_scores[i] = (relative_pos - 0.8) * 5  # Linearly increasing
            else:
                position_scores[i] = 0.2  # Middle has lower importance
        
        return {
            "sentence_embeddings": embeddings,
            "importance_scores": importance_scores,
            "position_scores": position_scores
        }
