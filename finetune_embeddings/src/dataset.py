from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import BertTokenizer
import spacy
import logging
import os
from tqdm.auto import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

logger = logging.getLogger(__name__)

class EmbeddingDataset(Dataset):
    """Dataset with pre-computed embeddings support"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 512,
        embedding_dir: Optional[str] = "embeddings"
    ) -> None:
        """
        Initialize dataset with embedding support.
        
        Args:
            texts: List of input texts
            labels: List of labels
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            embedding_dir: Directory to cache embeddings
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding_dir = Path(embedding_dir) if embedding_dir else None
        
        # Create directories
        if self.embedding_dir:
            self.embedding_dir.mkdir(parents=True, exist_ok=True)
            
        # Store texts for MLM
        self.texts = texts
        
        # Try to load cached embeddings
        if self.embedding_dir:
            cached_embeddings = self.embedding_dir / "cached_embeddings.pt"
            if cached_embeddings.exists():
                logger.info("Loading cached embeddings...")
                self.features = torch.load(cached_embeddings)
            else:
                logger.info("Computing and caching embeddings...")
                self.features = self._extract_features()
                torch.save(self.features, cached_embeddings)
        
        # Initialize spaCy for syntactic features
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'lemmatizer'])
        
        # Tokenize texts
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Extract features if no caching
        if not self.embedding_dir:
            self.features = self._extract_features()

    def _load_or_extract_features(self) -> np.ndarray:
        """Load cached features or extract new ones."""
        bert_embedding_file = self.embedding_dir / "bert_embeddings.npy"
        syntactic_feature_file = self.embedding_dir / "syntactic_features.npy"
        
        if bert_embedding_file.exists() and syntactic_feature_file.exists():
            logger.info("Loading cached embeddings...")
            contextual_features = np.load(bert_embedding_file)
            syntactic_features = np.load(syntactic_feature_file)
            return np.hstack([contextual_features, syntactic_features])
        
        logger.info("Computing embeddings...")
        
        # Extract BERT embeddings
        contextual_features = self._extract_contextual_features()
        
        # Extract syntactic features
        syntactic_features = self._extract_syntactic_features()
        
        # Save features
        np.save(bert_embedding_file, contextual_features)
        np.save(syntactic_feature_file, syntactic_features)
        
        return np.hstack([contextual_features, syntactic_features])

    def _extract_contextual_features(self, batch_size: int = 32) -> np.ndarray:
        """Extract BERT contextual embeddings."""
        features = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load BERT model
        from transformers import BertModel
        model = BertModel.from_pretrained(self.tokenizer.name_or_path).to(device)
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.texts), batch_size), desc="BERT embeddings"):
                batch_texts = self.texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(device)
                
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.extend(embeddings)
        
        del model
        torch.cuda.empty_cache()
        return np.array(features)

    def _extract_syntactic_features(self) -> np.ndarray:
        """Extract spaCy-based syntactic features."""
        features = []
        for text in tqdm(self.texts, desc="Syntactic features"):
            doc = self.nlp(str(text))
            pos_tags = [token.pos_ for token in doc]
            features.append([
                len(doc),  # Document length
                len(set(pos_tags)) / max(len(pos_tags), 1),  # POS diversity
                pos_tags.count('NOUN') / max(len(pos_tags), 1),  # Noun ratio
                pos_tags.count('VERB') / max(len(pos_tags), 1),  # Verb ratio
            ])
        return np.array(features)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            key: val[idx].clone() for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        if self.features is not None:
            item['features'] = torch.tensor(self.features[idx], dtype=torch.float32)
        
        return item

def create_datasets(
    texts: List[str],
    labels: List[int],
    tokenizer: BertTokenizer,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_length: int = 512,
    embedding_dir: Optional[str] = "embeddings",
    seed: int = 42
) -> tuple[EmbeddingDataset, EmbeddingDataset, EmbeddingDataset]:
    """
    Create train, validation and test datasets with stratification.
    
    Args:
        texts: List of input texts
        labels: List of labels
        tokenizer: BERT tokenizer
        val_split: Validation split ratio
        test_split: Test split ratio
        max_length: Maximum sequence length
        embedding_dir: Directory to cache embeddings
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from sklearn.model_selection import train_test_split
    
    # First split off test set
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=test_split,
        random_state=seed,
        stratify=labels
    )
    
    # Then split training into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=val_split / (1 - test_split),
        random_state=seed,
        stratify=train_labels
    )
    
    # Create datasets
    train_dataset = EmbeddingDataset(
        train_texts, train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        embedding_dir=embedding_dir
    )
    
    val_dataset = EmbeddingDataset(
        val_texts, val_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        embedding_dir=embedding_dir
    )
    
    test_dataset = EmbeddingDataset(
        test_texts, test_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        embedding_dir=embedding_dir
    )
    
    return train_dataset, val_dataset, test_dataset
