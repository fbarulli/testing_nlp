from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm.auto import tqdm
import random
import numpy as np
import numpy.typing as npt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling"""
    def __init__(
        self,
        texts: List[str],
        tokenizer: BertTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15
    ) -> None:
        """
        Initialize MLM Dataset.
        
        Args:
            texts: List of input texts
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            mlm_probability: Probability of masking a token
        """
        logger.info("Initializing MLMDataset")
        try:
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.mlm_probability = mlm_probability
            
            # Tokenize all texts
            with tqdm(total=1, desc="Tokenizing texts", leave=False) as pbar:
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                pbar.update(1)
            
            logger.info("MLMDataset initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MLMDataset: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.encodings.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item with masked tokens.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        try:
            # Get the original tokens
            item = {key: val[idx].clone() for key, val in self.encodings.items()}
            input_ids = item["input_ids"]
            
            # Create labels before masking (for loss computation)
            labels = input_ids.clone()
            
            # Create masking mask
            probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(),
                already_has_special_tokens=True
            )
            probability_matrix.masked_fill_(
                torch.tensor(special_tokens_mask, dtype=torch.bool),
                value=0.0
            )
            
            # Mask tokens
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens
            
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            
            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            input_ids[indices_random] = random_words[indices_random]
            
            item["input_ids"] = input_ids
            item["labels"] = labels
            
            return item
        except Exception as e:
            logger.error(f"Error getting item at index {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class MLMTrainer:
    """Trainer for Masked Language Modeling"""
    def __init__(
        self,
        model: BertForMaskedLM,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        embeddings: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initialize MLM Trainer.
        
        Args:
            model: BERT model for masked language modeling
            train_loader: DataLoader for training data
            optimizer: Optimizer for training
            device: Device to run on (cuda/cpu)
            config: Training configuration
        """
        logger.info("Initializing MLMTrainer")
        try:
            self.model = model
            self.train_loader = train_loader
            self.optimizer = optimizer
            self.device = device
            self.config = config
            
            # Initialize training monitoring
            self.train_losses: List[float] = []
            self.perplexities: List[float] = []
            self.embeddings = embeddings
            
            logger.info("MLMTrainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MLMTrainer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Starting MLM training epoch")
        try:
            self.model.train()
            total_loss = 0.0
            total_perplexity = 0.0
            
            # Training loop with progress bar
            pbar = tqdm(
                self.train_loader,
                desc="MLM Training",
                leave=False,
                dynamic_ncols=True
            )
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass with embeddings if available
                    if self.embeddings is not None:
                        batch_embeddings = self.embeddings[batch_idx * self.train_loader.batch_size:
                                                        (batch_idx + 1) * self.train_loader.batch_size].to(self.device)
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            inputs_embeds=batch_embeddings
                        )
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                    
                    loss = outputs.loss
                    perplexity = torch.exp(loss)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.get('clip_grad_norm', True):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get('max_grad_norm', 1.0)
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_perplexity += perplexity.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'ppl': f"{perplexity.item():.4f}"
                    })
                    
                except Exception as e:
                    logger.error(f"Error in MLM training batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
            
            # Calculate epoch metrics
            avg_loss = total_loss / len(self.train_loader)
            avg_perplexity = total_perplexity / len(self.train_loader)
            
            # Update history
            self.train_losses.append(avg_loss)
            self.perplexities.append(avg_perplexity)
            
            metrics = {
                'loss': avg_loss,
                'perplexity': avg_perplexity
            }
            
            logger.info("MLM training epoch completed successfully")
            return metrics
        except Exception as e:
            logger.error(f"Error in MLM training epoch: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting MLM training for {num_epochs} epochs")
        try:
            epoch_pbar = tqdm(
                range(num_epochs),
                desc="MLM Training Progress",
                position=0
            )
            
            for epoch in epoch_pbar:
                metrics = self.train_epoch()
                
                epoch_pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'ppl': f"{metrics['perplexity']:.4f}"
                })
            
            logger.info("MLM training completed successfully")
            return {
                'loss': self.train_losses,
                'perplexity': self.perplexities
            }
        except Exception as e:
            logger.error(f"Error during MLM training: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def create_mlm_trainer(
    texts: List[str],
    config: Dict[str, Any],
    embeddings: Optional[torch.Tensor] = None
) -> MLMTrainer:
    """
    Create MLM trainer with all components.
    
    Args:
        texts: List of training texts
        config: Configuration dictionary
        
    Returns:
        Configured MLM trainer
    """
    logger.info("Creating MLM trainer")
    try:
        # Initialize components with progress bar
        setup_pbar = tqdm(total=4, desc="Setting up MLM", leave=False)
        
        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        setup_pbar.update(1)
        
        # Create dataset
        dataset = MLMDataset(
            texts,
            tokenizer,
            max_length=config['max_length'],
            mlm_probability=config.get('mlm_probability', 0.15)
        )
        setup_pbar.update(1)
        
        # Create dataloader
        train_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        setup_pbar.update(1)
        
        # Initialize model and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BertForMaskedLM.from_pretrained(config['model_name']).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )
        setup_pbar.update(1)
        
        # Create trainer
        trainer = MLMTrainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            embeddings=embeddings
        )
        
        setup_pbar.close()
        logger.info("MLM trainer created successfully")
        return trainer
    except Exception as e:
        logger.error(f"Error creating MLM trainer: {str(e)}")
        logger.error(traceback.format_exc())
        raise
