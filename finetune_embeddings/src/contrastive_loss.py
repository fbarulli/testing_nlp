from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm.auto import tqdm
import numpy.typing as npt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise-Contrastive Estimation) Loss for contrastive learning.
    This loss helps learn better embeddings by pulling similar samples together
    and pushing dissimilar samples apart in the embedding space.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean',
        contrast_mode: str = 'all',
        chunk_size: int = 256  # Process similarity matrix in chunks
    ) -> None:
        """
        Initialize InfoNCE Loss.
        
        Args:
            temperature: Temperature parameter for scaling
            reduction: Reduction method ('mean' or 'sum')
            contrast_mode: How to compute contrasts ('all' or 'one')
            chunk_size: Size of chunks for similarity computation
        """
        logger.info("Initializing InfoNCE Loss")
        try:
            super().__init__()
            self.temperature = temperature
            self.reduction = reduction
            self.contrast_mode = contrast_mode
            self.chunk_size = chunk_size
            logger.info(
                f"InfoNCE Loss initialized with temperature={temperature}, "
                f"reduction={reduction}, contrast_mode={contrast_mode}"
            )
        except Exception as e:
            logger.error(f"Error initializing InfoNCE Loss: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def compute_similarity_chunk(
        self,
        features: torch.Tensor,
        chunk_start: int,
        chunk_size: int
    ) -> torch.Tensor:
        """
        Compute similarity matrix for a chunk of features.
        
        Args:
            features: Normalized feature tensor
            chunk_start: Start index of chunk
            chunk_size: Size of chunk
            
        Returns:
            Chunk of similarity matrix
        """
        try:
            chunk_end = min(chunk_start + chunk_size, features.size(0))
            chunk_features = features[chunk_start:chunk_end]
            
            # Compute similarity between chunk and all features
            sim_chunk = torch.matmul(chunk_features, features.T)
            return sim_chunk
            
        except Exception as e:
            logger.error(f"Error computing similarity chunk: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            features: Tensor of shape (batch_size, feature_dim)
            labels: Optional tensor for supervised contrastive learning
            mask: Optional mask for valid pairs
            
        Returns:
            Computed loss value
        """
        logger.debug("Computing InfoNCE loss")
        try:
            # Normalize features
            features = F.normalize(features, dim=1)
            batch_size = features.size(0)
            
            total_loss = 0.0
            total_pairs = 0
            
            # Process in chunks to save memory
            for i in range(0, batch_size, self.chunk_size):
                chunk_start = i
                chunk_end = min(i + self.chunk_size, batch_size)
                chunk_size = chunk_end - chunk_start
                
                # Get chunk features and labels
                chunk_features = features[chunk_start:chunk_end]
                chunk_labels = labels[chunk_start:chunk_end] if labels is not None else None
                
                try:
                    # Compute similarity with gradient clipping
                    chunk_features = torch.clamp(chunk_features, min=-1e3, max=1e3)
                    features_clipped = torch.clamp(features, min=-1e3, max=1e3)
                    sim_chunk = torch.matmul(chunk_features, features_clipped.T)
                    
                    # Apply temperature with stability check
                    temperature = max(self.temperature, 1e-4)  # Prevent division by zero
                    sim_chunk = sim_chunk / temperature
                    
                    # Create chunk masks
                    chunk_mask_self = torch.ones_like(sim_chunk, dtype=torch.bool)
                    chunk_mask_self[:, chunk_start:chunk_end].fill_diagonal_(False)
                    
                    if chunk_labels is not None:
                        chunk_labels = chunk_labels.contiguous().view(-1, 1)
                        chunk_mask_pos = chunk_labels == labels.view(1, -1)
                        chunk_mask_pos = chunk_mask_pos & chunk_mask_self
                        # Ensure at least one positive pair
                        if not chunk_mask_pos.any():
                            chunk_mask_pos = chunk_mask_self
                    else:
                        chunk_mask_pos = chunk_mask_self
                    
                    # For numerical stability
                    sim_max, _ = torch.max(sim_chunk, dim=1, keepdim=True)
                    sim_chunk = sim_chunk - sim_max.detach()
                    sim_chunk = torch.clamp(sim_chunk, min=-1e3, max=1e3)
                    
                    # Compute exp and log-sum-exp with stability
                    exp_sim = torch.exp(sim_chunk)
                    exp_sim = torch.clamp(exp_sim, min=1e-8)  # Prevent zero exp
                    exp_sim = exp_sim * chunk_mask_self
                    log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
                    
                    # Compute log probabilities
                    log_prob = sim_chunk - log_sum_exp
                    log_prob = torch.clamp(log_prob, min=-1e3)  # Prevent -inf
                    
                    # Compute mean of positive pairs for chunk
                    pos_pairs = chunk_mask_pos.sum(1)
                    chunk_loss = -(chunk_mask_pos * log_prob).sum(1)
                    valid_pairs = pos_pairs > 0
                    if valid_pairs.any():
                        chunk_loss = chunk_loss[valid_pairs] / pos_pairs[valid_pairs]
                    else:
                        chunk_loss = torch.zeros(1, device=chunk_features.device)
                except Exception as e:
                    logger.error(f"Error in chunk computation: {str(e)}")
                    chunk_loss = torch.zeros(1, device=chunk_features.device)
                
                # Accumulate loss
                total_loss += chunk_loss.sum()
                total_pairs += (pos_pairs > 0).sum()
            
            # Compute mean loss
            mean_loss = total_loss / (total_pairs + 1e-8)
            
            # Handle reduction
            if self.reduction == 'mean':
                loss = mean_loss
            elif self.reduction == 'sum':
                loss = mean_loss * total_pairs
            else:
                loss = mean_loss
            
            return loss
            
        except Exception as e:
            logger.error(f"Error computing InfoNCE loss: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class ContrastiveLearningWrapper:
    """
    Wrapper class to handle contrastive learning with InfoNCE loss.
    """
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 0.07,
        queue_size: int = 65536,
        chunk_size: int = 256  # Process features in chunks
    ) -> None:
        """
        Initialize ContrastiveLearningWrapper.
        
        Args:
            model: Base model to wrap
            temperature: Temperature for InfoNCE loss
            queue_size: Size of memory queue for negative samples
            chunk_size: Size of chunks for processing
        """
        logger.info("Initializing ContrastiveLearningWrapper")
        try:
            self.model = model
            self.criterion = InfoNCELoss(
                temperature=temperature,
                chunk_size=chunk_size
            )
            self.queue_size = queue_size
            self.chunk_size = chunk_size
            self.register_queue()
            logger.info("ContrastiveLearningWrapper initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ContrastiveLearningWrapper: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def register_queue(self) -> None:
        """Initialize the queue for storing negative samples"""
        try:
            self.queue: Optional[torch.Tensor] = None
            self.queue_ptr = torch.zeros(1, dtype=torch.long)
            logger.info(f"Queue initialized with size {self.queue_size}")
        except Exception as e:
            logger.error(f"Error initializing queue: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """
        Update queue of negative samples.
        
        Args:
            keys: New samples to add to queue
        """
        try:
            batch_size = keys.shape[0]
            
            # Initialize queue if not exists
            if self.queue is None:
                self.queue = torch.zeros(
                    (self.queue_size, keys.shape[1]),
                    dtype=keys.dtype,
                    device=keys.device
                )
                self.queue_ptr[0] = 0
            
            # Get current pointer
            ptr = int(self.queue_ptr[0])
            
            # Compute how many samples we can add
            space_left = self.queue_size - ptr
            samples_to_add = min(batch_size, space_left)
            
            # Add samples to queue
            if samples_to_add > 0:
                self.queue[ptr:ptr + samples_to_add] = keys[:samples_to_add]
                ptr = (ptr + samples_to_add) % self.queue_size
                self.queue_ptr[0] = ptr
            
            # If queue is not full, don't use it yet
            if ptr < self.queue_size and self.queue_ptr[0] == 0:
                self.queue = None
            
        except Exception as e:
            logger.error(f"Error in dequeue and enqueue: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss using current features and queue.
        
        Args:
            features: Current batch features
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Computed contrastive loss
        """
        logger.debug("Computing contrastive loss")
        try:
            # Process features in chunks if needed
            if features.size(0) > self.chunk_size:
                total_loss = 0.0
                num_chunks = (features.size(0) + self.chunk_size - 1) // self.chunk_size
                
                for i in range(num_chunks):
                    start_idx = i * self.chunk_size
                    end_idx = min((i + 1) * self.chunk_size, features.size(0))
                    
                    chunk_features = features[start_idx:end_idx]
                    chunk_labels = labels[start_idx:end_idx] if labels is not None else None
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        # Get negative samples from queue
                        if self.queue is not None:
                            all_features = torch.cat([chunk_features, self.queue], dim=0)
                            if chunk_labels is not None:
                                all_labels = torch.cat([
                                    chunk_labels,
                                    torch.zeros(self.queue_size, device=chunk_labels.device)
                                ])
                            else:
                                all_labels = None
                        else:
                            all_features = chunk_features
                            all_labels = chunk_labels
                        
                        # Compute loss for chunk
                        loss = self.criterion(all_features, all_labels)
                        if torch.isfinite(loss):
                            total_loss += loss * (end_idx - start_idx)
                        else:
                            logger.warning("Skipping invalid chunk loss")
                
                # Update queue with all features
                self._dequeue_and_enqueue(features)
                
                return total_loss / features.size(0)
            else:
                # Process entire batch at once
                if self.queue is not None:
                    all_features = torch.cat([features, self.queue], dim=0)
                    if labels is not None:
                        all_labels = torch.cat([
                            labels,
                            torch.zeros(self.queue_size, device=labels.device)
                        ])
                    else:
                        all_labels = None
                else:
                    all_features = features
                    all_labels = labels
                
                # Compute loss
                loss = self.criterion(all_features, all_labels)
                
                # Update queue
                self._dequeue_and_enqueue(features)
                
                return loss
            
        except Exception as e:
            logger.error(f"Error computing contrastive loss: {str(e)}")
            logger.error(traceback.format_exc())
            raise
