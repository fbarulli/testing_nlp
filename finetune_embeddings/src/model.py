from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel, DistilBertModel
import logging
import traceback
from typing import Dict, Optional, Tuple, Union, Any
from tqdm.auto import tqdm
import numpy.typing as npt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinetunedBERT(nn.Module):
    """
    Enhanced BERT model incorporating multiple sentiment-specific models for end-to-end finetuning.
    """
    def __init__(
        self, 
        num_labels: int,
        pretrained_model: str = 'bert-base-uncased',
        dropout_rate: float = 0.1,
        hidden_dim: int = 768,
        activation_monitoring: bool = True,
        model_batch_size: int = 2  # Process models in batches
    ) -> None:
        """
        Initialize the FinetunedBERT model with multiple sentiment models.

        Args:
            num_labels: Number of output classes
            pretrained_model: Name of the pretrained BERT model
            dropout_rate: Dropout probability
            hidden_dim: Dimension of hidden layers
            activation_monitoring: Whether to monitor activations
            model_batch_size: Number of models to process at once
        """
        logger.info("Initializing Enhanced FinetunedBERT")
        try:
            super().__init__()
            self.activation_monitoring = activation_monitoring
            self.activation_stats: Dict[str, float] = {}
            self.model_batch_size = model_batch_size
            
            # Initialize all models with progress bar
            models_to_load = {
                'base_bert': pretrained_model,
                'twitter': 'cardiffnlp/twitter-roberta-base-sentiment',
                'finbert': 'ProsusAI/finbert',
                'multilingual': 'nlptown/bert-base-multilingual-uncased-sentiment',
                'roberta': 'siebert/sentiment-roberta-large-english',
                'emotion': 'bhadresh-savani/bert-base-go-emotion',
                'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english'
            }
            
            self.models = nn.ModuleDict()
            with tqdm(total=len(models_to_load), desc="Loading models", leave=False) as pbar:
                for name, model_path in models_to_load.items():
                    if 'roberta' in model_path.lower():
                        self.models[name] = RobertaModel.from_pretrained(model_path)
                    elif 'distilbert' in model_path.lower():
                        self.models[name] = DistilBertModel.from_pretrained(model_path)
                    else:
                        self.models[name] = BertModel.from_pretrained(model_path)
                    # Enable gradient checkpointing for memory efficiency
                    self.models[name].gradient_checkpointing_enable()
                    pbar.update(1)
            
            # Calculate total embedding dimension
            model_dims = {
                'base_bert': 768,
                'twitter': 768,
                'finbert': 768,
                'multilingual': 768,
                'roberta': 1024,  # RoBERTa large has 1024 dim
                'emotion': 768,
                'distilbert': 768  # DistilBERT dimension
            }
            total_dim = sum(model_dims.values())
            
            # Project combined embeddings to hidden dimension
            self.projection = nn.Linear(total_dim, hidden_dim)
            
            # Feature extraction layers with residual connections
            self.feature_layers = nn.ModuleList([
                ResidualBlock(hidden_dim, dropout_rate) for _ in range(2)
            ])
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_labels)
            )
            
            # Initialize weights for added layers
            self._init_weights()
            
            # Register hooks for monitoring
            if activation_monitoring:
                self._register_hooks()
            
            logger.info("FinetunedBERT initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing FinetunedBERT: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization"""
        logger.info("Initializing weights")
        try:
            for module in tqdm(self.modules(), desc="Initializing weights", leave=False):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            logger.info("Weights initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing weights: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _register_hooks(self) -> None:
        """Register hooks for monitoring gradients and activations"""
        logger.info("Registering monitoring hooks")
        try:
            for name, module in tqdm(self.named_modules(), desc="Registering hooks", leave=False):
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    module.register_forward_hook(
                        lambda m, i, o, name=name: self._activation_hook(name, o)
                    )
                    module.register_full_backward_hook(
                        lambda m, i, o, name=name: self._gradient_hook(name, o[0])
                    )
            logger.info("Hooks registered successfully")
        except Exception as e:
            logger.error(f"Error registering hooks: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _activation_hook(self, name: str, output: torch.Tensor) -> None:
        """
        Monitor layer activations.
        
        Args:
            name: Name of the layer
            output: Layer output tensor
        """
        try:
            if self.training and self.activation_monitoring:
                with torch.no_grad():
                    self.activation_stats[f"{name}_mean"] = float(output.mean().item())
                    self.activation_stats[f"{name}_std"] = float(output.std().item())
                    self.activation_stats[f"{name}_norm"] = float(output.norm().item())
                    self.activation_stats[f"{name}_max"] = float(output.abs().max().item())
        except Exception as e:
            logger.error(f"Error in activation hook: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _gradient_hook(self, name: str, grad: torch.Tensor) -> None:
        """
        Monitor gradients.
        
        Args:
            name: Name of the layer
            grad: Gradient tensor
        """
        try:
            if self.training and self.activation_monitoring:
                with torch.no_grad():
                    self.activation_stats[f"{name}_grad_norm"] = float(grad.norm().item())
                    self.activation_stats[f"{name}_grad_std"] = float(grad.std().item())
                    self.activation_stats[f"{name}_grad_max"] = float(grad.abs().max().item())
        except Exception as e:
            logger.error(f"Error in gradient hook: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_monitoring_stats(self) -> Dict[str, float]:
        """
        Return current monitoring statistics.
        
        Returns:
            Dictionary of monitoring statistics
        """
        try:
            return self.activation_stats.copy()
        except Exception as e:
            logger.error(f"Error getting monitoring stats: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass incorporating all sentiment models.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Optional token type IDs
            position_ids: Optional position IDs
            return_dict: Whether to return a dictionary of outputs
            
        Returns:
            Model outputs either as tensor or dictionary
        """
        try:
            # Process models in batches to save memory
            all_embeddings = []
            model_items = list(self.models.items())
            
            for i in range(0, len(model_items), self.model_batch_size):
                batch_models = model_items[i:i + self.model_batch_size]
                batch_embeddings = []
                
                # Free up memory from previous batch
                if i > 0:
                    torch.cuda.empty_cache()
                
                for name, model in batch_models:
                    if isinstance(model, RobertaModel):
                        # RoBERTa doesn't use token_type_ids
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True
                        )
                    elif isinstance(model, DistilBertModel):
                        # DistilBERT doesn't use token_type_ids or position_ids
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            return_dict=True
                        )
                    # Get pooled output (CLS token representation)
                    pooled_output = outputs.last_hidden_state[:, 0]
                    batch_embeddings.append(pooled_output)
                
                # Concatenate batch embeddings
                all_embeddings.extend(batch_embeddings)
            
            # Normalize each embedding before concatenation
            normalized_embeddings = []
            for embedding in all_embeddings:
                # L2 normalize each embedding
                normalized = F.normalize(embedding, p=2, dim=1)
                normalized_embeddings.append(normalized)
            
            # Concatenate normalized embeddings
            combined_embeddings = torch.cat(normalized_embeddings, dim=1)
            
            # Project and normalize again
            features = self.projection(combined_embeddings)
            features = F.normalize(features, p=2, dim=1)
            
            # Apply feature extraction with residual connections
            for layer in self.feature_layers:
                features = layer(features)
            
            # Apply classification head
            logits = self.classifier(features)
            
            if return_dict:
                return {
                    'logits': logits,
                    'pooled_output': combined_embeddings,
                    'features': features,
                    'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                    'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
                }
            return logits
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class ResidualBlock(nn.Module):
    """
    Residual block with normalization and dropout
    """
    def __init__(self, hidden_dim: int, dropout_rate: float) -> None:
        """
        Initialize ResidualBlock.
        
        Args:
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
        """
        logger.info(f"Initializing ResidualBlock with dim={hidden_dim}")
        try:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            logger.info("ResidualBlock initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ResidualBlock: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        try:
            # Apply residual connection and normalize
            out = self.layers(x)
            out = F.normalize(out, p=2, dim=1)
            return F.normalize(x + out, p=2, dim=1)  # Normalized residual connection
        except Exception as e:
            logger.error(f"Error in ResidualBlock forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def create_attention_mask(input_ids: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """
    Create attention mask from input ids.
    
    Args:
        input_ids: Input token IDs
        padding_idx: Index used for padding
        
    Returns:
        Attention mask tensor
    """
    try:
        return (input_ids != padding_idx).float()
    except Exception as e:
        logger.error(f"Error creating attention mask: {str(e)}")
        logger.error(traceback.format_exc())
        raise
