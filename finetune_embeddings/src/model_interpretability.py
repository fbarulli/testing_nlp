from __future__ import annotations
import torch
import torch.nn as nn
from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    NeuronConductance,
    LayerActivation,
    LayerConductance,
    InternalInfluence,
    NoiseTunnel
)
from captum.attr._utils.visualization import visualize_text
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm.auto import tqdm
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingInterpreter:
    """
    Uses Captum to analyze and interpret embedding behavior and model decisions.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device
    ) -> None:
        """
        Initialize the interpreter.
        
        Args:
            model: The BERT model
            tokenizer: Tokenizer for text processing
            device: Device to run computations on
        """
        logger.info("Initializing EmbeddingInterpreter")
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            
            # Initialize Captum attribution methods
            self.integrated_gradients = LayerIntegratedGradients(
                self.forward_func,
                self.model.bert.embeddings
            )
            
            self.neuron_conductor = NeuronConductance(self.forward_func)
            self.layer_conductor = LayerConductance(
                self.forward_func,
                self.model.bert.encoder.layer[-1]
            )
            
            logger.info("EmbeddingInterpreter initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing EmbeddingInterpreter: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def forward_func(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward function for attribution.
        
        Args:
            inputs: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Model output tensor
        """
        try:
            outputs = self.model(
                input_ids=inputs,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs['logits']
        except Exception as e:
            logger.error(f"Error in forward function: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_token_attributions(
        self,
        text: str,
        target_label: int,
        n_steps: int = 50,
        internal_batch_size: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze token-level attributions using Integrated Gradients.
        
        Args:
            text: Input text to analyze
            target_label: Target class label
            n_steps: Number of steps for attribution
            internal_batch_size: Batch size for internal processing
            
        Returns:
            Dictionary containing attribution results
        """
        logger.info("Analyzing token attributions")
        try:
            # Tokenize input
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(tokens)
            
            # Get attributions
            attributions = self.integrated_gradients.attribute(
                inputs=tokens,
                target=target_label,
                additional_forward_args=(attention_mask,),
                n_steps=n_steps,
                internal_batch_size=internal_batch_size
            )
            
            # Process attributions
            attribution_sum = torch.sum(attributions, dim=-1)
            attributions_norm = attribution_sum / torch.norm(attribution_sum)
            
            # Get word tokens for visualization
            word_tokens = self.tokenizer.convert_ids_to_tokens(tokens[0])
            
            return {
                'tokens': word_tokens,
                'attributions': attributions_norm[0].cpu().numpy(),
                'raw_attributions': attributions[0].cpu().numpy()
            }
        except Exception as e:
            logger.error(f"Error analyzing token attributions: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_neuron_behavior(
        self,
        text: str,
        layer_idx: int,
        neuron_idx: int
    ) -> Dict[str, Any]:
        """
        Analyze individual neuron behavior using NeuronConductance.
        
        Args:
            text: Input text to analyze
            layer_idx: Index of the layer to analyze
            neuron_idx: Index of the neuron to analyze
            
        Returns:
            Dictionary containing neuron analysis results
        """
        logger.info(f"Analyzing neuron behavior for layer {layer_idx}, neuron {neuron_idx}")
        try:
            # Tokenize input
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(tokens)
            
            # Get neuron attributions
            neuron_attrs = self.neuron_conductor.attribute(
                inputs=tokens,
                neuron_selector=(layer_idx, neuron_idx),
                additional_forward_args=(attention_mask,)
            )
            
            # Process attributions
            neuron_attrs_sum = torch.sum(neuron_attrs, dim=-1)
            word_tokens = self.tokenizer.convert_ids_to_tokens(tokens[0])
            
            return {
                'tokens': word_tokens,
                'neuron_attributions': neuron_attrs_sum[0].cpu().numpy(),
                'raw_attributions': neuron_attrs[0].cpu().numpy()
            }
        except Exception as e:
            logger.error(f"Error analyzing neuron behavior: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_layer_influence(
        self,
        text: str,
        target_label: int
    ) -> Dict[str, Any]:
        """
        Analyze layer influence using LayerConductance.
        
        Args:
            text: Input text to analyze
            target_label: Target class label
            
        Returns:
            Dictionary containing layer influence results
        """
        logger.info("Analyzing layer influence")
        try:
            # Tokenize input
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(tokens)
            
            # Get layer attributions
            layer_attrs = self.layer_conductor.attribute(
                inputs=tokens,
                target=target_label,
                additional_forward_args=(attention_mask,)
            )
            
            # Process attributions
            layer_attrs_sum = torch.sum(layer_attrs, dim=-1)
            word_tokens = self.tokenizer.convert_ids_to_tokens(tokens[0])
            
            return {
                'tokens': word_tokens,
                'layer_attributions': layer_attrs_sum[0].cpu().numpy(),
                'raw_attributions': layer_attrs[0].cpu().numpy()
            }
        except Exception as e:
            logger.error(f"Error analyzing layer influence: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def visualize_attributions(
        self,
        attributions_dict: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize token attributions.
        
        Args:
            attributions_dict: Dictionary containing attribution results
            save_path: Optional path to save visualization
        """
        logger.info("Visualizing attributions")
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot token attributions
            sns.barplot(
                x=list(range(len(attributions_dict['tokens']))),
                y=attributions_dict['attributions']
            )
            
            # Customize plot
            plt.xticks(
                range(len(attributions_dict['tokens'])),
                attributions_dict['tokens'],
                rotation=45,
                ha='right'
            )
            plt.xlabel('Tokens')
            plt.ylabel('Attribution Score')
            plt.title('Token-level Attribution Analysis')
            
            # Save or show
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing attributions: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_embedding_space(
        self,
        texts: List[str],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Analyze embedding space structure using attribution methods.
        
        Args:
            texts: List of input texts
            labels: Optional tensor of labels
            
        Returns:
            Dictionary containing embedding analysis results
        """
        logger.info("Analyzing embedding space")
        try:
            results = {}
            
            # Process texts in batches
            batch_size = 32
            all_embeddings = []
            all_attributions = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # Get embeddings
                tokens = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**tokens, return_dict=True)
                    embeddings = outputs['pooled_output']
                    all_embeddings.append(embeddings.cpu())
                
                # Get attributions for each text
                if labels is not None:
                    batch_labels = labels[i:i + batch_size]
                    attributions = self.integrated_gradients.attribute(
                        inputs=tokens['input_ids'],
                        target=batch_labels,
                        additional_forward_args=(tokens['attention_mask'],)
                    )
                    all_attributions.append(attributions.cpu())
            
            # Combine results
            embeddings = torch.cat(all_embeddings, dim=0)
            results['embeddings'] = embeddings
            
            if all_attributions:
                attributions = torch.cat(all_attributions, dim=0)
                results['attributions'] = attributions
            
            # Compute embedding space metrics
            results['embedding_norm'] = torch.norm(embeddings, dim=1).mean().item()
            results['embedding_std'] = embeddings.std(dim=0).mean().item()
            
            if len(all_attributions) > 0:
                attribution_magnitude = torch.cat(all_attributions, dim=0).abs().mean().item()
                results['attribution_magnitude'] = attribution_magnitude
            
            logger.info("Embedding space analysis complete")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing embedding space: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def create_interpreter(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device
) -> EmbeddingInterpreter:
    """
    Create an EmbeddingInterpreter instance.
    
    Args:
        model: The BERT model
        tokenizer: Tokenizer for text processing
        device: Device to run computations on
        
    Returns:
        Configured EmbeddingInterpreter
    """
    logger.info("Creating EmbeddingInterpreter")
    try:
        interpreter = EmbeddingInterpreter(model, tokenizer, device)
        logger.info("EmbeddingInterpreter created successfully")
        return interpreter
    except Exception as e:
        logger.error(f"Error creating interpreter: {str(e)}")
        logger.error(traceback.format_exc())
        raise
