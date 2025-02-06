from __future__ import annotations
import torch
import numpy as np
import logging
import traceback
from typing import Dict, List, Optional, Union, Any, Callable
from tqdm.auto import tqdm
import numpy.typing as npt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DynamicHyperparameters:
    """
    Manages dynamic hyperparameters that adapt during training based on metrics
    and training progress.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        metrics_history: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """
        Initialize dynamic hyperparameter scheduler.
        
        Args:
            config: Configuration dictionary
            metrics_history: Optional history of training metrics
        """
        logger.info("Initializing DynamicHyperparameters")
        try:
            self.config = config
            self.metrics_history = metrics_history or {}
            
            # Initialize dynamic parameters
            self.params: Dict[str, float] = {
                'temperature': config['pretraining']['contrastive']['temperature'],
                'mlm_probability': config['pretraining']['mlm']['probability'],
                'contrastive_weight': config['pretraining']['contrastive']['loss_weight'],
                'learning_rate': config['learning_rate']
            }
            
            # Track parameter history
            self.param_history: Dict[str, List[float]] = {
                k: [v] for k, v in self.params.items()
            }
            
            logger.info("DynamicHyperparameters initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DynamicHyperparameters: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_temperature(
        self,
        epoch: int,
        isotropy: float,
        cluster_quality: float
    ) -> None:
        """
        Dynamically adjust InfoNCE temperature based on embedding quality.
        
        Args:
            epoch: Current training epoch
            isotropy: Current isotropy metric
            cluster_quality: Current clustering quality metric
        """
        logger.debug("Updating temperature")
        try:
            # Increase temperature if embeddings are too scattered
            if isotropy < 0.4:
                self.params['temperature'] *= 1.1
            # Decrease temperature if clusters are not well-separated
            elif cluster_quality < 0.7:
                self.params['temperature'] *= 0.9
            
            # Keep temperature in reasonable bounds
            self.params['temperature'] = max(0.01, min(1.0, self.params['temperature']))
            self.param_history['temperature'].append(self.params['temperature'])
            
        except Exception as e:
            logger.error(f"Error updating temperature: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_mlm_probability(
        self,
        epoch: int,
        mlm_loss: float,
        perplexity: float
    ) -> None:
        """
        Adjust MLM masking probability based on model performance.
        
        Args:
            epoch: Current training epoch
            mlm_loss: Current MLM loss
            perplexity: Current perplexity
        """
        logger.debug("Updating MLM probability")
        try:
            # Increase masking if model is doing too well
            if perplexity < 2.0:
                self.params['mlm_probability'] *= 1.1
            # Decrease masking if model is struggling
            elif mlm_loss > 3.0:
                self.params['mlm_probability'] *= 0.9
            
            # Keep probability in reasonable bounds
            self.params['mlm_probability'] = max(0.05, min(0.25, self.params['mlm_probability']))
            self.param_history['mlm_probability'].append(self.params['mlm_probability'])
            
        except Exception as e:
            logger.error(f"Error updating MLM probability: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_contrastive_weight(
        self,
        epoch: int,
        contrastive_loss: float,
        mlm_loss: float
    ) -> None:
        """
        Adjust weight of contrastive loss based on relative performance.
        
        Args:
            epoch: Current training epoch
            contrastive_loss: Current contrastive loss
            mlm_loss: Current MLM loss
        """
        logger.debug("Updating contrastive weight")
        try:
            # Balance losses by adjusting weight
            ratio = mlm_loss / (contrastive_loss + 1e-8)
            if ratio > 2.0:
                self.params['contrastive_weight'] *= 1.1
            elif ratio < 0.5:
                self.params['contrastive_weight'] *= 0.9
            
            # Keep weight in reasonable bounds
            self.params['contrastive_weight'] = max(0.01, min(1.0, self.params['contrastive_weight']))
            self.param_history['contrastive_weight'].append(self.params['contrastive_weight'])
            
        except Exception as e:
            logger.error(f"Error updating contrastive weight: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_learning_rate(
        self,
        epoch: int,
        loss_history: List[float],
        plateau_patience: int = 3
    ) -> None:
        """
        Adjust learning rate based on loss trends.
        
        Args:
            epoch: Current training epoch
            loss_history: History of total loss values
            plateau_patience: Number of epochs to wait before reducing LR
        """
        logger.debug("Updating learning rate")
        try:
            if len(loss_history) >= plateau_patience:
                recent_losses = loss_history[-plateau_patience:]
                if all(abs(recent_losses[i] - recent_losses[i-1]) < 1e-4 
                       for i in range(1, len(recent_losses))):
                    self.params['learning_rate'] *= 0.5
            
            # Keep learning rate in reasonable bounds
            self.params['learning_rate'] = max(1e-6, min(1e-3, self.params['learning_rate']))
            self.param_history['learning_rate'].append(self.params['learning_rate'])
            
        except Exception as e:
            logger.error(f"Error updating learning rate: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_current_params(self) -> Dict[str, float]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary of current parameter values
        """
        return self.params.copy()

    def get_param_history(self) -> Dict[str, List[float]]:
        """
        Get parameter history.
        
        Returns:
            Dictionary of parameter histories
        """
        return self.param_history.copy()

class MetricBasedScheduler:
    """
    Scheduler that adjusts hyperparameters based on multiple metrics.
    """
    def __init__(
        self,
        dynamic_params: DynamicHyperparameters,
        update_frequency: int = 100
    ) -> None:
        """
        Initialize metric-based scheduler.
        
        Args:
            dynamic_params: DynamicHyperparameters instance
            update_frequency: Steps between parameter updates
        """
        logger.info("Initializing MetricBasedScheduler")
        try:
            self.dynamic_params = dynamic_params
            self.update_frequency = update_frequency
            self.steps = 0
            
            # Initialize metric trackers
            self.metric_trackers: Dict[str, List[float]] = {
                'mlm_loss': [],
                'contrastive_loss': [],
                'total_loss': [],
                'perplexity': [],
                'isotropy': [],
                'cluster_quality': []
            }
            
            logger.info("MetricBasedScheduler initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MetricBasedScheduler: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def step(
        self,
        metrics: Dict[str, float],
        epoch: int
    ) -> Dict[str, float]:
        """
        Update step for scheduler.
        
        Args:
            metrics: Current training metrics
            epoch: Current training epoch
            
        Returns:
            Updated parameter values
        """
        logger.debug("Performing scheduler step")
        try:
            # Update metric history
            for name, value in metrics.items():
                if name in self.metric_trackers:
                    self.metric_trackers[name].append(value)
            
            self.steps += 1
            
            # Update parameters at specified frequency
            if self.steps % self.update_frequency == 0:
                # Update temperature based on embedding quality
                self.dynamic_params.update_temperature(
                    epoch,
                    metrics.get('isotropy', 0.0),
                    metrics.get('cluster_quality', 0.0)
                )
                
                # Update MLM probability based on performance
                self.dynamic_params.update_mlm_probability(
                    epoch,
                    metrics.get('mlm_loss', 0.0),
                    metrics.get('perplexity', 0.0)
                )
                
                # Update contrastive weight based on loss balance
                self.dynamic_params.update_contrastive_weight(
                    epoch,
                    metrics.get('contrastive_loss', 0.0),
                    metrics.get('mlm_loss', 0.0)
                )
                
                # Update learning rate based on loss trends
                self.dynamic_params.update_learning_rate(
                    epoch,
                    self.metric_trackers['total_loss']
                )
            
            return self.dynamic_params.get_current_params()
            
        except Exception as e:
            logger.error(f"Error in scheduler step: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_metric_history(self) -> Dict[str, List[float]]:
        """
        Get history of tracked metrics.
        
        Returns:
            Dictionary of metric histories
        """
        return self.metric_trackers.copy()

def create_scheduler(
    config: Dict[str, Any],
    update_frequency: int = 100
) -> MetricBasedScheduler:
    """
    Create a metric-based scheduler.
    
    Args:
        config: Configuration dictionary
        update_frequency: Steps between parameter updates
        
    Returns:
        Configured MetricBasedScheduler
    """
    logger.info("Creating metric-based scheduler")
    try:
        dynamic_params = DynamicHyperparameters(config)
        scheduler = MetricBasedScheduler(dynamic_params, update_frequency)
        logger.info("Metric-based scheduler created successfully")
        return scheduler
    except Exception as e:
        logger.error(f"Error creating scheduler: {str(e)}")
        logger.error(traceback.format_exc())
        raise
