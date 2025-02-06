from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import logging
import traceback
from typing import Dict, List, Optional, Union, Any, Tuple
from tqdm.auto import tqdm
import numpy as np
import csv
from dynamic_scheduler import create_scheduler
from contrastive_loss import ContrastiveLearningWrapper
from mlm_trainer import MLMTrainer, create_mlm_trainer

# Configure logging to file
log_file_handler = logging.FileHandler('training.log')
log_file_handler.setFormatter(
    logging.Formatter('%(asctime)s,%(levelname)s,%(message)s')
)
logger = logging.getLogger(__name__)
logger.handlers = [log_file_handler]  # Replace default handlers
logger.setLevel(logging.INFO)

class BERTTrainer:
    """
    BERT Trainer class handling pretraining and fine-tuning with advanced features.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config: Dict[str, Any],
        output_dir: Optional[Path] = None,
        project_name: str = "bert-finetuning"
    ) -> None:
        """Initialize trainer with all components."""
        logger.info("Initializing Trainer")
        try:
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.device = device
            self.config = config
            self.output_dir = Path(output_dir) if output_dir else Path(config['output_dir'])
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize metrics tracking
            self.train_metrics = {'loss': [], 'accuracy': []}
            self.val_metrics = {'loss': [], 'accuracy': []}
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            
            # Setup mixed precision training
            self.scaler = amp.GradScaler() if config['fp16'] else None
            
            # Setup dynamic scheduler if enabled
            self.dynamic_scheduler = (
                create_scheduler(config) if config.get('use_dynamic_scheduler') else None
            )
            
            # Setup contrastive learning if enabled
            if config['pretraining']['contrastive']['enabled']:
                self.contrastive_wrapper = ContrastiveLearningWrapper(
                    model,
                    temperature=config['pretraining']['contrastive']['temperature'],
                    queue_size=min(config['pretraining']['contrastive']['queue_size'], 1024),  # Smaller queue
                    chunk_size=64  # Process in very small chunks
                )
            else:
                self.contrastive_wrapper = None
            
            # Setup MLM if enabled
            if config['pretraining']['mlm']['enabled']:
                self.mlm_trainer = create_mlm_trainer(
                    texts=train_loader.dataset.texts,  # Access texts directly from dataset
                    config={
                        'model_name': config['model_name'],
                        'max_length': 512,
                        'batch_size': config['batch_size'],
                        'num_workers': 4,
                        'learning_rate': float(config['pretraining']['mlm']['learning_rate']),
                        'weight_decay': float(config['pretraining']['mlm']['weight_decay']),
                        'mlm_probability': config['pretraining']['mlm']['probability']
                    },
                    embeddings=train_loader.dataset.features if hasattr(train_loader.dataset, 'features') else None
                )
            else:
                self.mlm_trainer = None
            
            # Initialize monitoring
            self.monitoring_stats: Dict[str, Any] = {}
            
            # Setup CSV logging
            metrics_file = self.output_dir / 'metrics.csv'
            self.csv_writer = csv.DictWriter(
                open(metrics_file, 'w', newline=''),
                fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
            )
            self.csv_writer.writeheader()
            
            # Initialize wandb with API key
            wandb.login(key="6d8b76b5019f1abc6a2e78a467ce9232a7fa80b5")
            wandb.init(
                project="all_embeddings_tune",
                name=f"run_{wandb.util.generate_id()}",
                config=config,
                dir=str(self.output_dir),
                resume="allow"
            )
            logger.info("Initialized wandb tracking")
            
            logger.info("Trainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Trainer: {str(e)}")
            logger.error(traceback.format_exc())
            # Clean up wandb if initialization fails
            try:
                wandb.finish()
            except:
                pass
            raise

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        try:
            self.model.train()
            total_loss = 0.0
            total_acc = 0.0
            total_samples = 0
            
            # Training loop with progress bar
            pbar = tqdm(
                self.train_loader,
                desc="Training",
                leave=False,
                dynamic_ncols=True,
                position=1
            )
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Clear cache before batch
                    torch.cuda.empty_cache()
                    
                    # Compute loss and metrics
                    metrics = self._training_step(batch)
                    
                    # Update progress bar
                    pbar.set_postfix(metrics, refresh=False)
                    
                    # Update totals
                    total_loss += metrics['loss'] * len(batch['input_ids'])
                    if 'accuracy' in metrics:
                        total_acc += metrics['accuracy'] * len(batch['input_ids'])
                    total_samples += len(batch['input_ids'])
                    
                    # Log batch metrics to wandb
                    wandb.log({
                        'batch/loss': metrics['loss'],
                        'batch/accuracy': metrics.get('accuracy', 0.0),
                        'batch/contrastive_loss': metrics.get('contrastive_loss', 0.0),
                        'batch/mlm_loss': metrics.get('mlm_loss', 0.0),
                        'batch/mlm_accuracy': metrics.get('mlm_accuracy', 0.0),
                        'batch/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'batch/grad_norm': self._get_grad_norm(),
                        **{f"batch/{k}": v for k, v in self.monitoring_stats.items()}
                    }, step=self.current_epoch * len(self.train_loader) + batch_idx)
                    
                    # Clear cache after batch
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
            
            # Calculate epoch metrics
            epoch_metrics = {
                'loss': total_loss / total_samples,
                'accuracy': total_acc / total_samples if total_acc > 0 else 0.0
            }
            
            # Update metric history
            self.train_metrics['loss'].append(epoch_metrics['loss'])
            self.train_metrics['accuracy'].append(epoch_metrics['accuracy'])
            
            return epoch_metrics
        except Exception as e:
            logger.error(f"Error in training epoch: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _training_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Perform single training step."""
        try:
            metrics = {}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Separate labels from input batch
            labels = batch.pop('labels').to(self.device)
            # Move tensors to device, handle lists of tensors
            inputs = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
                elif isinstance(v, list) and all(isinstance(x, torch.Tensor) for x in v):
                    inputs[k] = [x.to(self.device) for x in v]
                else:
                    inputs[k] = v
            
            # Clear cache before forward pass
            torch.cuda.empty_cache()
            
            # Mixed precision context
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                # Filter inputs to only include model-expected arguments
                model_inputs = {
                    k: v for k, v in inputs.items() 
                    if k in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']
                }
                # Forward pass
                outputs = self.model(**model_inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Compute loss
                loss = F.cross_entropy(logits, labels)
                
                # Add MLM loss if enabled
                if self.mlm_trainer is not None:
                    mlm_outputs = self.mlm_trainer.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['input_ids'].clone(),
                        inputs_embeds=inputs.get('features', None)
                    )
                    mlm_loss = mlm_outputs.loss
                    loss = loss + self.config['pretraining']['mlm']['loss_weight'] * mlm_loss
                    metrics['mlm_loss'] = mlm_loss.item()
                    
                    # Calculate MLM accuracy
                    mlm_logits = mlm_outputs.logits
                    mlm_preds = mlm_logits.argmax(dim=-1)
                    mlm_labels = inputs['input_ids'].clone()
                    mlm_mask = mlm_labels != -100
                    mlm_accuracy = (mlm_preds[mlm_mask] == mlm_labels[mlm_mask]).float().mean().item()
                    metrics['mlm_accuracy'] = mlm_accuracy
                
                # Add contrastive loss if enabled
                if self.contrastive_wrapper is not None:
                    features = outputs['pooled_output']
                    # Get and validate contrastive loss
                    contrastive_loss = self.contrastive_wrapper.get_contrastive_loss(
                        features,
                        labels
                    )
                    
                    # Check for valid loss
                    if torch.isfinite(contrastive_loss):
                        loss = loss + self.config['pretraining']['contrastive']['loss_weight'] * contrastive_loss
                        metrics['contrastive_loss'] = contrastive_loss.item()
                    else:
                        logger.warning("Skipping invalid contrastive loss")
                        metrics['contrastive_loss'] = 0.0
            
            # Check for valid loss before backward
            if torch.isfinite(loss):
                # Backward pass with mixed precision if enabled
                if self.scaler:
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                    
                    # Unscale gradients for clipping
                    if self.config['max_grad_norm'] > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['max_grad_norm']
                        )
                    
                    # Handle optimizer step based on type
                    if isinstance(self.optimizer, (torch.optim.LBFGS, torch.optim.ASGD)):
                        def closure():
                            self.optimizer.zero_grad()
                            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                                # Filter inputs to only include model-expected arguments
                                model_inputs = {
                                    k: v for k, v in inputs.items() 
                                    if k in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']
                                }
                                outputs = self.model(**model_inputs)
                                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                                loss = F.cross_entropy(logits, labels)
                                if self.contrastive_wrapper is not None:
                                    features = outputs['pooled_output']
                                    contrastive_loss = self.contrastive_wrapper.get_contrastive_loss(
                                        features,
                                        labels
                                    )
                                    if torch.isfinite(contrastive_loss):
                                        loss = loss + self.config['pretraining']['contrastive']['loss_weight'] * contrastive_loss
                            scaled_loss = self.scaler.scale(loss)
                            scaled_loss.backward()
                            return scaled_loss
                        try:
                            self.scaler.step(self.optimizer, closure)
                            self.scaler.update()
                        except RuntimeError as e:
                            logger.warning(f"Optimizer step failed: {str(e)}")
                            self.scaler.update()
                    else:
                        try:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        except RuntimeError as e:
                            logger.warning(f"Optimizer step failed: {str(e)}")
                            self.scaler.update()
                else:
                    loss.backward()
                    if self.config['max_grad_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['max_grad_norm']
                        )
                    # Check gradients
                    valid_gradients = True
                    for param in self.model.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            valid_gradients = False
                            break
                    
                    if valid_gradients:
                        self.optimizer.step()
                    else:
                        logger.warning("Skipping step due to invalid gradients")
            else:
                logger.warning("Skipping backward pass due to invalid loss")
            
            # Update learning rate (after optimizer step)
            if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            # Update dynamic scheduler if enabled
            if self.dynamic_scheduler is not None:
                updated_params = self.dynamic_scheduler.step(metrics, self.current_epoch)
                # Apply updated parameters
                if self.contrastive_wrapper:
                    self.contrastive_wrapper.criterion.temperature = updated_params['temperature']
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = updated_params['learning_rate']
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean().item()
            metrics['accuracy'] = accuracy
            metrics['loss'] = loss.item()
            
            # Update monitoring stats
            if hasattr(self.model, 'get_monitoring_stats'):
                self.monitoring_stats.update(self.model.get_monitoring_stats())
            
            # Clear cache after step
            torch.cuda.empty_cache()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        try:
            self.model.eval()
            total_loss = 0.0
            total_acc = 0.0
            total_samples = 0
            
            # Evaluation loop with progress bar
            pbar = tqdm(
                self.val_loader,
                desc="Evaluating",
                leave=False,
                dynamic_ncols=True,
                position=1
            )
            
            for batch_idx, batch in enumerate(pbar):
                # Clear cache before batch
                torch.cuda.empty_cache()
                
                # Separate labels from input batch
                labels = batch.pop('labels').to(self.device)
                # Move tensors to device, handle lists of tensors
                inputs = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                    elif isinstance(v, list) and all(isinstance(x, torch.Tensor) for x in v):
                        inputs[k] = [x.to(self.device) for x in v]
                    else:
                        inputs[k] = v
                
                # Filter inputs to only include model-expected arguments
                model_inputs = {
                    k: v for k, v in inputs.items() 
                    if k in ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']
                }
                # Forward pass
                outputs = self.model(**model_inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Initialize metrics dict
                metrics = {}
                
                # Compute loss
                loss = F.cross_entropy(logits, labels)
                
                # Add MLM validation metrics if enabled
                if self.mlm_trainer is not None:
                    mlm_outputs = self.mlm_trainer.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['input_ids'].clone(),
                        inputs_embeds=inputs.get('features', None)
                    )
                    mlm_loss = mlm_outputs.loss
                    
                    # Calculate MLM validation accuracy
                    mlm_logits = mlm_outputs.logits
                    mlm_preds = mlm_logits.argmax(dim=-1)
                    mlm_labels = inputs['input_ids'].clone()
                    mlm_mask = mlm_labels != -100
                    mlm_accuracy = (mlm_preds[mlm_mask] == mlm_labels[mlm_mask]).float().mean().item()
                    
                    # Add to metrics
                    metrics['mlm_loss'] = mlm_loss.item()
                    metrics['mlm_accuracy'] = mlm_accuracy
                
                # Compute accuracy
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == labels).float().mean().item()
                metrics['accuracy'] = accuracy
                total_acc += accuracy * len(inputs['input_ids'])
                
                total_loss += loss.item() * len(inputs['input_ids'])
                total_samples += len(inputs['input_ids'])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / total_samples,
                    'accuracy': total_acc / total_samples
                }, refresh=False)
                
                # Clear cache after batch
                torch.cuda.empty_cache()
            
            # Calculate metrics
            metrics = {
                'loss': total_loss / total_samples,
                'accuracy': total_acc / total_samples
            }
            
            # Update metric history
            self.val_metrics['loss'].append(metrics['loss'])
            self.val_metrics['accuracy'].append(metrics['accuracy'])
            
            # Check for best model
            if metrics['loss'] < self.best_val_loss:
                self.best_val_loss = metrics['loss']
                self.save_checkpoint('best_model.pt')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            return metrics
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train for specified number of epochs."""
        try:
            # Training loop with progress bar
            epoch_pbar = tqdm(
                range(num_epochs),
                desc="Training Progress",
                position=0,
                leave=True
            )
            
            for epoch in epoch_pbar:
                self.current_epoch = epoch
                
                # Clear cache before epoch
                torch.cuda.empty_cache()
                
                # Training epoch
                train_metrics = self.train_epoch()
                
                # Evaluation
                val_metrics = self.evaluate()
                
                # Update progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f"{train_metrics['loss']:.4f}",
                    'val_loss': f"{val_metrics['loss']:.4f}",
                    'train_acc': f"{train_metrics['accuracy']:.4f}",
                    'val_acc': f"{val_metrics['accuracy']:.4f}"
                }, refresh=False)
                
                # Log metrics to CSV
                self.csv_writer.writerow({
                    'epoch': epoch + 1,
                    'train_loss': f"{train_metrics['loss']:.4f}",
                    'train_acc': f"{train_metrics['accuracy']:.4f}",
                    'val_loss': f"{val_metrics['loss']:.4f}",
                    'val_acc': f"{val_metrics['accuracy']:.4f}"
                })
                
                # Log metrics to wandb
                wandb.log({
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'train/mlm_loss': train_metrics.get('mlm_loss', 0.0),
                    'train/mlm_accuracy': train_metrics.get('mlm_accuracy', 0.0),
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/mlm_loss': val_metrics.get('mlm_loss', 0.0),
                    'val/mlm_accuracy': val_metrics.get('mlm_accuracy', 0.0),
                    'monitoring/learning_rate': self.optimizer.param_groups[0]['lr'],
                    **{f"monitoring/{k}": v for k, v in self.monitoring_stats.items()}
                })
                
                # Plot training curves
                if (epoch + 1) % self.config['logging_steps'] == 0:
                    self.plot_metrics()
                
                # Early stopping
                if self.patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Clear cache after epoch
                torch.cuda.empty_cache()
            
            # Save final model
            self.save_checkpoint('final_model.pt')
            
            # Plot final metrics
            self.plot_metrics()
            
            # Finish wandb run
            metrics = {
                'train_loss': self.train_metrics['loss'],
                'train_accuracy': self.train_metrics['accuracy'],
                'val_loss': self.val_metrics['loss'],
                'val_accuracy': self.val_metrics['accuracy']
            }
            wandb.finish()
            return metrics
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(traceback.format_exc())
            # Clean up wandb on error
            try:
                wandb.finish()
            except:
                pass
            raise

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'config': self.config,
                'best_val_loss': self.best_val_loss,
                'monitoring_stats': self.monitoring_stats
            }
            
            torch.save(checkpoint, self.output_dir / filename)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(self.output_dir / filename)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_metrics = checkpoint['train_metrics']
            self.val_metrics = checkpoint['val_metrics']
            self.best_val_loss = checkpoint['best_val_loss']
            self.monitoring_stats = checkpoint['monitoring_stats']
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def plot_metrics(self) -> None:
        """Plot and save training curves."""
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('seaborn-darkgrid')
        
        # Create a figure with two rows for loss and accuracy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Training and Validation Metrics', fontsize=16, y=0.95)
        
        epochs = range(1, len(self.train_metrics['loss']) + 1)
        
        # Plot Loss
        min_loss = min(min(self.train_metrics['loss']), min(self.val_metrics['loss']))
        max_loss = max(max(self.train_metrics['loss']), max(self.val_metrics['loss']))
        loss_margin = (max_loss - min_loss) * 0.1
        
        ax1.plot(epochs, self.train_metrics['loss'], 'b-', label='Training Loss', 
                linewidth=2, marker='o', markersize=6)
        ax1.plot(epochs, self.val_metrics['loss'], 'r-', label='Validation Loss', 
                linewidth=2, marker='o', markersize=6)
        ax1.set_title('Loss Curves', fontsize=14, pad=10)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_ylim(min_loss - loss_margin, max_loss + loss_margin)
        
        # Add best validation loss point
        best_epoch = np.argmin(self.val_metrics['loss'])
        best_val_loss = min(self.val_metrics['loss'])
        ax1.plot(best_epoch + 1, best_val_loss, 'r*', markersize=15, 
                label=f'Best Val Loss: {best_val_loss:.4f}')
        ax1.axvline(x=best_epoch + 1, color='g', linestyle='--', alpha=0.3)
        
        # Plot Accuracy
        min_acc = min(min(self.train_metrics['accuracy']), min(self.val_metrics['accuracy']))
        max_acc = max(max(self.train_metrics['accuracy']), max(self.val_metrics['accuracy']))
        acc_margin = (max_acc - min_acc) * 0.1
        
        ax2.plot(epochs, self.train_metrics['accuracy'], 'b-', label='Training Accuracy', 
                linewidth=2, marker='o', markersize=6)
        ax2.plot(epochs, self.val_metrics['accuracy'], 'r-', label='Validation Accuracy', 
                linewidth=2, marker='o', markersize=6)
        ax2.set_title('Accuracy Curves', fontsize=14, pad=10)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.set_ylim(min_acc - acc_margin, max_acc + acc_margin)
        
        # Add best validation accuracy point
        best_acc_epoch = np.argmax(self.val_metrics['accuracy'])
        best_val_acc = max(self.val_metrics['accuracy'])
        ax2.plot(best_acc_epoch + 1, best_val_acc, 'r*', markersize=15, 
                label=f'Best Val Acc: {best_val_acc:.4f}')
        ax2.axvline(x=best_acc_epoch + 1, color='g', linestyle='--', alpha=0.3)
        
        # Add final metrics as text
        final_metrics = (
            f"Final Metrics:\n"
            f"Train Loss: {self.train_metrics['loss'][-1]:.4f}\n"
            f"Val Loss: {self.val_metrics['loss'][-1]:.4f}\n"
            f"Train Acc: {self.train_metrics['accuracy'][-1]:.4f}\n"
            f"Val Acc: {self.val_metrics['accuracy'][-1]:.4f}"
        )
        fig.text(0.02, 0.02, final_metrics, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = plots_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log enhanced plots to wandb
        wandb.log({
            "charts/training_curves": wandb.Image(str(plot_path)),
            "metrics/train_loss": self.train_metrics['loss'][-1],
            "metrics/val_loss": self.val_metrics['loss'][-1],
            "metrics/train_accuracy": self.train_metrics['accuracy'][-1],
            "metrics/val_accuracy": self.val_metrics['accuracy'][-1],
            "metrics/best_val_loss": best_val_loss,
            "metrics/best_val_accuracy": best_val_acc,
            "metrics/best_loss_epoch": best_epoch + 1,
            "metrics/best_accuracy_epoch": best_acc_epoch + 1
        })

    def _get_grad_norm(self) -> float:
        """Calculate gradient norm for monitoring."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

class nullcontext:
    """Context manager that does nothing"""
    def __enter__(self):
        return None
    def __exit__(self, *excinfo):
        pass
