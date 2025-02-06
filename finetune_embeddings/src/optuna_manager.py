from __future__ import annotations
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from model import FinetunedBERT
from trainer import BERTTrainer
import os

logger = logging.getLogger(__name__)

class OptunaManager:
    """Manages Optuna hyperparameter optimization studies"""
    
    def __init__(
        self,
        study_name: str,
        base_config: Dict[str, Any],
        storage_dir: Optional[Path] = None,
        direction: str = "minimize",
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        seed: int = 42
    ):
        """
        Initialize the Optuna study manager.
        
        Args:
            study_name: Name of the study
            storage_url: URL for the study storage (SQLite)
            base_config: Base configuration dictionary
            direction: Optimization direction ("minimize" or "maximize")
        """
        self.base_config = base_config
        # Configure advanced TPE sampler
        sampler = TPESampler(
            n_startup_trials=n_startup_trials,  # Number of random trials before TPE starts
            n_ei_candidates=n_ei_candidates,    # Number of candidates for expected improvement
            multivariate=True,                  # Enable multivariate optimization
            seed=seed,                          # Set random seed for reproducibility
            consider_endpoints=True,            # Consider endpoint values in range
            constraints_func=None,              # Can be used to add parameter constraints
            warn_independent_sampling=True,     # Warn if parameters are sampled independently
            constant_liar=True                  # Handle parallel optimization better
        )
        
        # Setup storage in project directory
        if storage_dir is None:
            storage_dir = Path('results') / 'optuna_storage'
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Use SQLite storage with proper path handling
        db_path = storage_dir / f"{study_name}.db"
        storage_url = f"sqlite:///{db_path.absolute()}"
        logger.info(f"Optuna database will be stored at: {db_path}")
        
        try:
            # Try to load existing study
            self.study = optuna.load_study(
                study_name=study_name,
                storage=storage_url
            )
            logger.info(f"Loaded existing study '{study_name}' from {db_path}")
            logger.info(f"Found {len(self.study.trials)} existing trials")
        except Exception:
            # Create new study if it doesn't exist
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction=direction,
                sampler=sampler,
                load_if_exists=True
            )
            logger.info(f"Created new study '{study_name}' at {db_path}")
        
    def suggest_parameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Note: Parameter ranges and distributions are carefully chosen based on:
        1. Common BERT finetuning practices
        2. Literature recommendations
        3. Computational constraints
        """
        # Create a deep copy of base config and ensure numeric types
        config = self.base_config.copy()
        
        # Ensure base numeric parameters are float
        for key in ['learning_rate', 'weight_decay', 'dropout_rate', 'warmup_ratio',
                   'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm',
                   'label_smoothing']:
            if key in config:
                config[key] = float(config[key])
        
        # Ensure integer parameters
        for key in ['num_epochs', 'batch_size', 'num_labels', 'gradient_accumulation_steps']:
            if key in config:
                config[key] = int(config[key])
        
        # Core hyperparameters with informed distributions
        # Learning rate - log scale without step
        config['learning_rate'] = float(trial.suggest_float(
            'learning_rate', 1e-6, 1e-4, log=True
        ))
        
        # Weight decay - log scale without step
        config['weight_decay'] = float(trial.suggest_float(
            'weight_decay', 1e-4, 1e-2, log=True
        ))
        
        # Dropout rate - linear scale with step
        config['dropout_rate'] = float(trial.suggest_float(
            'dropout_rate', 0.1, 0.5,
            step=0.05  # Step size based on common practice
        ))
        
        # Warmup ratio - linear scale with step
        config['warmup_ratio'] = float(trial.suggest_float(
            'warmup_ratio', 0.05, 0.2,
            step=0.01  # Fine control over warmup
        ))
        
        # Architectural choices
        config['batch_size'] = int(trial.suggest_categorical(
            'batch_size', [16, 24, 32, 48, 64]  # More granular options
        ))
        config['hidden_dim'] = int(trial.suggest_categorical(
            'hidden_dim', [128, 256, 384, 512, 768]  # More options including BERT sizes
        ))
        
        # MLM specific parameters if enabled
        if config['pretraining']['mlm']['enabled']:
            # MLM probability - linear scale with step
            config['pretraining']['mlm']['probability'] = float(trial.suggest_float(
                'mlm_probability', 0.1, 0.2,
                step=0.02  # Fine-grained control over masking
            ))
            
            # MLM learning rate - log scale without step
            config['pretraining']['mlm']['learning_rate'] = float(trial.suggest_float(
                'mlm_learning_rate', 1e-5, 1e-4, log=True
            ))
            
            # Ensure other MLM parameters are float
            for key in ['weight_decay', 'max_grad_norm', 'warmup_ratio']:
                if key in config['pretraining']['mlm']:
                    config['pretraining']['mlm'][key] = float(config['pretraining']['mlm'][key])
        
        # Contrastive learning specific parameters if enabled
        if config['pretraining']['contrastive']['enabled']:
            # Temperature - linear scale with step
            config['pretraining']['contrastive']['temperature'] = float(trial.suggest_float(
                'contrastive_temperature', 0.05, 0.1,
                step=0.01  # Fine control over temperature
            ))
            
            # Loss weight - linear scale with step
            config['pretraining']['contrastive']['loss_weight'] = float(trial.suggest_float(
                'contrastive_loss_weight', 0.05, 0.2,
                step=0.025  # Balanced steps for loss weight
            ))
            
            # Additional contrastive parameters
            if 'queue_size' in config['pretraining']['contrastive']:
                config['pretraining']['contrastive']['queue_size'] = int(trial.suggest_categorical(
                    'contrastive_queue_size', [32768, 65536, 131072]
                ))
            
            # Ensure other contrastive parameters are float
            for key in ['learning_rate', 'weight_decay', 'warmup_ratio']:
                if key in config['pretraining']['contrastive']:
                    config['pretraining']['contrastive'][key] = float(config['pretraining']['contrastive'][key])
        
        return config

    def objective(
        self,
        trial: Trial,
        full_dataset: torch.utils.data.Dataset,
        labels: List[int],
        batch_size: int,
        device: torch.device,
        n_splits: int = 5
    ) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial object
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Torch device
            
        Returns:
            Validation metric to optimize
        """
        # Get suggested parameters
        config = self.suggest_parameters(trial)
        
        try:
            # Setup cross-validation
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = []
            
            # Perform cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), labels)):
                logger.info(f"Starting fold {fold_idx + 1}/{n_splits}")
                
                # Create data loaders for this fold
                train_subset = Subset(full_dataset, train_idx)
                val_subset = Subset(full_dataset, val_idx)
                
                train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4
                )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=batch_size,
                    num_workers=4
                )
                
                # Initialize model with trial parameters
                model = FinetunedBERT(
                    num_labels=config['num_labels'],
                    pretrained_model=config['model_name'],
                    dropout_rate=config['dropout_rate'],
                    hidden_dim=config['hidden_dim'],
                    activation_monitoring=True
                ).to(device)
                
                # Setup optimizer and scheduler
                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {
                        'params': [p for n, p in model.named_parameters() 
                                  if not any(nd in n for nd in no_decay)],
                        'weight_decay': config['weight_decay']
                    },
                    {
                        'params': [p for n, p in model.named_parameters() 
                                  if any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0
                    }
                ]
                
                # Ensure numeric types for optimizer parameters
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=float(config['learning_rate']),
                    betas=(float(config['adam_beta1']), float(config['adam_beta2'])),
                    eps=float(config['adam_epsilon'])
                )
                
                num_training_steps = len(train_loader) * config['num_epochs']
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(config['learning_rate']),
                    total_steps=int(num_training_steps),
                    pct_start=float(config['warmup_ratio'])
                )
                
                # Initialize trainer
                trainer = BERTTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    config=config
                )
            
                # Initialize wandb for this fold
                wandb.init(
                    project="all_embeddings_tune",
                    name=f"trial_{trial.number}_fold_{fold_idx}",
                    config={
                        **config,
                        'trial_number': trial.number,
                        'fold': fold_idx,
                        'study_name': self.study.study_name
                    },
                    reinit=True
                )
                
                # Train and get metrics
                metrics = trainer.train(int(config['num_epochs']))
                
                # Report intermediate values
                for epoch, epoch_metrics in enumerate(metrics):
                    trial.report(epoch_metrics['val_loss'], epoch)
                    
                    # Handle pruning
                    if trial.should_prune():
                        wandb.finish()
                        raise optuna.TrialPruned()
                
                # Log final metrics to wandb
                wandb.log({
                    'fold': fold_idx,
                    'fold_val_loss': trainer.best_val_loss,
                    'trial_completed': True
                })
                wandb.finish()
                
                cv_scores.append(trainer.best_val_loss)
                logger.info(f"Fold {fold_idx + 1} completed with val_loss: {trainer.best_val_loss:.4f}")
            
            # Calculate mean validation loss across folds
            mean_val_loss = sum(cv_scores) / len(cv_scores)
            std_val_loss = torch.tensor(cv_scores).std().item()
            
            # Log cross-validation summary
            wandb.init(
                project="all_embeddings_tune",
                name=f"trial_{trial.number}_summary",
                config={
                    **config,
                    'trial_number': trial.number,
                    'study_name': self.study.study_name
                },
                reinit=True
            )
            wandb.log({
                'mean_val_loss': mean_val_loss,
                'std_val_loss': std_val_loss,
                'cv_scores': cv_scores,
                'trial_completed': True
            })
            wandb.finish()
            
            return mean_val_loss
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            # Clean up wandb on error
            try:
                wandb.finish()
            except:
                pass
            raise optuna.TrialPruned()

    def save_best_parameters(self, output_dir: Optional[Path] = None) -> None:
        """
        Save the best trial parameters to a file if any trials have completed.
        
        Args:
            output_dir: Directory to save the parameters
        """
        # Use storage directory if output_dir not provided
        if output_dir is None:
            output_dir = Path('optuna_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logger.warning("No completed trials found. Skipping parameter saving.")
            return
            
        try:
            # Get best trial from completed trials only
            best_trial = min(completed_trials, key=lambda t: t.value)
            
            # Save best parameters with more details
            best_params = {
                'value': best_trial.value,
                'params': best_trial.params,
                'number': best_trial.number,
                'datetime_start': best_trial.datetime_start.isoformat(),
                'datetime_complete': best_trial.datetime_complete.isoformat(),
                'study_name': self.study.study_name,
                'direction': self.study.direction.name,
                'system_attrs': best_trial.system_attrs,
                'user_attrs': best_trial.user_attrs
            }
            
            output_path = output_dir / 'best_params.yaml'
            with open(output_path, 'w') as f:
                yaml.dump(best_params, f, default_flow_style=False)
            
            logger.info(f"Best parameters saved to {output_path}")
            
            # Also save study statistics
            study_stats = {
                'n_trials': len(self.study.trials),
                'n_complete': len(completed_trials),
                'n_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'best_value': best_trial.value
            }
            
            stats_path = output_dir / 'study_stats.yaml'
            with open(stats_path, 'w') as f:
                yaml.dump(study_stats, f, default_flow_style=False)
                
            logger.info(f"Study statistics saved to {stats_path}")
            
        except Exception as e:
            logger.error(f"Error saving parameters: {str(e)}")
            logger.error("This may occur if no trials have completed successfully.")

    def load_best_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and apply best parameters from previous trials to config.
        
        Args:
            config: Base configuration to update
            
        Returns:
            Updated configuration with best parameters
        """
        try:
            # Get best trial
            best_trial = self.study.best_trial
            
            # Update config with best parameters
            for param_name, param_value in best_trial.params.items():
                # Handle nested parameters
                if '_' in param_name:
                    # e.g., 'contrastive_temperature' -> ['contrastive', 'temperature']
                    parts = param_name.split('_')
                    if parts[0] in config['pretraining']:
                        # Update nested parameter
                        config['pretraining'][parts[0]][parts[1]] = param_value
                else:
                    # Update top-level parameter
                    config[param_name] = param_value
            
            logger.info(f"Loaded best parameters from trial {best_trial.number} "
                       f"with value {best_trial.value}")
            return config
            
        except Exception as e:
            logger.warning(f"Could not load best parameters: {str(e)}")
            logger.warning("Using base configuration instead")
            return config

    def visualize_study(self, output_dir: Optional[Path] = None) -> None:
        """
        Create and save visualization plots for the study.
        
        Args:
            output_dir: Directory to save the plots
        """
        try:
            import optuna.visualization as vis
            import plotly.io as pio
            
            # Use storage directory if output_dir not provided
            if output_dir is None:
                output_dir = Path('optuna_results')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create visualization plots
            plots = {
                'optimization_history': vis.plot_optimization_history(self.study),
                'parallel_coordinate': vis.plot_parallel_coordinate(self.study),
                'param_importances': vis.plot_param_importances(self.study),
                'slice': vis.plot_slice(self.study),
                'contour': vis.plot_contour(self.study),
                'edf': vis.plot_edf(self.study)
            }
            
            # Save plots
            for name, fig in plots.items():
                plot_path = output_dir / f"{name}.html"
                pio.write_html(fig, str(plot_path))
                logger.info(f"Saved {name} plot to {plot_path}")
            
            # Save parameter relationships plot
            param_plot = vis.plot_param_relationships(
                self.study,
                params=['learning_rate', 'weight_decay', 'dropout_rate', 'warmup_ratio']
            )
            plot_path = output_dir / "param_relationships.html"
            pio.write_html(param_plot, str(plot_path))
            logger.info(f"Saved parameter relationships plot to {plot_path}")
            
        except ImportError:
            logger.warning("Could not create visualizations. Please install plotly.")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
