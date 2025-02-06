from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import BertTokenizer, BertModel
from model import FinetunedBERT
from trainer import BERTTrainer
import logging
import traceback
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import argparse
from tqdm.auto import tqdm
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split
import hashlib
import pickle
from transformers import PreTrainedModel, PreTrainedTokenizer
import optuna
from optuna_manager import OptunaManager
import concurrent.futures
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Custom Dataset for text classification"""
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 512,
        cache_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        logger.info("Initializing TextDataset")
        try:
            # Store original texts
            self.texts = texts
            
            # Cache version to handle format changes
            CACHE_VERSION = 2  # Increment when cache format changes
            
            if cache_dir is not None:
                # Create cache key from data and parameters
                cache_key = hashlib.md5(
                    str((texts, labels, tokenizer.name_or_path, max_length, CACHE_VERSION)).encode()
                ).hexdigest()
                cache_path = cache_dir / f"dataset_{cache_key}.pkl"
                
                # Try to load from cache
                if cache_path.exists():
                    logger.info("Loading dataset from cache")
                    try:
                        with open(cache_path, 'rb') as f:
                            cached_data = pickle.load(f)
                        # Verify version and required fields
                        if cached_data.get('version') == CACHE_VERSION and all(k in cached_data for k in ['encodings', 'labels', 'texts']):
                            self.encodings = cached_data['encodings']
                            self.labels = cached_data['labels']
                            self.texts = cached_data['texts']
                            logger.info("Successfully loaded dataset from cache")
                        else:
                            logger.warning("Cache format mismatch - processing without cache")
                            cache_path.unlink()  # Remove invalid cache file
                            raise KeyError("Missing required fields in cache")
                    except (KeyError, pickle.UnpicklingError) as e:
                        logger.warning(f"Failed to load cache: {str(e)}")
                        # Process without cache since loading failed
                        with tqdm(total=1, desc="Tokenizing texts", leave=False) as pbar:
                            self.encodings = tokenizer(
                                texts,
                                truncation=True,
                                padding=True,
                                max_length=max_length,
                                return_tensors='pt'
                            )
                            self.labels = torch.tensor(labels, dtype=torch.long)
                            pbar.update(1)
                        
                        # Save to cache
                        with open(cache_path, 'wb') as f:
                            pickle.dump({
                                'version': CACHE_VERSION,
                                'encodings': self.encodings,
                                'labels': self.labels,
                                'texts': self.texts
                            }, f)
                else:
                    # Create and cache dataset
                    with tqdm(total=1, desc="Tokenizing texts", leave=False) as pbar:
                        self.encodings = tokenizer(
                            texts,
                            truncation=True,
                            padding=True,
                            max_length=max_length,
                            return_tensors='pt'
                        )
                        self.labels = torch.tensor(labels, dtype=torch.long)
                        pbar.update(1)
                    
                    # Save to cache
                    with open(cache_path, 'wb') as f:
                            pickle.dump({
                                'version': CACHE_VERSION,
                                'encodings': self.encodings,
                                'labels': self.labels,
                                'texts': self.texts
                            }, f)
            else:
                # Process without caching
                with tqdm(total=1, desc="Tokenizing texts", leave=False) as pbar:
                    self.encodings = tokenizer(
                        texts,
                        truncation=True,
                        padding=True,
                        max_length=max_length,
                        return_tensors='pt'
                    )
                    self.labels = torch.tensor(labels, dtype=torch.long)
                    pbar.update(1)
            logger.info("TextDataset initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TextDataset: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        try:
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            item['text'] = self.texts[idx]
            return item
        except Exception as e:
            logger.error(f"Error getting item at index {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def __len__(self) -> int:
        """Get the length of the dataset"""
        return len(self.labels)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Setting up logging")
    try:
        log_path = Path(config['output_dir']) / 'training.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        logger.info("Logging setup completed")
    except Exception as e:
        logger.error(f"Error setting up logging: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def cache_pretrained_model(
    model_name: str,
    cache_dir: Path
) -> Path:
    """Cache pretrained model files locally"""
    logger.info(f"Caching pretrained model {model_name}")
    try:
        # Create model-specific cache directory
        model_cache_dir = cache_dir / 'models' / model_name.replace('/', '_')
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and save tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        
        # Save tokenizer and model to cache directory
        tokenizer.save_pretrained(model_cache_dir)
        model.save_pretrained(model_cache_dir)
        
        logger.info(f"Model and tokenizer cached at {model_cache_dir}")
        return model_cache_dir
    except Exception as e:
        logger.error(f"Error caching model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def prepare_data(
    texts: List[str],
    labels: List[int],
    config: Dict,
    tokenizer: BertTokenizer,
    cache_dir: Optional[Path] = None
) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and validation dataloaders"""
    logger.info("Preparing data")
    try:
        # Split data
        indices = np.random.permutation(len(texts))
        split_idx = int(len(texts) * (1 - config['val_split']))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # Create datasets with progress bars
        with tqdm(total=2, desc="Creating datasets", leave=False) as pbar:
            train_dataset = TextDataset(
                [texts[i] for i in train_indices],
                [labels[i] for i in train_indices],
                tokenizer,
                config['max_length'],
                cache_dir=cache_dir
            )
            pbar.update(1)
            
            val_dataset = TextDataset(
                [texts[i] for i in val_indices],
                [labels[i] for i in val_indices],
                tokenizer,
                config['max_length'],
                cache_dir=cache_dir
            )
            pbar.update(1)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True
        )

        logger.info("Data preparation completed successfully")
        return train_loader, val_loader
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def setup_optimizer(
    model: FinetunedBERT,
    config: Dict,
    num_training_steps: int
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Setup optimizer and scheduler"""
    logger.info("Setting up optimizer and scheduler")
    try:
        # Differential learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': float(config['weight_decay'])
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        # Create optimizer
        # Convert learning rate to float
        learning_rate = float(config['learning_rate'])
        adam_beta1 = float(config['adam_beta1'])
        adam_beta2 = float(config['adam_beta2'])
        adam_epsilon = float(config['adam_epsilon'])

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon
        )

        # Create scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=num_training_steps,
            pct_start=float(config['warmup_ratio'])
        )

        logger.info("Optimizer and scheduler setup completed")
        return optimizer, scheduler
    except Exception as e:
        logger.error(f"Error setting up optimizer: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main(config_path: str, study_name: str = None, storage_url: str = None, n_trials: int = None, n_jobs: int = None):
    """Main training function"""
    logger.info(f"Starting optimization/training with config from {config_path}")
    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Config: {json.dumps(config, indent=2)}")

        # Setup output directory and logging
        output_dir = Path('/content/drive/MyDrive/outputs') if config['output_dir'].startswith('/content/') else Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Configure logging with both file and console handlers
        log_file = output_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Outputs will be saved to {output_dir.absolute()}")

        # Set and log device
        device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Initialize components with progress bar
        setup_pbar = tqdm(total=4, desc="Setting up training", leave=False)

        # Set up cache directory
        cache_dir = Path(config['data']['cache_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache and load pretrained model and tokenizer
        model_cache_dir = cache_pretrained_model(config['model_name'], cache_dir)
        tokenizer = BertTokenizer.from_pretrained(model_cache_dir)
        setup_pbar.update(1)

        # Initialize model from cache
        model = FinetunedBERT(
            num_labels=config['num_labels'],
            pretrained_model=str(model_cache_dir),
            dropout_rate=config['dropout_rate'],
            hidden_dim=config['hidden_dim'],
            activation_monitoring=True
        ).to(device)
        setup_pbar.update(1)

        # Load and prepare data with caching
        logger.info("Loading and preparing data")
        
        # Load and split datasets
        logger.info("Loading and splitting datasets")
        if config['data']['val_csv_path']:
            # Use separate validation file
            train_df = pd.read_csv(config['data']['train_csv_path'])
            val_df = pd.read_csv(config['data']['val_csv_path'])
            
            train_df[config['data']['text_column']] = train_df[config['data']['text_column']].astype(str)
            val_df[config['data']['text_column']] = val_df[config['data']['text_column']].astype(str)
            
            train_texts = train_df[config['data']['text_column']].tolist()
            train_labels = np.array(train_df[config['data']['label_column']].tolist()) - 1
            val_texts = val_df[config['data']['text_column']].tolist()
            val_labels = np.array(val_df[config['data']['label_column']].tolist()) - 1
        else:
            # Split training data
            df = pd.read_csv(config['data']['train_csv_path'])
            df[config['data']['text_column']] = df[config['data']['text_column']].astype(str)
            labels = np.array(df[config['data']['label_column']].tolist()) - 1
            
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                df[config['data']['text_column']].tolist(),
                labels,
                test_size=config['val_split'],
                random_state=config['seed'],
                stratify=labels
            )
        
        logger.info(f"Loaded {len(train_texts)} training examples and {len(val_texts)} validation examples")
        
        # Create datasets using our custom TextDataset
        train_dataset = TextDataset(
            train_texts,
            train_labels,
            tokenizer,
            config['max_length'],
            cache_dir=cache_dir
        )
        val_dataset = TextDataset(
            val_texts,
            val_labels,
            tokenizer,
            config['max_length'],
            cache_dir=cache_dir
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True
        )
        setup_pbar.update(1)

        # Setup optimizer and scheduler
        num_training_steps = len(train_loader) * config['num_epochs']
        optimizer, scheduler = setup_optimizer(model, config, num_training_steps)
        setup_pbar.update(1)

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

        if study_name and storage_url:
            # Optuna hyperparameter optimization mode
            logger.info(f"Running Optuna optimization with study {study_name}")
            
            # Initialize Optuna manager
            optuna_manager = OptunaManager(
                study_name=study_name,
                storage_url=storage_url,
                base_config=config,
                direction="minimize"
            )
            
            def run_trial(trial):
                return optuna_manager.objective(
                    trial=trial,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device
                )
            
            # Run optimization with concurrent trials
            n_jobs = n_jobs or 1
            n_trials = n_trials or 100
            
            try:
                if n_jobs > 1:
                    study = optuna_manager.study
                    logger.info(f"Running {n_trials} trials with {n_jobs} concurrent workers")
                    
                    def optimize_trial(worker_id: int):
                        # Set worker ID in environment for signal handler check
                        os.environ['OPTUNA_WORKER_ID'] = str(worker_id)
                        try:
                            study.optimize(run_trial, n_trials=1)
                            return True
                        except Exception as e:
                            logger.error(f"Trial failed in worker {worker_id}: {str(e)}")
                            return False
                        finally:
                            # Clean up environment variable
                            if 'OPTUNA_WORKER_ID' in os.environ:
                                del os.environ['OPTUNA_WORKER_ID']
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                        futures = [executor.submit(optimize_trial, i) for i in range(n_trials)]
                        completed = 0
                        for future in concurrent.futures.as_completed(futures):
                            completed += 1
                            try:
                                success = future.result()
                                logger.info(f"Trial {completed}/{n_trials} {'completed' if success else 'failed'}")
                            except Exception as e:
                                logger.error(f"Error in trial {completed}: {str(e)}")
                else:
                    logger.info(f"Running {n_trials} sequential trials")
                    optuna_manager.study.optimize(run_trial, n_trials=n_trials, show_progress_bar=True)
                
                # Check if we have any completed trials
                completed_trials = [t for t in optuna_manager.study.trials if t.state.is_finished()]
                if completed_trials:
                    best_trial = optuna_manager.study.best_trial
                    logger.info(f"Optimization completed! Best trial value: {best_trial.value}")
                    logger.info(f"Number of completed trials: {len(completed_trials)}")
                    logger.info(f"Number of pruned trials: {len([t for t in optuna_manager.study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
                    
                    # Save parameters and statistics
                    optuna_manager.save_best_parameters(output_dir)
                else:
                    logger.warning("No trials completed successfully. No parameters to save.")
            except Exception as e:
                logger.error(f"Error during optimization: {str(e)}")
                logger.error(traceback.format_exc())
            
        else:
            # Standard single training mode
            metrics = trainer.train(config['num_epochs'])
            logger.info("Training completed successfully!")
            logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        
        setup_pbar.close()
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--study_name', type=str, help='Optuna study name')
    parser.add_argument('--storage', type=str, help='Optuna storage URL')
    parser.add_argument('--n_trials', type=int, help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, help='Number of concurrent jobs')
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        study_name=args.study_name,
        storage_url=args.storage,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs
    )
