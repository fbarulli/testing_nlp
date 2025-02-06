# BERT Finetuning Framework

A comprehensive framework for finetuning BERT models with advanced monitoring and optimization capabilities.

## Features

- **Advanced Model Architecture**
  - Residual connections for better gradient flow
  - Layer normalization for training stability
  - Gradient checkpointing for memory efficiency
  - Configurable hidden layers and dropout

- **Detailed Monitoring**
  - Layer activation tracking
  - Gradient norm monitoring
  - Weight distribution analysis
  - Training metrics visualization

- **Optimization Techniques**
  - Mixed precision training (FP16)
  - Gradient accumulation
  - Layer-wise learning rate decay
  - Weight decay scheduling
  - Label smoothing

- **Training Features**
  - Early stopping with configurable patience
  - Learning rate scheduling with warmup
  - Stochastic Weight Averaging (SWA) support
  - Automatic mixed precision training
  - Gradient clipping
  - Hyperparameter optimization with Optuna
  - Concurrent training with multiple trials
  - SQLite storage for optimization history

## Project Structure

```
bert_finetuning/
├── src/
│   ├── model.py       # BERT model architecture with monitoring
│   ├── trainer.py     # Training loop and optimization logic
│   └── main.py        # Main script for training
├── config.yaml        # Configuration file
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Modify the configuration in `config.yaml` according to your needs:
   - Model architecture settings
   - Training hyperparameters
   - Optimization settings
   - Monitoring preferences

2. Choose your training mode:

   a. Standard Training:
   ```bash
   python src/main.py --config config.yaml
   ```

   b. Hyperparameter Optimization:
   ```bash
   python src/main.py --config config.yaml \
                      --study_name "bert_optimization" \
                      --storage "sqlite:///optuna.db" \
                      --n_trials 100 \
                      --n_jobs 4
   ```

   The hyperparameter optimization mode supports:
   - `--study_name`: Name of the Optuna study (required for optimization)
   - `--storage`: SQLite database URL for storing results
   - `--n_trials`: Number of optimization trials to run (default: 100)
   - `--n_jobs`: Number of concurrent trials (default: 1)

## Model Architecture

The framework implements an enhanced BERT model with:

1. **Base BERT Layer**
   - Pretrained BERT model as the foundation
   - Gradient checkpointing for memory efficiency
   - Full attention mechanism

2. **Feature Extraction**
   - Residual blocks for better gradient flow
   - Layer normalization for stability
   - GELU activation functions

3. **Classification Head**
   - Adaptive hidden layers
   - Dropout for regularization
   - Label smoothing support

## Hyperparameter Optimization

The framework uses Optuna for efficient hyperparameter optimization:

1. **Optimized Parameters**
   - Learning rates (main training and pretraining)
   - Dropout rates
   - Weight decay
   - Batch sizes
   - Model architecture (hidden dimensions, layers)
   - MLM and contrastive learning parameters

2. **Advanced Optimization Features**
   - Tree-structured Parzen Estimators (TPE) with multivariate optimization
   - Intelligent parameter sampling strategies:
     * Log-scale sampling for learning rates and weight decay
     * Linear-scale sampling with steps for dropout and ratios
     * Categorical sampling for architectural choices
   - Concurrent trial execution with constant liar algorithm
   - Adaptive pruning of unpromising trials
   - Study persistence with SQLite
   - Parameter importance analysis
   - Optimization history visualization
   - Comprehensive trial statistics tracking

3. **Best Practices**
   - Start with 10 random trials before TPE optimization
   - Use multiple trials (recommended: 100+) for robust results
   - Enable multivariate optimization for parameter relationships
   - Leverage parallel execution with constant liar algorithm
   - Monitor optimization progress through SQLite database
   - Analyze parameter importance for future refinement
   - Review trial statistics and pruning rates
   - Examine best trial parameters and timing information

## Training Optimizations

The framework includes several optimizations:

1. **Memory Efficiency**
   - Gradient checkpointing
   - Mixed precision training
   - Efficient data loading

2. **Training Stability**
   - Learning rate warmup
   - Gradient clipping
   - Layer-wise learning rate decay

3. **Performance Monitoring**
   - Activation statistics
   - Gradient norms
   - Layer-wise metrics

## Monitoring and Visualization

The training process provides detailed insights:

1. **Training Metrics**
   - Loss curves
   - Accuracy tracking
   - Learning rate schedules

2. **Layer Analysis**
   - Activation patterns
   - Gradient flow
   - Weight distributions

3. **Model Behavior**
   - Attention patterns
   - Feature representations
   - Classification confidence

## Customization

The framework can be customized in several ways:

1. **Data Processing**
   - Modify TextDataset in main.py
   - Adjust tokenization parameters
   - Add data augmentation

2. **Model Architecture**
   - Add custom layers
   - Modify attention mechanism
   - Change pooling strategy

3. **Training Process**
   - Custom loss functions
   - New optimization strategies
   - Additional metrics

## Best Practices

1. **Training Setup**
   - Start with small learning rates (1e-5 to 5e-5)
   - Use gradient accumulation for larger batches
   - Enable mixed precision training

2. **Model Configuration**
   - Use residual connections
   - Apply layer normalization
   - Enable gradient checkpointing for large models

3. **Monitoring**
   - Watch for gradient norms
   - Monitor activation statistics
   - Track validation metrics

## Contributing

Feel free to contribute by:
- Opening issues for bugs or suggestions
- Submitting pull requests with improvements
- Adding new features or optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
