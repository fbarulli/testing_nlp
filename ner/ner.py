# text_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import spacy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import os
import asyncio
from functools import lru_cache

# Ensure CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU visibility
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SentimentAnalysisResults:
    """Container for sentiment analysis results"""
    predictions: np.ndarray
    probabilities: np.ndarray
    classification_report: dict
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]
    rating_distribution: Dict[int, int]
    text_length_stats: Dict[str, float]
    sentiment_by_length: Dict[str, Dict[int, float]]

class TextAnalyzer:
    def __init__(self, num_processes: Optional[int] = None):
        """
        Initialize the TextAnalyzer with CPU-optimized settings.

        Args:
            num_processes: Number of processes for multiprocessing. 
                           Defaults to the number of CPU cores.
        """
        self.num_processes = num_processes or os.cpu_count()
        
        # Load spaCy model with CPU-only settings
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        self.nlp.use_gpu = False  # Explicitly disable GPU
        
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            strip_accents='unicode',
            min_df=5,
            max_df=0.9
        )
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            n_jobs=self.num_processes  # Use multiple cores for training
        )
        
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """Preprocess text with basic cleaning and lemmatization (cached)."""
        try:
            doc = self.nlp(text.lower().strip())
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct 
                     and token.lemma_.strip()]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error preprocessing text: {text}\nError: {e}")
            traceback.print_exc()
            return text  # Return original text if preprocessing fails

    async def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts in parallel using multiprocessing (cached)."""
        try:
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(executor, self.preprocess_text, text)
                    for text in texts
                ]
                processed_texts = []
                for future in tqdm(asyncio.as_completed(futures), total=len(futures), desc="Preprocessing texts"):
                    processed_texts.append(await future)
            return processed_texts
        except Exception as e:
            logger.error(f"Error in batch preprocessing: {e}")
            traceback.print_exc()
            raise

    @lru_cache(maxsize=1000)
    def extract_feature_importance(self, feature_names: Tuple[str]) -> Dict[str, float]:
        """Extract feature importance from the logistic regression model (cached)."""
        try:
            if not hasattr(self.model, 'coef_'):
                return {}
                
            importance_per_class = np.abs(self.model.coef_)
            overall_importance = np.mean(importance_per_class, axis=0)
            
            return dict(sorted(
                zip(feature_names, overall_importance),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:100])  # Return top 100 features
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
            traceback.print_exc()
            return {}

    async def train_and_analyze(
        self,
        df: pd.DataFrame,
        text_column: str,
        rating_column: str,
        test_size: float = 0.2,
        batch_size: int = 1000
    ) -> SentimentAnalysisResults:
        """Train the model and perform comprehensive sentiment analysis with batch processing and caching."""
        try:
            logger.info("Starting sentiment analysis pipeline")
            
            # Prepare labels
            ratings = df[rating_column].values
            encoded_ratings = self.label_encoder.fit_transform(ratings)
            
            # Split data into train and test sets
            train_indices, test_indices = train_test_split(
                np.arange(len(df)),
                test_size=test_size,
                stratify=encoded_ratings,
                random_state=42
            )
            
            # Initialize containers for batch processing
            X_train, X_test = [], []
            y_train, y_test = encoded_ratings[train_indices], encoded_ratings[test_indices]
            lengths = []
            sentiment_by_length = defaultdict(lambda: defaultdict(float))
            
            # Process data in batches
            for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
                batch_texts = df[text_column].iloc[i:i + batch_size].tolist()
                batch_ratings = ratings[i:i + batch_size]
                
                # Preprocess batch
                processed_batch = await self.preprocess_batch(batch_texts)
                
                # Calculate lengths and sentiment distribution
                for text, rating in zip(processed_batch, batch_ratings):
                    length = len(text.split())
                    lengths.append(length)
                    
                    # Handle edge case where all lengths are the same
                    if len(set(lengths)) == 1:
                        length_bin = 'uniform_length'  # Assign a single bin
                    else:
                        try:
                            length_bin = pd.qcut(
                                lengths, q=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'],
                                duplicates='drop'  # Drop duplicate edges
                            )[-1]  # Get the bin for the current text
                        except ValueError:
                            # Fallback to equal-width binning if qcut fails
                            length_bin = pd.cut(
                                lengths, bins=5, labels=['very_short', 'short', 'medium', 'long', 'very_long']
                            )[-1]
                    
                    sentiment_by_length[str(length_bin)][rating] += 1
                
                # Append to train or test sets
                for idx, text in zip(range(i, i + len(batch_texts)), processed_batch):
                    if idx in train_indices:
                        X_train.append(text)
                    elif idx in test_indices:
                        X_test.append(text)
            
            # Vectorize texts
            logger.info("Vectorizing texts...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            logger.info("Training model...")
            self.model.fit(X_train_vec, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_vec)
            y_pred_proba = self.model.predict_proba(X_test_vec)
            
            # Get feature importance (cached)
            feature_importance = self.extract_feature_importance(
                tuple(self.vectorizer.get_feature_names_out())
            )
            
            # Analyze text lengths
            text_length_analysis = {
                'mean_length': np.mean(lengths),
                'median_length': np.median(lengths),
                'std_length': np.std(lengths),
                'length_by_rating': {
                    rating: np.mean([l for l, r in zip(lengths, ratings) if r == rating])
                    for rating in set(ratings)
                }
            }
            
            # Normalize sentiment distribution counts
            for length_bin in sentiment_by_length:
                total = sum(sentiment_by_length[length_bin].values())
                sentiment_by_length[length_bin] = {
                    rating: count / total
                    for rating, count in sentiment_by_length[length_bin].items()
                }
            
            # Prepare results
            results = SentimentAnalysisResults(
                predictions=y_pred,
                probabilities=y_pred_proba,
                classification_report=classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    target_names=[str(label) for label in self.label_encoder.classes_]
                ),
                confusion_matrix=confusion_matrix(y_test, y_pred),
                feature_importance=feature_importance,
                rating_distribution=Counter(ratings),
                text_length_stats=text_length_analysis,
                sentiment_by_length=dict(sentiment_by_length)
            )
            
            # Log summary statistics
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(
                y_test,
                y_pred,
                target_names=[str(label) for label in self.label_encoder.classes_]
            ))
            
            logger.info("\nTop predictive features:")
            for feature, importance in list(feature_importance.items())[:10]:
                logger.info(f"{feature}: {importance:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            traceback.print_exc()
            raise