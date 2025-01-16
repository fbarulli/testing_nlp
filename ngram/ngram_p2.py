
'''
NGram Analysis and Text Processing Module

This module provides tools for analyzing n-grams (contiguous sequences of n words) in text data
and their associations with labels (e.g., ratings, categories). It also includes advanced text
preprocessing functionality to clean and normalize text data before analysis.

The module is designed for applications such as sentiment analysis, topic modeling, and feature
engineering for machine learning models. It combines n-gram extraction, label association analysis,
and text preprocessing into a cohesive workflow.

Key Components:
1. **NGramLabelAnalyzer**:
   - Extracts n-grams from text data and computes their associations with labels.
   - Identifies the most significant n-grams for each label based on their occurrence probabilities.
   - Logs detailed information about the analysis process for debugging and monitoring.

2. **TextProcessor**:
   - Preprocesses text data to clean and normalize it before analysis.
   - Handles contractions, stop words, special characters, and other text normalization tasks.
   - Supports parallel processing for efficient text cleaning on large datasets.

3. **TextProcessorConfig**:
   - A configuration class for customizing text preprocessing settings, such as stop words,
     important terms, and n-gram ranges.

4. **Utilities**:
   - Includes helper functions for text cleaning, similarity calculation, and noun extraction.
   - Provides logging and error handling for robust execution.

Key Features:
- Customizable n-gram ranges (e.g., unigrams, bigrams, trigrams).
- Advanced text preprocessing, including stop word removal and contraction handling.
- Parallel processing for efficient text cleaning and n-gram extraction.
- Detailed logging for monitoring and debugging.
- Flexible configuration for preprocessing and analysis.

Example Use Cases:
- Sentiment analysis: Identify n-grams strongly associated with positive or negative ratings.
- Topic modeling: Extract meaningful n-grams to understand common themes in text data.
- Feature engineering: Generate n-gram features for machine learning models.

Dependencies:
- pandas: For handling data in DataFrames.
- scikit-learn: For n-gram extraction, label encoding, and model evaluation.
- nltk: For text preprocessing and noun extraction.
- shap: For interpreting n-gram importance in model predictions.
- tqdm: For progress tracking during n-gram analysis.
- logging: For detailed logging of the analysis process.

''' 
import shap
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from tqdm import tqdm
import logging
import time
import scipy
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Set, Tuple, Optional
import nltk
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass, field

# Configure logging
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ngram_analysis.log')  # Optional: Log to file
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
except Exception as e:
    print(f"Failed to configure logging: {str(e)}")
    raise

class NGramAnalysisError(Exception):
    """Custom exception for NGram analysis errors"""
    pass

@dataclass
class TextProcessorConfig:
    """Configuration for text processing."""
    ngram_range: Tuple[int, int] = (1, 3)
    top_n: int = 10
    min_df: float = 0.001
    stop_words: List[str] = field(default_factory=list)
    important_terms: Set[str] = field(default_factory=set)
    structural_starts: Set[str] = field(default_factory=set)
    structural_words: Set[str] = field(default_factory=set)

    def __post_init__(self):
        # Ensure important terms are not in stop words
        self.stop_words = [
            word for word in self.stop_words if word not in self.important_terms
        ]

class NGramLabelAnalyzer:
    """
    A class for analyzing the association between n-grams and labels in text data.
    """
    def __init__(self, ngram_range: tuple[int, int] = (1, 3)):
        try:
            self.ngram_range = ngram_range
            self.vectorizer = None
            self.ngram_label_associations = None
            self.label_encoder = LabelEncoder()
            self.text_processor_config = TextProcessorConfig(ngram_range=ngram_range)
            logger.info(f"Initialized NGramLabelAnalyzer with ngram_range={ngram_range}")
        except Exception as e:
            logger.error(f"Failed to initialize NGramLabelAnalyzer: {str(e)}")
            raise NGramAnalysisError(f"Initialization failed: {str(e)}")

    def _validate_input(self, df: pd.DataFrame, text_column: str, rating_column: str) -> None:
        """Validate input data and columns"""
        try:
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in DataFrame")
                
            if rating_column not in df.columns:
                raise ValueError(f"Rating column '{rating_column}' not found in DataFrame")
                
            if df[text_column].isnull().any():
                logger.warning(f"Found {df[text_column].isnull().sum()} null values in text column")
                
            if df[rating_column].isnull().any():
                logger.warning(f"Found {df[rating_column].isnull().sum()} null values in rating column")
                
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise NGramAnalysisError(f"Validation error: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean text using the TextProcessorConfig"""
        if not text or not isinstance(text, str):
            return ''

        text = text.lower().strip()
        words = text.split()
        cleaned_words = [word for word in words if word not in self.text_processor_config.stop_words]
        return ' '.join(cleaned_words).strip()

    def _extract_ngrams(self, texts: list[str]) -> tuple[np.ndarray, list[str]]:
        """Extract n-grams from the input texts."""
        try:
            if not texts:
                raise ValueError("Empty text list provided")

            logger.info(f"Extracting n-grams with range {self.ngram_range}")
            self.vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                stop_words='english',
                min_df=2  # Minimum document frequency
            )
            
            X = self.vectorizer.fit_transform(texts)
            ngrams = self.vectorizer.get_feature_names_out()
            
            if len(ngrams) == 0:
                raise NGramAnalysisError("No n-grams were extracted from the texts")
                
            logger.info(f"Successfully extracted {len(ngrams)} n-grams")
            return X, ngrams
            
        except Exception as e:
            logger.error(f"N-gram extraction failed: {str(e)}")
            raise NGramAnalysisError(f"N-gram extraction error: {str(e)}")

    def analyze(self, df: pd.DataFrame, text_column: str, rating_column: str, top_n: int = 10) -> dict[int, list[tuple[str, float]]]:
        """Main analysis method to find n-gram associations with labels"""
        try:
            logger.info("Starting n-gram label analysis")
            
            # Validate input
            self._validate_input(df, text_column, rating_column)
            
            # Clean and prepare data
            df_clean = df.dropna(subset=[text_column, rating_column])
            if len(df_clean) != len(df):
                logger.warning(f"Dropped {len(df) - len(df_clean)} rows with null values")
            
            # Clean text data
            df_clean[text_column] = df_clean[text_column].apply(self._clean_text)
            texts = df_clean[text_column].tolist()
            labels = df_clean[rating_column].tolist()
            
            # Extract n-grams
            X, ngrams = self._extract_ngrams(texts)
            
            # Initialize tracking
            failure_count = 0
            retry_success_count = 0
            failed_ngrams = []
            total_ngrams = len(ngrams)
            FAILURE_THRESHOLD = 0.1

            # Process n-grams
            ngram_label_associations = defaultdict(lambda: defaultdict(float))
            unique_labels = sorted(set(labels))

            for i, ngram in tqdm(enumerate(ngrams), total=total_ngrams, desc="Computing label associations"):
                column = X[:, i].toarray().flatten()
                doc_indices = np.where(column > 0)[0]
                
                if len(doc_indices) > 0:
                    doc_labels = [labels[idx] for idx in doc_indices]
                    label_counts = pd.Series(doc_labels).value_counts()
                    total_docs = len(doc_indices)
                    
                    label_probs = {}
                    for label in unique_labels:
                        prob = label_counts.get(label, 0) / total_docs
                        label_probs[label] = prob
                    
                    ngram_label_associations[ngram] = label_probs
                else:
                    failure_count += 1
                    failed_ngrams.append(ngram)

            # Check failure rate
            failure_rate = failure_count / total_ngrams
            if failure_rate > FAILURE_THRESHOLD:
                warning_msg = (
                    f"High n-gram processing failure rate detected: {failure_rate:.2%}\n"
                    f"Failed n-grams: {', '.join(failed_ngrams[:10])}{'...' if len(failed_ngrams) > 10 else ''}\n"
                    f"Retry successes: {retry_success_count}"
                )
                logger.warning(warning_msg)
                raise NGramAnalysisError(warning_msg)
            
            # Group n-grams by label
            result = {}
            for label in unique_labels:
                label_ngrams = []
                for ngram, probs in ngram_label_associations.items():
                    if label in probs:
                        label_ngrams.append((ngram, probs[label]))
                
                sorted_ngrams = sorted(label_ngrams, key=lambda x: x[1], reverse=True)
                result[label] = sorted_ngrams[:top_n]
                
                logger.info(f"Found {len(sorted_ngrams)} n-grams for label {label}")
            
            logger.info(f"""
N-gram processing completed:
- Total n-grams: {total_ngrams}
- Failed n-grams: {failure_count} ({failure_rate:.2%})
- Final success rate: {(total_ngrams - failure_count) / total_ngrams:.2%}
            """.strip())
                
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise NGramAnalysisError(f"Analysis error: {str(e)}")

def train_and_evaluate(
    df: pd.DataFrame,
    text_column: str,
    rating_column: str,
    ngram_range: tuple[int, int] = (1, 3),
    min_df: float = 0.001,
    max_df: float = 0.7,
    max_features: int = 10000
) -> tuple[LogisticRegression, LabelEncoder, TfidfVectorizer]:
    """
    Trains a logistic regression model, evaluates it, and performs SHAP analysis.

    Args:
        df: Input DataFrame.
        text_column: Name of the column containing text data.
        rating_column: Name of the column containing ratings or labels.
        ngram_range: Tuple (min_n, max_n) for n-gram range.
        min_df: Minimum document frequency for TF-IDF.
        max_df: Maximum document frequency for TF-IDF.
        max_features: Maximum number of features for TF-IDF.

    Returns:
        A tuple containing the trained model, label encoder, and TF-IDF vectorizer.
    """
    try:
        logger.info(f"Starting model training with n-gram range: {ngram_range}")

        # Input validation
        if df.empty:
            raise ValueError("Empty DataFrame provided")

        if text_column not in df.columns or rating_column not in df.columns:
            raise ValueError(
                f"Required columns not found: {text_column}, {rating_column}"
            )

        if ngram_range[0] > ngram_range[1]:
            raise ValueError(f"Invalid n-gram range: {ngram_range}")

        # Create single vectorizer for the entire n-gram range
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words="english",
        )

        # Transform text to feature matrix
        logger.info("Extracting n-gram features...")
        X = vectorizer.fit_transform(df[text_column])
        feature_names = vectorizer.get_feature_names_out()

        # Log n-gram distribution
        ngram_lengths = [len(feat.split()) for feat in feature_names]
        for n in range(ngram_range[0], ngram_range[1] + 1):
            count = sum(1 for length in ngram_lengths if length == n)
            logger.info(f"Number of {n}-grams: {count}")

        # Prepare labels
        try:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df[rating_column])
            logger.info(f"Encoded {len(np.unique(y))} unique labels")
        except Exception as e:
            logger.error(f"Label encoding failed: {str(e)}")
            raise

        # Model training and evaluation
        try:
            model = LogisticRegression(solver="lbfgs", max_iter=1000, n_jobs=-1)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            roc_auc_scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="roc_auc_ovr_weighted",
                n_jobs=-1,  # Parallel cross-validation
            )

            logger.info(f"Cross-validation ROC-AUC scores: {roc_auc_scores}")
            logger.info(f"Mean ROC-AUC: {np.mean(roc_auc_scores):.4f}")

        except Exception as e:
            logger.error(f"Model training/evaluation failed: {str(e)}")
            raise

        # Feature importance analysis using mutual information
        try:
            mi_scores = mutual_info_classif(X, y, discrete_features=True, n_jobs=-1)
            mi_ranking = sorted(
                zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True
            )

            # Group important features by n-gram length
            ngram_importance = defaultdict(list)
            for feature, score in mi_ranking[:50]:  # Analyze top 50 features
                n = len(feature.split())
                ngram_importance[n].append((feature, score))

            # Log importance by n-gram length
            for n in range(ngram_range[0], ngram_range[1] + 1):
                if n in ngram_importance:
                    logger.info(f"\nTop {n}-grams by Mutual Information:")
                    for feature, score in ngram_importance[n][:5]:  # Show top 5
                        logger.info(f"  {feature}: {score:.4f}")

        except Exception as e:
            logger.error(f"Feature importance analysis failed: {str(e)}")
            logger.warning("Continuing despite feature importance error")

        # SHAP analysis
        try:
            model.fit(X, y)
            df_interactions = calculate_and_analyze_interactions(
                model, X, feature_names, y
            )

        except Exception as e:
            logger.error(f"SHAP analysis failed: {str(e)}")
            logger.warning("Continuing despite SHAP analysis error")

        # Final model fitting
        model.fit(X, y)
        logger.info("Model training and evaluation completed successfully")

        return model, label_encoder, vectorizer

    except Exception as e:
        logger.error(f"Training and evaluation failed: {str(e)}")
        raise