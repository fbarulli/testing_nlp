"""
This script performs TF-IDF optimization for text classification using Optuna. 
It compares different stop word lists (NLTK, spaCy, and custom) and preprocessing 
methods to find the best combination of parameters that maximizes ROC AUC, 
while also tracking accuracy and F1-score. The optimization is done in parallel 
using ProcessPoolExecutor to speed up the process. Results are saved to a CSV 
file after each trial, allowing for pausing and resumption of the optimization 
process.

Key Features:
- Parallel optimization of TF-IDF parameters.
- Comparison of NLTK, spaCy, and custom stop word lists.
- Option to use custom text preprocessing.
- Tracks ROC AUC, accuracy, and F1-score.
- Logs results to a CSV file, enabling pausing and resumption.
- Ngram range is now a global variable, accepting a single number input in the notebook.
"""

import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import nltk
from collections import defaultdict, Counter
from functools import lru_cache
import spacy
import os

# Download NLTK and spaCy stop words
nltk.download('stopwords', quiet=True)
nlp = spacy.load("en_core_web_sm")

# Stop word sets (can be customized in the notebook)
NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
SPACY_STOP_WORDS = nlp.Defaults.stop_words
CUSTOM_STOP_WORDS = set()  # Placeholder for custom stop words
CUSTOM_STOP_WORDS.update({'ll', 've'})  # Add potentially problematic tokens

# Global variable for ngram range (set in the notebook)
NGRAM_RANGE = (1, 1)  # Default value

# Constants for text cleaning
CONTRACTIONS = {
    # removing to fix: UserWarning: Your stop_words may be inconsistent with your preprocessing.
}

REMOVE_PATTERNS = [
    (r'`', ''),  # Remove backticks
    (r"'", ''),  # Remove apostrophes
    (r'[-]{2,}', ' '),  # Replace multiple hyphens with space
    (r'[^\w\s-]', ' '),  # Replace special chars with space (keep hyphen)
    (r'\bt\b', ''),  # Remove standalone t
    (r'\s+t\s+', ' '),  # Remove t surrounded by spaces
    (r'(?<=\s)t(?=\s)', ''),  # Remove t between words
    (r'(?<=^)t(?=\s)', ''),  # Remove t at start
    (r'(?<=\s)t(?=$)', ''),  # Remove t at end
    (r'\s+', ' ')  # Normalize spaces
]

# Define important terms and structural words
IMPORTANT_TERMS = {
    'service', 'quality', 'customer', 'product', 'shipping',
    'recommend', 'happy', 'great', 'good', 'bad', 'terrible',
    'excellent', 'awful', 'amazing', 'horrible', 'best', 'worst',
    'love', 'hate', 'helpful', 'useless', 'complaint', 'thank'
}
STRUCTURAL_STARTS = {'the', 'and', 'to', 'in', 'of', 'with', 'for', 'on'}
STRUCTURAL_WORDS = {'was', 'been', 'have', 'had', 'would', 'will', 'could'}

# Text cleaning functions
def clean_text(text: str) -> str:
    """Clean text with all patterns and rules"""
    if not text or not isinstance(text, str):
        return ''

    text = text.lower().strip()

    # Apply removal patterns
    for pattern, replacement in REMOVE_PATTERNS:
        text = re.sub(pattern, replacement, text)

    # Remove structural words except important terms
    words = text.split()
    cleaned_words = [
        word for word in words
        if word in IMPORTANT_TERMS or (word not in STRUCTURAL_STARTS and word not in STRUCTURAL_WORDS)
    ]
    text = ' '.join(cleaned_words).strip()

    # Final cleanup
    text = re.sub(r'\bt\b', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_single_text(text: str) -> tuple:
    """Process a single text entry with full cleaning"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    original_words = text.split()
    processed_text = clean_text(text)
    processed_words = processed_text.split()

    replacements = []
    if len(processed_words) != len(original_words):
        replacements.append('word_count_changed')

    return processed_text, replacements

def process_text_chunk(texts: pd.DataFrame) -> pd.DataFrame:
    """Process a chunk of texts, preserving the original index"""
    results = []
    for index, row in texts.iterrows():
        processed_text, replacements = process_single_text(row['text'])
        results.append({
            'index': index,
            'original_text': row['text'],
            'modified_text': processed_text,
            'replacements': replacements
        })
    return pd.DataFrame(results).set_index('index')

class TextProcessor:
    def __init__(self, num_processes: int = None, chunk_size: int = 10000):
        self.num_processes = num_processes or min(ProcessPoolExecutor()._max_workers, 8)
        self.chunk_size = chunk_size

    def process_dataframe(self, df: pd.DataFrame, text_column: str, batch_size: int = 1000) -> pd.DataFrame:
        """Process a DataFrame, cleaning the specified text column"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size
        results = []

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx].copy()

                num_chunks = min(self.num_processes, len(batch_df))
                chunks = np.array_split(batch_df, num_chunks)

                for chunk in chunks:
                    futures.append(executor.submit(process_text_chunk, chunk.copy()))

            for future in tqdm(futures, desc="Processing Batches", disable=not True):
                results.append(future.result())

        return pd.concat(results, axis=0)

    def process_text_for_comparison(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process the DataFrame for text comparison"""
        result_df = self.process_dataframe(df.copy(), text_column)

        if not result_df.empty:
            df = df.merge(result_df[['modified_text']],
                         left_index=True,
                         right_index=True,
                         how='left')
        else:
            df['modified_text'] = df[text_column]

        return df

class FeatureExtractor:
    def __init__(self, params: dict, stop_word_set: set = None):
        self.params = params
        self.stop_word_set = stop_word_set

        # Use stop words only if analyzer is 'word'
        if self.params['analyzer'] == 'word' and self.stop_word_set is not None:
            stop_words = list(self.stop_word_set)
        else:
            stop_words = None

        self.vectorizer = TfidfVectorizer(
            max_features=params['max_features'],
            min_df=params['min_df'],
            max_df=params['max_df'],
            ngram_range=NGRAM_RANGE,  # Use the global NGRAM_RANGE
            analyzer=params['analyzer'],
            stop_words=stop_words
        )

        # Initialize LSA component
        if params['svd_method'] == 'truncated':
            self.lsa = TruncatedSVD(n_components=params['n_components'], random_state=42)
        elif params['svd_method'] == 'incremental':
            self.lsa = IncrementalPCA(n_components=params['n_components'])
        else:
            self.lsa = None

        self.selector = SelectKBest(chi2, k=params['num_features_to_select'])

    def _update_stop_words_based_on_semantic_threshold(self, components, feature_names):
        """Calculate semantic scores and update stop words based on threshold"""
        semantic_scores = np.abs(components).mean(axis=0)
        threshold = np.percentile(semantic_scores, self.params['semantic_threshold'])
        stop_indices = np.where(semantic_scores < threshold)[0]
        new_stop_words = feature_names[stop_indices].tolist()

        if self.params['analyzer'] == 'word':
            current_stop_words = set(self.vectorizer.stop_words or [])
            self.vectorizer.stop_words = list(current_stop_words.union(new_stop_words))

        return new_stop_words

    def fit_transform(self, texts, labels):
        """Fit the vectorizer and LSA model, then transform the texts"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        # Initial vectorization
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        X_selected = None  # Initialize X_selected

        if self.lsa:
            # Adjust num_features_to_select if necessary (before feature selection)
            if self.selector.k > tfidf_matrix.shape[1]:
                self.selector.k = tfidf_matrix.shape[1]

            # Feature selection
            X_selected = self.selector.fit_transform(tfidf_matrix, labels)

            # Adjust components if necessary
            n_features = X_selected.shape[1]
            if isinstance(self.lsa, TruncatedSVD) and self.params['n_components'] > n_features:
                self.lsa.n_components = n_features
            elif isinstance(self.lsa, IncrementalPCA) and self.params['n_components'] > n_features:
                self.lsa.n_components = n_features

            # LSA transformation
            X = self.lsa.fit_transform(X_selected)

            # Handle semantic threshold
            if self.params['semantic_threshold'] > 0:
                feature_names = np.array(self.vectorizer.get_feature_names_out())
                new_stop_words = self._update_stop_words_based_on_semantic_threshold(self.lsa.components_.copy(), feature_names)

                # Refit with updated stop words if using word analyzer and lsa
                if self.params['analyzer'] == 'word' and new_stop_words:
                    tfidf_matrix = self.vectorizer.fit_transform(texts)

                    # Adjust num_features_to_select if necessary (before feature selection)
                    if self.selector.k > tfidf_matrix.shape[1]:
                        self.selector.k = tfidf_matrix.shape[1]

                    X_selected = self.selector.fit_transform(tfidf_matrix, labels)

                    # Re-adjust components if necessary
                    n_features = X_selected.shape[1]
                    if isinstance(self.lsa, TruncatedSVD) and self.params['n_components'] > n_features:
                        self.lsa.n_components = n_features
                    elif isinstance(self.lsa, IncrementalPCA) and self.params['n_components'] > n_features:
                        self.lsa.n_components = n_features

                    X = self.lsa.fit_transform(X_selected)

        else:  # self.lsa is False
            # Adjust num_features_to_select if necessary (before feature selection)
            if self.selector.k > tfidf_matrix.shape[1]:
                self.selector.k = tfidf_matrix.shape[1]

            X_selected = self.selector.fit_transform(tfidf_matrix, labels)
            X = X_selected  # Always assign X_selected to X when not using LSA

        # Ensure X is always assigned
        if X_selected is not None and X is None:
            X = X_selected

        if X is None:
            raise ValueError("Neither LSA nor feature selection was performed, or an error occurred during the process. 'X' remained unassigned.")

        return X

    def transform(self, texts):
        """Transform new texts using the fitted vectorizer and LSA model"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        tfidf_matrix = self.vectorizer.transform(texts)
        if self.lsa:
            X_selected = self.selector.transform(tfidf_matrix)
            return self.lsa.transform(X_selected)
        return self.selector.transform(tfidf_matrix)

def optimize_tfidf(df: pd.DataFrame, text_col: str, rating_col: str,
                   stop_word_type: str, use_custom_preprocessing: bool,
                   n_trials: int, results_df: pd.DataFrame,
                   completed_trials: set, results_file: str,
                   study_name: str) -> pd.DataFrame:
    """Optimizes TF-IDF parameters using Optuna, tracks metrics, and logs results to CSV."""

    def objective(trial: optuna.Trial):
        # Define parameters for optimization
        params = {
            'max_features': trial.suggest_int('max_features', 100, 2000),
            'min_df': trial.suggest_float('min_df', 0.0001, 0.01),
            'max_df': trial.suggest_float('max_df', 0.95, 0.999),
            'analyzer': trial.suggest_categorical('analyzer', ['word', 'char']),
            'svd_method': trial.suggest_categorical('svd_method', ['truncated', 'incremental', 'none']),
            'semantic_threshold': trial.suggest_float('semantic_threshold', 0, 20),
            'num_features_to_select': trial.suggest_int('num_features_to_select', 10, 2000),
            'n_components': trial.suggest_int('n_components', 1, 100)
        }

        # Create a unique key for the current trial's parameters
        trial_params_key = (stop_word_type, use_custom_preprocessing, frozenset(params.items()))

        # Skip trial if already completed
        if trial_params_key in completed_trials:
            print(f"Skipping trial with identical parameters: {trial_params_key}")
            raise optuna.exceptions.TrialPruned()

        # Get stop words based on type
        if stop_word_type == 'nltk':
            stop_words = NLTK_STOP_WORDS
        elif stop_word_type == 'spacy':
            stop_words = SPACY_STOP_WORDS
        elif stop_word_type == 'custom':
            stop_words = CUSTOM_STOP_WORDS
        else:
            stop_words = None

        # Process text data
        if use_custom_preprocessing:
            text_processor = TextProcessor()
            processed_df = text_processor.process_text_for_comparison(
                df[[text_col]].copy(),
                text_col
            )
            processed_texts = processed_df['modified_text'].fillna('').tolist()
        else:
            processed_texts = df[text_col].fillna('').tolist()

        # Initialize feature extractor
        feature_extractor = FeatureExtractor(params, stop_words)

        # Handle potential errors during feature extraction
        try:
            X = feature_extractor.fit_transform(processed_texts, df[rating_col].values)
        except ValueError as e:
            print(f"Error during feature extraction: {e}")
            print(f"Parameters causing the error: {params}")
            raise optuna.exceptions.TrialPruned()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, df[rating_col].values, test_size=0.2, random_state=42, stratify=df[rating_col].values
        )

        # Train model
        model = LogisticRegression(max_iter=10000, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Update results DataFrame
        nonlocal results_df
        new_row = pd.DataFrame([{
            'trial_number': trial.number,
            'stop_word_type': stop_word_type,
            'use_custom_preprocessing': use_custom_preprocessing,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'f1': f1,
            **params,
            'ngram_range': f"1-{NGRAM_RANGE[1]}" if NGRAM_RANGE[0] == 1 else f"{NGRAM_RANGE[0]}-{NGRAM_RANGE[1]}"
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update CSV
        results_df.to_csv(results_file, index=False)

        # Add current trial parameters to completed trials set
        completed_trials.add(trial_params_key)

        return roc_auc

    # Create or load Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    return results_df

def run_optimizations(df: pd.DataFrame, text_col: str, rating_col: str,
                      n_trials: int = 100, num_processes: int = 4,
                      results_file: str = 'optimization_results.csv'):
    """
    Runs TF-IDF optimizations in parallel for different stop word types and preprocessing options.

    Args:
        df (pd.DataFrame): DataFrame containing text and rating columns.
        text_col (str): Name of the text column.
        rating_col (str): Name of the rating column.
        n_trials (int): Number of trials for each optimization.
        num_processes (int): Number of processes for parallel execution.
        results_file (str): Path to the CSV file for storing results.
    """

    # Load existing results or create a new DataFrame
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        completed_trials = set()
        for _, row in results_df.iterrows():
            # More robust parsing of ngram_range
            ngram_range_str = row['ngram_range']
            if '-' in ngram_range_str:
                ngram_start, ngram_end = map(int, ngram_range_str.split('-'))
                ngram_range_tuple = (ngram_start, ngram_end)
            elif ',' in ngram_range_str:  # Handle old format (1, 1)
                ngram_range_str = ngram_range_str.strip('()')  # Remove parentheses
                ngram_start, ngram_end = map(int, ngram_range_str.split(','))
                ngram_range_tuple = (ngram_start, ngram_end)
            else:  # Should ideally not reach here if data is clean
                ngram_range_tuple = (1, int(ngram_range_str))

            params = {
                'max_features': row['max_features'],
                'min_df': row['min_df'],
                'max_df': row['max_df'],
                'ngram_range': ngram_range_tuple,
                'analyzer': row['analyzer'],
                'svd_method': row['svd_method'],
                'semantic_threshold': row['semantic_threshold'],
                'num_features_to_select': row['num_features_to_select'],
                'n_components': row['n_components']
            }
            completed_trials.add((row['stop_word_type'], row['use_custom_preprocessing'], frozenset(params.items())))
    else:
        results_df = pd.DataFrame()
        completed_trials = set()

    # Run optimizations in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for stop_word_type in ['nltk', 'spacy', 'custom']:
            for use_custom_preprocessing in [True, False]:
                study_name = f"{stop_word_type}_{'custom' if use_custom_preprocessing else 'baseline'}"
                futures.append(
                    executor.submit(
                        optimize_tfidf,
                        df,
                        text_col,
                        rating_col,
                        stop_word_type,
                        use_custom_preprocessing,
                        n_trials,
                        results_df,
                        completed_trials,
                        results_file,
                        study_name
                    )
                )

        # Wait for all futures to complete
        for future in tqdm(futures, desc="Optimizing TF-IDF"):
            results_df = future.result()

    print("Optimization complete. Results saved to:", results_file)