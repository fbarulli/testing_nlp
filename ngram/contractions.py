import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Set, Tuple, Optional
import nltk
from collections import defaultdict, Counter
from functools import lru_cache, partial
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import CountVectorizer
import os
import shap
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Set up NLTK data path
CUSTOM_PATH = "/home/aboveclouds49/nltk_data"
os.makedirs(CUSTOM_PATH, exist_ok=True)

# Download required NLTK packages
packages = [
    'punkt',
    'averaged_perceptron_tagger',
    'wordnet',
    'punkt_tab',
    'averaged_perceptron_tagger_eng'
]

for package in packages:
    nltk.download(package, download_dir=CUSTOM_PATH)

nltk.data.path.append(CUSTOM_PATH)

# Constants for text processing
CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are", "i've": "i have",
    "you've": "you have", "we've": "we have", "they've": "they have", "i'll": "i will",
    "you'll": "you will", "he'll": "he will", "she'll": "she will", "it'll": "it will",
    "we'll": "we will", "they'll": "they will", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "it'd": "it would", "we'd": "we would",
    "they'd": "they would", "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
    "mightn't": "might not", "mustn't": "must not"
}

REMOVE_PATTERNS = [
    (r'`', ''),
    (r"'", ''),
    (r'-t-', ' '),
    (r'[^\w\s-]', ' '),
    (r'\bt\b', ''),
    (r'\s+', ' ')
]

# Default configuration parameters
DEFAULT_MIN_DF = 0.001
DEFAULT_STOP_WORDS = [
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might',
    'must', 'ought', 'i', 'me', 'my', 'myself', 'we', 'us', 'our',
    'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'whose', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'upon', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't"
]
DEFAULT_IMPORTANT_TERMS = {
    'service', 'quality', 'customer', 'product', 'shipping',
    'recommend', 'happy', 'great', 'good', 'bad', 'terrible',
    'excellent', 'awful', 'amazing', 'horrible', 'best', 'worst',
    'love', 'hate', 'helpful', 'useless', 'complaint', 'thank'
}
DEFAULT_STRUCTURAL_STARTS = {
    'the', 'and', 'to', 'in', 'of', 'with', 'for', 'on'
}
DEFAULT_STRUCTURAL_WORDS = {
    'was', 'been', 'have', 'had', 'would', 'will', 'could'
}
DEFAULT_NGRAM_RANGE = (2, 5)
DEFAULT_TOP_N = 20

@dataclass
class TextProcessorConfig:
    """Configuration for text processing."""
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE
    top_n: int = DEFAULT_TOP_N
    min_df: float = DEFAULT_MIN_DF
    stop_words: List[str] = field(default_factory=lambda: DEFAULT_STOP_WORDS)
    important_terms: Set[str] = field(default_factory=lambda: DEFAULT_IMPORTANT_TERMS)
    structural_starts: Set[str] = field(default_factory=lambda: DEFAULT_STRUCTURAL_STARTS)
    structural_words: Set[str] = field(default_factory=lambda: DEFAULT_STRUCTURAL_WORDS)

    def __post_init__(self):
        # Ensure important terms are not in stop words
        self.stop_words = [
            word for word in self.stop_words if word not in self.important_terms
        ]

def clean_text(text: str, config: TextProcessorConfig) -> str:
    """Aggressively clean text by removing special characters, normalizing, and applying stop word removal"""
    if not text or not isinstance(text, str):
        return ''

    text = text.lower().strip()

    # Handle contractions
    words = text.split()
    cleaned_words = [CONTRACTIONS.get(word, word) for word in words]
    text = ' '.join(cleaned_words).strip()

    # Apply all removal patterns
    for pattern, replacement in REMOVE_PATTERNS:
        text = re.sub(pattern, replacement, text)

    # Remove stop words, structural starts, and structural words, except for important terms
    words = text.split()
    cleaned_words = [
        word for word in words
        if word in config.important_terms or (
            word not in config.stop_words and
            word not in config.structural_starts and
            word not in config.structural_words
        )
    ]
    text = ' '.join(cleaned_words).strip()

    # Final step: Remove standalone 't' and backticks
    text = re.sub(r'\bt\b', '', text)
    text = re.sub(r'`', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_single_text(text: str, config: TextProcessorConfig) -> Tuple[str, List[str]]:
    """Process a single text entry with aggressive cleaning"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    original_words = text.split()
    processed_text = clean_text(text, config)
    processed_words = processed_text.split()

    replacements = []
    if len(processed_words) != len(original_words):
        replacements.append('word_count_changed')

    return processed_text, replacements

def process_text_chunk(texts: np.ndarray, config: TextProcessorConfig) -> pd.DataFrame:
    """Process a chunk of texts"""
    results = []
    for text in texts:
        processed_text, replacements = process_single_text(text, config)
        results.append({
            'original_text': text,
            'modified_text': processed_text,
            'replacements': replacements
        })
    return pd.DataFrame(results)

def _process_chunk_helper(chunk: np.ndarray, config: TextProcessorConfig) -> pd.DataFrame:
    """Helper function to process a text chunk with a given config (for pickling)."""
    return process_text_chunk(chunk, config)

def calculate_similarity_ratio(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts"""
    if not text1 or not text2:
        return 0.0

    # Clean both texts - Assuming you want to use a default config here
    default_config = TextProcessorConfig()
    text1 = clean_text(text1, default_config)
    text2 = clean_text(text2, default_config)

    if not text1 or not text2:
        return 0.0

    # Calculate word-based similarity
    words1 = set(text1.split())
    words2 = set(text2.split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    if union == 0:
        return 0.0

    return intersection / union

def add_similarity_scores(comparison_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Add similarity scores to the comparison DataFrame"""
    df = comparison_df.copy()
    df['original_vs_modified_similarity'] = df.apply(
        lambda x: calculate_similarity_ratio(x.get(text_column), x.get('modified_text')),
        axis=1
    )
    return df

def generate_analysis_report(df: pd.DataFrame, text_column: str, comparison_df: pd.DataFrame) -> None:
    """Generate analysis report with similarity scores and statistics"""
    comparison_df = add_similarity_scores(comparison_df, text_column)

    def safe_word_count(text):
        return len(str(text).split()) if pd.isna(text) or isinstance(text, str) else 0

    comparison_df['word_count_original'] = comparison_df[text_column].apply(safe_word_count)
    comparison_df['word_count_modified'] = comparison_df['modified_text'].apply(safe_word_count)
    comparison_df['words_removed'] = comparison_df['word_count_original'] - comparison_df['word_count_modified']

    print("\nSimilarity Analysis:")
    print(comparison_df[[text_column, 'original_vs_modified_similarity']])

    print("\nSummary of Modifications:")
    print(f"Average words removed: {comparison_df['words_removed'].mean():.2f}")
    print(f"Average similarity to original: {comparison_df['original_vs_modified_similarity'].mean():.2f}")

@lru_cache(maxsize=10000)
def get_nouns(text: str) -> Tuple[str, ...]:
    """Extract nouns from text with caching"""
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    return tuple(word for word, pos in tagged if pos.startswith('NN'))

def analyze_nouns_by_rating(df: pd.DataFrame, text_column: str, rating_column: str, min_frequency: int = 5) -> Dict[int, List[Tuple[str, int]]]:
    """Analyze nouns grouped by rating"""
    grouped = df.groupby(rating_column)[text_column]
    noun_counts = defaultdict(Counter)

    for rating, texts in grouped:
        nouns = [get_nouns(str(text)) for text in texts]
        noun_counts[rating].update([noun for noun_list in nouns for noun in noun_list])

    return {
        rating: sorted(
            [(noun, count) for noun, count in counts.items() if count >= min_frequency],
            key=lambda x: x[1],
            reverse=True
        )
        for rating, counts in noun_counts.items()
    }

def print_noun_analysis(noun_analysis: Dict[int, List[Tuple[str, int]]], top_n: int = 10) -> None:
    """Print noun analysis results"""
    for rating in sorted(noun_analysis.keys()):
        print(f"\nRating {rating} - Top {top_n} nouns:")
        print("\n".join(f"{noun}: {count}" for noun, count in noun_analysis[rating][:top_n]))

def extract_top_ngrams(texts: List[str], ngram_range: Tuple[int, int], top_n: int, min_df: float = 0.001) -> List[Tuple[str, float]]:
    """Extract top n-grams from a list of texts."""
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    X = vectorizer.fit_transform(texts)
    ngram_counts = X.sum(axis=0)
    ngram_freq = [(ngram, ngram_counts[0, idx]) for ngram, idx in vectorizer.vocabulary_.items()]
    ngram_freq.sort(key=lambda x: x[1], reverse=True)
    return ngram_freq[:top_n]

def print_top_ngrams_by_label(df: pd.DataFrame, text_column: str, label_column: str, ngram_range: Tuple[int, int], top_n: int, min_df: float = 0.001):
    """Print top n-grams for each label."""
    for label in df[label_column].unique():
        print(f"Top n-grams for label '{label}':")
        texts = df[df[label_column] == label][text_column].tolist()
        top_ngrams = extract_top_ngrams(texts, ngram_range, top_n, min_df)
        for ngram, freq in top_ngrams:
            print(f"  {ngram}: {freq:.4f}")

class TextProcessor:
    def __init__(self, config: Optional[TextProcessorConfig] = None, num_processes: int = None, chunk_size: int = 10000):
        """Initialize TextProcessor with multiprocessing settings and configuration"""
        self.num_processes = num_processes or min(ProcessPoolExecutor()._max_workers, 8)
        self.chunk_size = chunk_size
        self.config = config or TextProcessorConfig()

    def process_dataframe(self, df: pd.DataFrame, text_column: str, batch_size: int = 1000) -> pd.DataFrame:
        """Process DataFrame in parallel using batched operations"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size
        results = []

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_rows)
                batch_texts = df[text_column].iloc[start_idx:end_idx].values

                num_chunks = min(self.num_processes, len(batch_texts))
                if num_chunks > 0:
                    chunks = np.array_split(batch_texts, num_chunks)
                    
                    # Use the helper function directly with executor.map
                    batch_results = list(executor.map(
                        _process_chunk_helper,
                        chunks,
                        [self.config] * len(chunks)  # Pass config for each chunk
                    ))
                    results.extend(batch_results)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def process_text_for_comparison(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process the DataFrame for text comparison"""
        result_df = self.process_dataframe(df.copy(), text_column)

        if not result_df.empty:
            df = pd.merge(
                df,
                result_df[['original_text', 'modified_text']],
                left_on=text_column,
                right_on='original_text',
                how='left'
            )
        else:
            df['modified_text'] = df[text_column]

        return df

    def analyze_nouns(self, df: pd.DataFrame, text_column: str, rating_column: str, min_frequency: int = 5, top_n: int = 10) -> None:
        """Analyze nouns in text data with ratings"""
        results = analyze_nouns_by_rating(df, text_column, rating_column, min_frequency)
        print_noun_analysis(results, top_n)

    def analyze_ngrams(self, df: pd.DataFrame, text_column: str, label_column: str, ngram_range: Tuple[int, int], top_n: int, min_df: float = 0.001):
        """Analyze and print top n-grams for each label."""
        print_top_ngrams_by_label(df, text_column, label_column, ngram_range, top_n, min_df)




































from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from scipy.sparse import hstack, csr_matrix














class NounSentimentAnalyzer:
    """Class for analyzing sentiment patterns associated with nouns"""
    def __init__(self, min_noun_freq: int = 3):
        self.min_noun_freq = min_noun_freq
        self.noun_sentiment_scores = {}
        self.noun_context_patterns = defaultdict(lambda: defaultdict(int))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        
    def extract_noun_contexts(self, text: str, window_size: int = 3) -> List[Tuple[str, List[str]]]:
        """Extract context windows around nouns"""
        tokens = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        
        noun_contexts = []
        for i, (word, pos) in enumerate(tagged):
            if pos.startswith('NN'):
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                context = tokens[start:i] + tokens[i+1:end]
                noun_contexts.append((word, context))
        
        return noun_contexts

    def build_noun_sentiment_patterns(self, texts: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
        """Build patterns of sentiment associated with nouns"""
        noun_label_counts = defaultdict(lambda: defaultdict(int))
        noun_total_counts = defaultdict(int)
        
        for text, label in zip(texts, labels):
            nouns = get_nouns(text)
            for noun in nouns:
                noun_label_counts[noun][label] += 1
                noun_total_counts[noun] += 1
        
        # Calculate sentiment scores for frequent nouns
        sentiment_patterns = {}
        for noun, counts in noun_label_counts.items():
            if noun_total_counts[noun] >= self.min_noun_freq:
                sentiment_patterns[noun] = {
                    label: count / noun_total_counts[noun]
                    for label, count in counts.items()
                }
        
        return sentiment_patterns

    def create_noun_features(self, texts: List[str]) -> csr_matrix:
        """Create noun-specific features for classification"""
        noun_features = []
        
        for text in texts:
            # Get noun frequency features
            nouns = get_nouns(text)
            noun_freq = Counter(nouns)
            
            # Get noun context features
            noun_contexts = self.extract_noun_contexts(text)
            context_features = defaultdict(float)
            
            for noun, context in noun_contexts:
                if noun in self.noun_sentiment_scores:
                    # Add sentiment score features
                    for label, score in self.noun_sentiment_scores[noun].items():
                        context_features[f'noun_{noun}_sentiment_{label}'] = score * noun_freq[noun]
                        
                    # Add context pattern features
                    context_str = ' '.join(context)
                    for pattern, count in self.noun_context_patterns[noun].items():
                        if pattern in context_str:
                            context_features[f'noun_{noun}_pattern_{pattern}'] += count
            
            noun_features.append(dict(context_features))
        
        # Convert to sparse matrix
        feature_names = sorted(set().union(*[d.keys() for d in noun_features]))
        feature_matrix = np.zeros((len(texts), len(feature_names)))
        
        for i, features in enumerate(noun_features):
            for j, name in enumerate(feature_names):
                feature_matrix[i, j] = features.get(name, 0.0)
                
        return csr_matrix(feature_matrix)

    def fit(self, texts: List[str], labels: List[str]) -> 'NounSentimentAnalyzer':
        """Fit the analyzer on training data"""
        # Encode labels
        self.label_encoder.fit(labels)
        
        # Build noun sentiment patterns
        self.noun_sentiment_scores = self.build_noun_sentiment_patterns(texts, labels)
        
        # Build noun context patterns
        for text, label in zip(texts, labels):
            noun_contexts = self.extract_noun_contexts(text)
            for noun, context in noun_contexts:
                if noun in self.noun_sentiment_scores:
                    context_str = ' '.join(context)
                    self.noun_context_patterns[noun][context_str] += 1
        
        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """Transform texts into noun-based features"""
        return self.create_noun_features(texts)

# Add these methods to the TextProcessor class
def add_noun_sentiment_features(self, df: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    """Add noun sentiment features to the DataFrame"""
    analyzer = NounSentimentAnalyzer()
    
    # Fit and transform
    noun_features = analyzer.fit_transform(
        df[text_column].tolist(),
        df[label_column].tolist()
    )
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(
        noun_features.toarray(),
        columns=[f'noun_feature_{i}' for i in range(noun_features.shape[1])]
    )
    
    return pd.concat([df, feature_df], axis=1)

def analyze_noun_sentiment_patterns(self, df: pd.DataFrame, text_column: str, label_column: str) -> Dict[str, Dict[str, Any]]:
    """Analyze sentiment patterns associated with nouns"""
    analyzer = NounSentimentAnalyzer()
    patterns = analyzer.build_noun_sentiment_patterns(
        df[text_column].tolist(),
        df[label_column].tolist()
    )
    
    # Add additional analysis
    noun_analysis = {}
    for noun, sentiment_scores in patterns.items():
        # Get most common contexts
        contexts = [
            context for text in df[text_column]
            for n, context in analyzer.extract_noun_contexts(text)
            if n == noun
        ]
        
        noun_analysis[noun] = {
            'sentiment_distribution': sentiment_scores,
            'total_occurrences': sum(sentiment_scores.values()),
            'most_common_contexts': Counter(map(lambda x: ' '.join(x), contexts)).most_common(5),
            'sentiment_strength': max(sentiment_scores.values()) - min(sentiment_scores.values())
        }