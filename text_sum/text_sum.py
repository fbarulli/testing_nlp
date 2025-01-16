from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summa import summarizer as textrank
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from tqdm import tqdm
import functools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    batch_size: int = 32
    max_length: int = 128
    min_length: int = 10
    num_workers: int = 4
    cache_dir: str = "./model_cache"

class TextProcessor:
    def __init__(self, config: ProcessingConfig = ProcessingConfig()):
        self.config = config
        logger.info("Initializing TextProcessor...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            cache_dir=config.cache_dir
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
    "sshleifer/distilbart-cnn-12-6",
    cache_dir=config.cache_dir
)


        # Check for fast tokenizer
        if self.tokenizer.is_fast:
            logger.info("Using fast tokenizer.")
        else:
            logger.warning("Fast tokenizer not available. Consider upgrading transformers library.")

        self.extractive_cache = {}  # Cache for extractive summaries
        self.abstractive_cache = {} # Cache for abstractive summaries

        logger.info("TextProcessor initialized successfully")

    @functools.lru_cache(maxsize=None)
    def _extractive_summarize(self, text: str) -> str:
        try:
            summary = textrank.summarize(
                text,
                ratio=0.3,
                words=self.config.max_length
            )
            return summary if summary else text[:self.config.max_length]
        except Exception as e:
            logger.warning(f"Extractive summarization failed: {e}")
            return text[:self.config.max_length]

    @functools.lru_cache(maxsize=None)
    def _abstractive_summarize_batch(self, texts: Tuple[str]) -> List[str]:
        texts_list = list(texts)
        try:
            inputs = self.tokenizer(
                texts_list,
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            summaries = self.model.generate(
                inputs["input_ids"],
                max_length=self.config.max_length,
                min_length=self.config.min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

            return self.tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True
            )
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            return texts_list

    def process_texts(self, df: pd.DataFrame, text_column: str, rating_column: Optional[str] = None) -> pd.DataFrame:
        """Optimized single-pass processing pipeline"""
        logger.info(f"Processing {len(df)} texts...")
        result_df = df.copy()

        extractive_summaries = []
        abstractive_summaries = []
        
        total_batches = (len(df) + self.config.batch_size - 1) // self.config.batch_size

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor, tqdm(total=total_batches, desc="Processing batches") as pbar:
            for i in range(0, len(df), self.config.batch_size):
                batch = df[text_column].iloc[i:i + self.config.batch_size].tolist()

                # Extractive Summarization with Caching
                ext_summaries = []
                for text in batch:
                    if text not in self.extractive_cache:
                        self.extractive_cache[text] = self._extractive_summarize(text)
                    ext_summaries.append(self.extractive_cache[text])

                # Abstractive Summarization with Caching
                batch_tuple = tuple(batch)
                if batch_tuple not in self.abstractive_cache:
                    self.abstractive_cache[batch_tuple] = self._abstractive_summarize_batch(batch_tuple)
                abs_summaries = self.abstractive_cache[batch_tuple]

                extractive_summaries.extend(ext_summaries)
                abstractive_summaries.extend(abs_summaries)
                
                pbar.update(1)

        result_df['extractive_summary'] = extractive_summaries
        result_df['abstractive_summary'] = abstractive_summaries

        if rating_column:
            # Keep both raw rating and deviation
            result_df['rating'] = df[rating_column].astype(int)
            result_df['rating_deviation'] = df[rating_column] - df[rating_column].mean()

            # Add rating distribution info
            rating_counts = df[rating_column].value_counts()
            result_df['rating_frequency'] = result_df[rating_column].map(rating_counts)

        logger.info("Processing completed successfully")
        return result_df