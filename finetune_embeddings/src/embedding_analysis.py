from __future__ import annotations
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import traceback
from scipy.stats import spearmanr
import umap
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm.auto import tqdm
import numpy.typing as npt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingAnalyzer:
    """
    Comprehensive embedding analysis tool to evaluate embedding quality
    through various metrics and visualizations.
    """
    def __init__(self) -> None:
        logger.info("Initializing EmbeddingAnalyzer")
        try:
            self.metrics_history: Dict[str, List[float]] = defaultdict(list)
            logger.info("EmbeddingAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing EmbeddingAnalyzer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def analyze_embedding_space(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze the quality of the embedding space through various metrics.
        
        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)
            labels: Optional tensor of shape (n_samples,) for supervised metrics
            
        Returns:
            Dictionary containing various embedding quality metrics
        """
        logger.info("Starting embedding space analysis")
        try:
            metrics: Dict[str, float] = {}
            
            # Convert to numpy for calculations
            emb_np: npt.NDArray[np.float32] = embeddings.detach().cpu().numpy()
            
            # Progress bar for analysis steps
            analysis_steps = tqdm(total=4, desc="Analyzing embeddings", leave=False)
            
            # 1. Isotropy - How uniform is the distribution in the embedding space
            singular_values = np.linalg.svd(emb_np, compute_uv=False)
            metrics['isotropy'] = float(1 - (np.std(singular_values) / np.mean(singular_values)))
            analysis_steps.update(1)
            
            # 2. Average cosine similarity - Measure of embedding separation
            pairwise_sim = tqdm(
                enumerate(emb_np),
                total=len(emb_np),
                desc="Computing cosine similarities",
                leave=False
            )
            cosine_sims = []
            for i, emb in pairwise_sim:
                sims = cosine_similarity(emb.reshape(1, -1), emb_np[i+1:])
                cosine_sims.extend(sims.flatten())
            
            metrics['avg_cosine_sim'] = float(np.mean(cosine_sims))
            metrics['cosine_sim_std'] = float(np.std(cosine_sims))
            analysis_steps.update(1)
            
            # 3. Embedding norm statistics
            norms = np.linalg.norm(emb_np, axis=1)
            metrics['norm_mean'] = float(np.mean(norms))
            metrics['norm_std'] = float(np.std(norms))
            
            # 4. Effective dimensionality
            explained_var_ratio = singular_values / singular_values.sum()
            metrics['effective_dim'] = float(np.sum(explained_var_ratio > 0.01))
            analysis_steps.update(1)
            
            if labels is not None:
                labels_np: npt.NDArray[np.int64] = labels.detach().cpu().numpy()
                unique_labels = np.unique(labels_np)
                
                # 5. Intra-class cosine similarity
                intra_class_sim: List[float] = []
                for label in tqdm(unique_labels, desc="Computing intra-class similarities", leave=False):
                    mask = labels_np == label
                    if np.sum(mask) > 1:
                        class_embeddings = emb_np[mask]
                        class_sim = cosine_similarity(class_embeddings)
                        np.fill_diagonal(class_sim, 0)
                        intra_class_sim.append(float(np.mean(class_sim)))
                
                metrics['intra_class_sim'] = float(np.mean(intra_class_sim))
                
                # 6. Inter-class separation
                kmeans = KMeans(n_clusters=len(unique_labels))
                cluster_labels = kmeans.fit_predict(emb_np)
                metrics['cluster_label_alignment'] = float(np.mean(cluster_labels == labels_np))
            
            analysis_steps.update(1)
            analysis_steps.close()
            
            # Store metrics history
            for key, value in metrics.items():
                self.metrics_history[key].append(value)
            
            logger.info("Embedding space analysis completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in embedding space analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def visualize_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        method: str = 'tsne',
        save_path: Optional[str] = None,
        perplexity: int = 30,
        n_neighbors: int = 15
    ) -> None:
        """
        Visualize embeddings using dimensionality reduction techniques.
        
        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)
            labels: Optional tensor of shape (n_samples,) for coloring points
            method: Dimensionality reduction method ('tsne' or 'umap')
            save_path: Optional path to save the visualization
            perplexity: t-SNE perplexity parameter
            n_neighbors: UMAP n_neighbors parameter
        """
        logger.info(f"Starting embedding visualization using {method}")
        try:
            emb_np: npt.NDArray[np.float32] = embeddings.detach().cpu().numpy()
            
            # Progress bar for dimensionality reduction
            with tqdm(total=1, desc=f"Reducing dimensionality with {method.upper()}", leave=False) as pbar:
                if method.lower() == 'tsne':
                    reducer = TSNE(
                        n_components=2,
                        perplexity=perplexity,
                        n_iter=1000,
                        verbose=0
                    )
                else:  # UMAP
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        verbose=0
                    )
                
                reduced_embeddings = reducer.fit_transform(emb_np)
                pbar.update(1)
            
            plt.figure(figsize=(10, 8))
            if labels is not None:
                labels_np = labels.detach().cpu().numpy()
                scatter = plt.scatter(
                    reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1],
                    c=labels_np,
                    cmap='tab10',
                    alpha=0.6
                )
                plt.colorbar(scatter)
            else:
                plt.scatter(
                    reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1],
                    alpha=0.6
                )
            
            plt.title(f'Embedding Visualization ({method.upper()})')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Visualization saved to {save_path}")
            plt.close()
            
            logger.info("Embedding visualization completed successfully")
            
        except Exception as e:
            logger.error(f"Error in embedding visualization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def plot_metrics_history(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot the history of embedding quality metrics.
        
        Args:
            save_path: Optional path to save the visualization
        """
        logger.info("Starting metrics history visualization")
        try:
            n_metrics = len(self.metrics_history)
            fig, axes = plt.subplots(
                (n_metrics + 1) // 2, 2,
                figsize=(15, 5 * ((n_metrics + 1) // 2))
            )
            axes = axes.flatten()
            
            for (metric_name, metric_values), ax in tqdm(
                zip(self.metrics_history.items(), axes),
                total=n_metrics,
                desc="Plotting metrics",
                leave=False
            ):
                ax.plot(metric_values, marker='o')
                ax.set_title(f'{metric_name} Over Time')
                ax.set_xlabel('Step')
                ax.set_ylabel(metric_name)
                ax.grid(True)
            
            # Remove empty subplots if odd number of metrics
            if n_metrics % 2 != 0:
                fig.delaxes(axes[-1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Metrics history plot saved to {save_path}")
            plt.close()
            
            logger.info("Metrics history visualization completed successfully")
            
        except Exception as e:
            logger.error(f"Error in metrics history visualization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def compute_similarity_matrix(
        self,
        embeddings: torch.Tensor,
        batch_size: int = 128
    ) -> torch.Tensor:
        """
        Compute pairwise similarity matrix efficiently in batches.
        
        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)
            batch_size: Size of batches for memory efficiency
            
        Returns:
            Tensor containing pairwise similarities
        """
        logger.info("Computing similarity matrix")
        try:
            n_samples = embeddings.size(0)
            similarity_matrix = torch.zeros((n_samples, n_samples))
            
            n_batches = (n_samples + batch_size - 1) // batch_size
            batch_pbar = tqdm(
                total=n_batches * n_batches,
                desc="Computing similarity matrix",
                leave=False
            )
            
            for i in range(0, n_samples, batch_size):
                batch_i = embeddings[i:i + batch_size]
                for j in range(0, n_samples, batch_size):
                    batch_j = embeddings[j:j + batch_size]
                    similarity_matrix[i:i + batch_size, j:j + batch_size] = torch.mm(
                        batch_i,
                        batch_j.t()
                    )
                    batch_pbar.update(1)
            
            batch_pbar.close()
            logger.info("Similarity matrix computation completed successfully")
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error in similarity matrix computation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
