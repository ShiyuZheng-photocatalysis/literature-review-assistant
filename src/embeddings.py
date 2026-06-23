"""Embedding generation with sentence-transformers and TF-IDF."""

import numpy as np
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Embedder:
    """Lazy-loading wrapper for sentence-transformers.

    Uses paraphrase-multilingual-MiniLM-L12-v2 which supports 50+ languages
    including English and Chinese, and is small enough for CPU deployment.
    """

    _instance = None
    _model = None

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32,
              show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        # Filter empty texts
        non_empty = [t if t.strip() else " " for t in texts]
        embeddings = self.model.encode(
            non_empty,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        emb = self.embed([text_a, text_b])
        return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    def pairwise_similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """Compute pairwise cosine similarity for a list of texts."""
        emb = self.embed(texts)
        return cosine_similarity(emb)


class TfidfPhraseFinder:
    """TF-IDF based phrase similarity for boilerplate detection.

    Uses character n-grams for language-agnostic matching of near-identical
    expressions.
    """

    def __init__(self, char_ngram_range: tuple = (3, 6), max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngram_range,
            max_features=max_features,
            lowercase=True,
        )

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit the vectorizer and transform texts to TF-IDF vectors.

        Returns sparse matrix of shape (len(texts), n_features).
        """
        non_empty = [t if t.strip() else "placeholder" for t in texts]
        return self.vectorizer.fit_transform(non_empty)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform new texts using the fitted vectorizer."""
        return self.vectorizer.transform(texts)

    def find_similar_pairs(self, texts: list[str], threshold: float = 0.65,
                           max_pairs: int = 500) -> list[dict]:
        """Find pairs of texts with high TF-IDF cosine similarity.

        Args:
            texts: List of text strings to compare.
            threshold: Minimum cosine similarity to flag.
            max_pairs: Maximum number of pairs to return.

        Returns list of {idx_a, idx_b, similarity, text_a, text_b}.
        """
        if len(texts) < 2:
            return []

        tfidf_matrix = self.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)

        pairs = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i][j])
                if sim >= threshold:
                    pairs.append({
                        "idx_a": i,
                        "idx_b": j,
                        "similarity": round(sim, 4),
                        "text_a": texts[i][:300],
                        "text_b": texts[j][:300],
                    })
                    if len(pairs) >= max_pairs:
                        pairs.sort(key=lambda x: x["similarity"], reverse=True)
                        return pairs

        pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return pairs


def extract_shared_phrase(text_a: str, text_b: str, min_length: int = 10) -> str:
    """Extract the longest common substring from two texts (approximate).

    Uses a word-level sliding window for efficiency.
    """
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    longest = ""

    for i in range(len(words_a)):
        for j in range(i + 3, min(i + 40, len(words_a) + 1)):
            window = " ".join(words_a[i:j])
            if window in text_b.lower() and len(window) > len(longest):
                longest = window

    return longest if len(longest) >= min_length else ""


def cluster_texts(embeddings: np.ndarray, threshold: float = 0.75,
                  min_cluster_size: int = 2) -> list[list[int]]:
    """Cluster texts using agglomerative clustering on cosine distance.

    Args:
        embeddings: Normalized embedding matrix.
        threshold: Cosine distance threshold for clustering.
        min_cluster_size: Minimum number of items per cluster.

    Returns list of clusters, each cluster being a list of indices.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    if len(embeddings) < 2:
        return [[0]] if len(embeddings) == 1 else []

    # Compute cosine distance
    dist = pdist(embeddings, metric="cosine")
    # Ward linkage on cosine distance
    Z = linkage(dist, method="ward")
    # Cut at threshold (cosine distance threshold)
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    # Group by label
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    # Filter by min size
    return [c for c in clusters.values() if len(c) >= min_cluster_size]
