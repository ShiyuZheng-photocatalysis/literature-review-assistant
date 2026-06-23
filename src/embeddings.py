"""Text embedding and similarity — light TF-IDF by default, optional sentence-transformers.

Strategy: Use sklearn TF-IDF (fast, no GPU, tiny install) as the primary engine.
If sentence-transformers is installed, use it for semantic similarity instead.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Try to import sentence-transformers for better semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class Embedder:
    """Text embedder that auto-selects sentence-transformers or TF-IDF.

    Uses sentence-transformers if available (better semantic understanding).
    Falls back to TF-IDF word n-grams if not (fast, lightweight, always works).
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._st_model = None
        self._tfidf = None
        self._tfidf_texts = None
        self._mode = "sentence-transformers" if HAS_SENTENCE_TRANSFORMERS else "tfidf"

    @property
    def mode(self) -> str:
        return self._mode

    def _get_st_model(self):
        if self._st_model is None and HAS_SENTENCE_TRANSFORMERS:
            self._st_model = SentenceTransformer(self.model_name)
        return self._st_model

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])

        if HAS_SENTENCE_TRANSFORMERS:
            non_empty = [t if t.strip() else " " for t in texts]
            return self._get_st_model().encode(
                non_empty, show_progress_bar=show_progress,
                convert_to_numpy=True, normalize_embeddings=True,
            )
        else:
            # TF-IDF fallback
            if self._tfidf is None:
                self._tfidf = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=5000,
                    lowercase=True,
                    stop_words="english",
                )
            clean = [t if t.strip() else "placeholder" for t in texts]
            tfidf_matrix = self._tfidf.fit_transform(clean)
            # Convert to dense and normalize
            dense = tfidf_matrix.toarray().astype(np.float32)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense / norms

    def embed_with_cache(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Same as embed(), for backward compat."""
        return self.embed(texts, show_progress=show_progress)

    def similarity(self, text_a: str, text_b: str) -> float:
        emb = self.embed([text_a, text_b])
        return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    def pairwise_similarity_matrix(self, texts: list[str]) -> np.ndarray:
        emb = self.embed(texts)
        return cosine_similarity(emb)


class TfidfPhraseFinder:
    """TF-IDF based phrase similarity for boilerplate detection.

    Uses character n-grams for language-agnostic matching of near-identical
    expressions. Always uses sklearn — no heavy dependencies.
    """

    def __init__(self, char_ngram_range: tuple = (3, 6), max_features: int = 8000):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngram_range,
            max_features=max_features,
            lowercase=True,
        )

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        non_empty = [t if t.strip() else "placeholder" for t in texts]
        return self.vectorizer.fit_transform(non_empty)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts)

    def find_similar_pairs(self, texts: list[str], threshold: float = 0.65,
                           max_pairs: int = 500) -> list[dict]:
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
    """Extract the longest common substring from two texts.

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
    """Cluster texts using agglomerative clustering on cosine distance."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    if len(embeddings) < 2:
        return [[0]] if len(embeddings) == 1 else []

    dist = pdist(embeddings, metric="cosine")
    Z = linkage(dist, method="ward")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    return [c for c in clusters.values() if len(c) >= min_cluster_size]
