"""Cross-paper method similarity detection.

Clusters similar methodology paragraphs across papers to identify
shared methods, techniques, and approaches.
"""

import numpy as np
from collections import defaultdict

from ..embeddings import Embedder, cluster_texts
from ..utils import extract_sentences


def analyze_method_similarity(papers: list, embedder: Embedder = None,
                              threshold: float = 0.75) -> dict:
    """Find shared methods across papers.

    Args:
        papers: List of Paper objects.
        embedder: Embedder instance (created if None).
        threshold: Cosine similarity threshold for clustering.

    Returns dict with:
        - clusters: list of {label, key_terms, papers, paragraphs}
        - similarity_matrix: NxN pairwise similarity
        - paper_labels: labels for each paper (for heatmap)
    """
    if embedder is None:
        embedder = Embedder()

    n = len(papers)
    if n < 2:
        return {"clusters": [], "similarity_matrix": [], "paper_labels": []}

    # Get methods section text from each paper
    methods_texts = []
    paper_labels = []
    for p in papers:
        label = _short_label(p)
        paper_labels.append(label)
        section_text = p.get_section("methods") or p.get_section("results") or p.all_section_text()
        methods_texts.append(section_text)

    # Compute full-section similarity matrix
    emb_methods = embedder.embed(methods_texts)
    sim_matrix = _compute_sim_matrix(emb_methods)
    sim_list = sim_matrix.tolist()

    # Split into paragraphs for fine-grained clustering
    all_paragraphs = []  # List of (paper_idx, para_text)
    for i, text in enumerate(methods_texts):
        paragraphs = _split_paragraphs(text)
        for para in paragraphs:
            if len(para) > 100:  # Only substantial paragraphs
                all_paragraphs.append((i, para))

    if not all_paragraphs:
        return {"clusters": [], "similarity_matrix": sim_list, "paper_labels": paper_labels}

    para_texts = [p[1] for p in all_paragraphs]
    para_embeddings = embedder.embed(para_texts)

    # Cluster paragraphs
    raw_clusters = cluster_texts(para_embeddings, threshold=threshold, min_cluster_size=2)

    # Build result clusters
    clusters_out = []
    for cluster_indices in raw_clusters:
        # Get paper membership
        paper_set = set()
        combined_text = ""
        cluster_paras = []
        for idx in cluster_indices:
            paper_idx, para_text = all_paragraphs[idx]
            paper_set.add(paper_idx)
            combined_text += para_text + " "
            cluster_paras.append({
                "paper_idx": paper_idx,
                "paper_label": paper_labels[paper_idx],
                "text": para_text[:500],
            })

        # Only include clusters that span multiple papers
        if len(paper_set) >= 2:
            key_terms = _extract_key_terms(combined_text)
            clusters_out.append({
                "label": _generate_label(key_terms),
                "key_terms": key_terms,
                "papers": [paper_labels[i] for i in paper_set],
                "paper_count": len(paper_set),
                "paragraph_count": len(cluster_paras),
                "paragraphs": cluster_paras,
            })

    # Sort by number of papers (more shared = more interesting)
    clusters_out.sort(key=lambda c: c["paper_count"], reverse=True)

    return {
        "clusters": clusters_out,
        "similarity_matrix": sim_list,
        "paper_labels": paper_labels,
    }


def analyze_shared_methods_intersection(papers: list, embedder: Embedder = None) -> dict:
    """Analyze which specific methods appear in which papers (intersection matrix).

    Returns dict with method_names (list), paper_labels (list),
    and presence_matrix (list of lists).
    """
    if embedder is None:
        embedder = Embedder()

    # Common method keywords to search for
    method_keywords = [
        # Computational
        "density functional theory", "DFT", "molecular dynamics", "Monte Carlo",
        "finite element", "FEM", "machine learning", "neural network",
        "deep learning", "ab initio", "first principles", "tight binding",
        "phase field", "kinetic Monte Carlo", "coarse grained",
        # Experimental
        "X-ray diffraction", "XRD", "scanning electron microscopy", "SEM",
        "transmission electron microscopy", "TEM", "atomic force microscopy", "AFM",
        "X-ray photoelectron spectroscopy", "XPS", "Raman spectroscopy",
        "nuclear magnetic resonance", "NMR", "differential scanning calorimetry",
        "thermogravimetric analysis", "TGA", "FTIR", "UV-vis",
        # Analysis
        "principal component analysis", "PCA", "singular value decomposition",
        "Fourier transform", "wavelet", "regression", "classification",
        "clustering", "Bayesian", "statistical", "error analysis",
        # Chinese
        "第一性原理", "密度泛函", "分子动力学", "蒙特卡罗", "机器学习",
        "神经网络", "X射线衍射", "扫描电镜", "透射电镜", "拉曼光谱",
    ]

    paper_labels = [_short_label(p) for p in papers]
    presence_matrix = []
    detected_methods = []

    for kw in method_keywords:
        row = []
        for p in papers:
            full_text = p.full_text.lower()
            row.append(1 if kw.lower() in full_text else 0)
        if sum(row) >= 2:  # At least 2 papers use this method
            presence_matrix.append(row)
            detected_methods.append(kw)

    return {
        "method_names": detected_methods,
        "paper_labels": paper_labels,
        "presence_matrix": presence_matrix,
    }


def _short_label(paper) -> str:
    """Generate a short label for a paper."""
    if paper.title:
        words = paper.title.split()
        # First author-like label or first few words
        if len(words) <= 5:
            return paper.title[:60]
        return " ".join(words[:5]) + "..."
    return paper.source[:40]


def _split_paragraphs(text: str, min_len: int = 50) -> list[str]:
    """Split text into meaningful paragraphs."""
    if not text:
        return []
    paras = text.split("\n\n")
    result = []
    for p in paras:
        p = p.strip().replace("\n", " ")
        if len(p) >= min_len:
            result.append(p)
    # If no paragraph breaks, split by sentences and group
    if not result:
        sentences = extract_sentences(text, min_length=20)
        # Group sentences into pseudo-paragraphs of ~3 sentences
        for i in range(0, len(sentences), 3):
            chunk = " ".join(sentences[i:i + 3])
            if len(chunk) >= min_len:
                result.append(chunk)
    return result


def _compute_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embeddings)
    return np.round(sim, 4)


def _extract_key_terms(text: str, top_n: int = 5) -> list[str]:
    """Extract key technical terms from text using simple frequency."""
    import re
    # Remove common words and extract technical-looking terms
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "and", "but", "or", "nor", "not", "so",
        "than", "too", "very", "this", "that", "these", "those",
        "it", "its", "we", "they", "them", "their", "our", "my",
        "his", "her", "he", "she", "which", "who", "whom", "what",
        "when", "where", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "no", "only",
        "also", "been", "being", "used", "using", "based",
    }

    # Extract n-grams (1-3 words) that look technical
    words = re.findall(r"[a-zA-Z]{2,}(?:-[a-zA-Z]{2,})*", text.lower())
    phrases = []
    for i in range(len(words)):
        if words[i] not in stopwords:
            # Single word
            phrases.append(words[i])
            # Bigram
            if i + 1 < len(words):
                phrases.append(f"{words[i]} {words[i+1]}")
            # Trigram
            if i + 2 < len(words):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

    # Count frequency, prefer multi-word phrases
    from collections import Counter
    counts = Counter(phrases)
    # Weight longer phrases higher
    scored = [(p, c * len(p.split())) for p, c in counts.items() if len(p) > 4]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:top_n]]


def _generate_label(key_terms: list[str]) -> str:
    """Generate a human-readable label from key terms."""
    if not key_terms:
        return "Unlabeled method cluster"
    if len(key_terms) >= 2:
        return f"{key_terms[0].title()} + {key_terms[1].title()}"
    return key_terms[0].title()
