"""Introduction background clustering.

Groups papers that share similar introduction backgrounds,
indicating they address the same research context.
"""

import numpy as np
from ..embeddings import Embedder, cluster_texts


def cluster_introductions(papers: list, embedder: Embedder = None,
                          threshold: float = 0.72) -> dict:
    """Cluster papers by introduction similarity.

    Args:
        papers: List of Paper objects.
        embedder: Embedder instance.
        threshold: Cosine similarity threshold for clustering.

    Returns dict with:
        - clusters: list of {label, papers, key_terms, representative_excerpt}
        - similarity_matrix: pairwise similarity of introductions
        - paper_labels: short labels
        - graph_edges: list of (source, target, weight) for visualization
    """
    if embedder is None:
        embedder = Embedder()

    n = len(papers)
    if n < 2:
        return {"clusters": [], "similarity_matrix": [], "paper_labels": [], "graph_edges": []}

    paper_labels = [_short_label(p) for p in papers]

    # Get introduction text from each paper
    intro_texts = []
    for p in papers:
        intro = p.get_section("introduction") or p.get_section("related_work") or p.abstract
        intro_texts.append(intro)

    # Embed introductions
    intro_emb = embedder.embed(intro_texts)

    # Similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(intro_emb)
    sim_list = np.round(sim_matrix, 4).tolist()

    # Cluster
    raw_clusters = cluster_texts(intro_emb, threshold=threshold, min_cluster_size=1)

    # Build cluster outputs
    clusters_out = []
    for idx_list in raw_clusters:
        # Generate a representative excerpt and key terms
        combined = " ".join([intro_texts[i] for i in idx_list])
        key_terms = _extract_key_terms(combined, top_n=8)

        # Representative excerpt: the central paper's intro (first few sentences)
        central_idx = idx_list[0]
        excerpt = intro_texts[central_idx][:500]

        clusters_out.append({
            "label": _generate_label(key_terms),
            "key_terms": key_terms,
            "papers": [paper_labels[i] for i in idx_list],
            "paper_indices": idx_list,
            "paper_count": len(idx_list),
            "representative_excerpt": excerpt,
        })

    clusters_out.sort(key=lambda c: c["paper_count"], reverse=True)

    # Build graph edges for visualization
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] > 0.55:  # Lower threshold for edges
                edges.append({
                    "source": paper_labels[i],
                    "target": paper_labels[j],
                    "weight": round(float(sim_matrix[i][j]), 3),
                })

    return {
        "clusters": clusters_out,
        "similarity_matrix": sim_list,
        "paper_labels": paper_labels,
        "graph_edges": edges,
    }


def analyze_background_overlap(papers: list, embedder: Embedder = None) -> list[dict]:
    """Find overlapping background statements shared in introductions.

    Analyzes sentence-level similarity across introduction sections to identify
    common background claims (e.g., "X has attracted significant attention").
    """
    if embedder is None:
        embedder = Embedder()

    from ..utils import extract_sentences
    from ..embeddings import TfidfPhraseFinder

    # Collect all introduction sentences with paper index
    sent_map = []
    for i, p in enumerate(papers):
        intro = p.get_section("introduction") or ""
        sentences = extract_sentences(intro, min_length=25)
        for s in sentences:
            sent_map.append((i, s))

    if len(sent_map) < 3:
        return []

    sentences = [s[1] for s in sent_map]
    tfidf = TfidfPhraseFinder(char_ngram_range=(4, 7), max_features=5000)
    pairs = tfidf.find_similar_pairs(sentences, threshold=0.72, max_pairs=200)

    # Group by which papers they appear in
    from collections import defaultdict
    overlap_groups = []
    seen = set()

    for pair in pairs:
        paper_a = sent_map[pair["idx_a"]][0]
        paper_b = sent_map[pair["idx_b"]][0]
        if paper_a == paper_b:
            continue
        key = tuple(sorted([pair["idx_a"], pair["idx_b"]]))
        if key in seen:
            continue
        seen.add(key)

        overlap_groups.append({
            "paper_a": _short_label(papers[paper_a]),
            "paper_b": _short_label(papers[paper_b]),
            "text_a": pair["text_a"][:300],
            "text_b": pair["text_b"][:300],
            "similarity": pair["similarity"],
        })

    overlap_groups.sort(key=lambda x: x["similarity"], reverse=True)
    return overlap_groups[:50]


def _short_label(paper) -> str:
    if paper.title:
        words = paper.title.split()
        return " ".join(words[:4]) + ("..." if len(words) > 4 else "")
    return paper.source[:30]


def _extract_key_terms(text: str, top_n: int = 8) -> list[str]:
    """Extract key terms from text."""
    import re

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "and", "but", "or", "nor", "not", "so",
        "than", "too", "very", "this", "that", "these", "those",
        "it", "its", "we", "they", "them", "their", "our",
        "which", "who", "what", "when", "where", "how",
        "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "only", "also", "based",
        "been", "used", "using", "however", "although",
        "therefore", "thus", "hence", "due", "since", "well",
    }

    words = re.findall(r"[a-zA-Z]{2,}(?:-[a-zA-Z]{2,})*", text.lower())
    phrases = []
    for i in range(len(words)):
        if words[i] not in stopwords:
            phrases.append(words[i])
            if i + 1 < len(words):
                phrases.append(f"{words[i]} {words[i+1]}")
            if i + 2 < len(words):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

    from collections import Counter
    counts = Counter(phrases)
    scored = [(p, c * len(p.split())) for p, c in counts.items() if len(p) > 4 and c >= 2]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:top_n]]


def _generate_label(key_terms: list[str]) -> str:
    """Generate a label from key terms."""
    if not key_terms:
        return "Unlabeled cluster"
    if len(key_terms) >= 2:
        return f"{key_terms[0].title()} | {key_terms[1].title()}"
    return key_terms[0].title()
