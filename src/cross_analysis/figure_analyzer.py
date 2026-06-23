"""Figure similarity analysis and discussion pattern detection.

Groups similar figures across papers by caption similarity and identifies
templated discussion patterns for figure types.
"""

from ..embeddings import Embedder, cluster_texts, TfidfPhraseFinder
from ..utils import extract_sentences


def analyze_figures(papers: list, embedder: Embedder = None,
                    caption_threshold: float = 0.70) -> dict:
    """Analyze figure similarity and discussion patterns across papers.

    Args:
        papers: List of Paper objects with extracted figures.
        embedder: Embedder instance.
        caption_threshold: Cosine similarity threshold for caption clustering.

    Returns dict with:
        - figure_groups: list of figure groups with similar captions
        - discussion_patterns: templated discussion phrases per group
        - stats: overall statistics
    """
    if embedder is None:
        embedder = Embedder()

    # Collect all figures across papers
    all_figures = []  # List of (paper_idx, figure_info)
    for i, p in enumerate(papers):
        for fig in p.figures:
            if fig.caption and len(fig.caption) > 10:
                all_figures.append((i, fig))

    if len(all_figures) < 2:
        return {"figure_groups": [], "discussion_patterns": [], "stats": {"total_figures": len(all_figures)}}

    # Embed captions
    captions = [f[1].caption for f in all_figures]
    caption_emb = embedder.embed(captions)

    # Cluster captions
    raw_clusters = cluster_texts(caption_emb, threshold=caption_threshold, min_cluster_size=1)

    paper_labels = [_short_label(p) for p in papers]
    figure_groups = []
    discussion_patterns = []

    for idx_list in raw_clusters:
        if len(idx_list) < 2:  # Skip single-figure groups
            continue

        papers_in = set()
        captions_text = ""
        fig_items = []
        discussion_texts = []

        for idx in idx_list:
            paper_idx, fig = all_figures[idx]
            papers_in.add(paper_idx)
            captions_text += fig.caption + " "
            fig_items.append({
                "paper_label": paper_labels[paper_idx],
                "paper_idx": paper_idx,
                "figure_index": fig.index,
                "caption": fig.caption[:300],
                "has_image": fig.image_bytes is not None,
            })
            # Collect discussion paragraphs
            for dp in fig.discussion_paragraphs:
                discussion_texts.append(dp)

        if len(papers_in) >= 2:
            group_label = _summarize_caption_group(captions_text)
            fig_groups = {
                "label": group_label,
                "paper_count": len(papers_in),
                "figure_count": len(idx_list),
                "figures": fig_items,
            }
            figure_groups.append(fig_groups)

            # Find templated discussion patterns for this figure group
            if len(discussion_texts) >= 3:
                patterns = _find_discussion_patterns(discussion_texts)
                if patterns:
                    discussion_patterns.append({
                        "figure_group": group_label,
                        "patterns": patterns,
                    })

    figure_groups.sort(key=lambda g: g["paper_count"], reverse=True)

    return {
        "figure_groups": figure_groups,
        "discussion_patterns": discussion_patterns,
        "stats": {
            "total_figures": len(all_figures),
            "total_papers": len(papers),
            "figure_groups_found": len(figure_groups),
            "discussion_patterns_found": len(discussion_patterns),
        },
    }


def find_similar_figure_types(papers: list, embedder: Embedder = None) -> list[dict]:
    """Identify common figure types shared across papers.

    A "figure type" is a category like "band structure", "XRD pattern",
    "SEM image", "schematic diagram", etc.

    Returns list of {figure_type, papers, total_figures}.
    """
    if embedder is None:
        embedder = Embedder()

    figure_types = [
        # Common figure types in scientific papers
        ("band structure", ["band structure", "band diagram", "dispersion", "band gap"]),
        ("density of states", ["density of states", "DOS", "PDOS", "partial density"]),
        ("XRD pattern", ["XRD", "X-ray diffraction", "diffraction pattern"]),
        ("SEM image", ["SEM", "scanning electron", "morphology"]),
        ("TEM image", ["TEM", "transmission electron", "HRTEM"]),
        ("AFM image", ["AFM", "atomic force"]),
        ("Raman spectrum", ["Raman", "Raman shift", "Raman spectroscopy"]),
        ("XPS spectrum", ["XPS", "X-ray photoelectron"]),
        ("FTIR spectrum", ["FTIR", "infrared", "IR spectrum"]),
        ("UV-vis spectrum", ["UV-vis", "ultraviolet-visible", "absorption spectrum"]),
        ("TGA/DSC", ["TGA", "DSC", "thermogravimetric", "differential scanning"]),
        ("crystal structure", ["crystal structure", "unit cell", "lattice"]),
        ("microstructure", ["microstructure", "grain", "texture", "morphology"]),
        ("phase diagram", ["phase diagram", "phase transition"]),
        ("schematic diagram", ["schematic", "diagram", "illustration"]),
        ("energy level diagram", ["energy level", "energy diagram", "band alignment"]),
        ("performance comparison", ["comparison", "benchmark", "comparison table"]),
        ("error/uncertainty", ["error bar", "uncertainty", "standard deviation"]),
        ("correlation plot", ["correlation", "scatter plot", "vs.", "versus"]),
        ("convergence test", ["convergence", "k-point", "cutoff energy", "basis set"]),
        # Chinese
        ("能带结构", ["能带", "band structure", "色散"]),
        ("态密度", ["态密度", "DOS"]),
        ("XRD图谱", ["XRD", "衍射"]),
        ("SEM图像", ["SEM", "扫描电镜", "形貌"]),
        ("拉曼光谱", ["拉曼", "Raman"]),
        ("示意图", ["示意图", "schematic", "原理图"]),
        ("性能对比", ["对比", "comparison", "性能"]),
    ]

    results = []
    paper_labels = [_short_label(p) for p in papers]

    for ftype, keywords in figure_types:
        papers_with = []
        fig_count = 0
        for i, p in enumerate(papers):
            for fig in p.figures:
                cap_lower = fig.caption.lower()
                if any(kw.lower() in cap_lower for kw in keywords):
                    if i not in [x["paper_idx"] for x in papers_with]:
                        papers_with.append({
                            "paper_idx": i,
                            "paper_label": paper_labels[i],
                        })
                    fig_count += 1

        if len(papers_with) >= 2:
            results.append({
                "figure_type": ftype,
                "papers": papers_with,
                "paper_count": len(papers_with),
                "total_figures": fig_count,
            })

    results.sort(key=lambda r: r["paper_count"], reverse=True)
    return results


def _find_discussion_patterns(discussion_texts: list[str]) -> list[dict]:
    """Find templated discussion phrases among figure discussion paragraphs."""
    tfidf = TfidfPhraseFinder(char_ngram_range=(4, 7), max_features=3000)
    pairs = tfidf.find_similar_pairs(discussion_texts, threshold=0.68, max_pairs=50)

    patterns = []
    seen_texts = set()
    for pair in pairs[:20]:
        # Extract the shared phrasing
        from ..embeddings import extract_shared_phrase
        shared = extract_shared_phrase(pair["text_a"], pair["text_b"], min_length=15)
        if shared and shared not in seen_texts:
            seen_texts.add(shared)
            patterns.append({
                "shared_phrase": shared[:200],
                "similarity": pair["similarity"],
                "text_a": pair["text_a"][:250],
                "text_b": pair["text_b"][:250],
            })

    return patterns[:10]


def _summarize_caption_group(captions_text: str, max_len: int = 100) -> str:
    """Summarize a group of similar captions."""
    # Extract key terms from combined captions
    words = captions_text.lower().split()[:50]
    # Remove common words
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "to", "for",
                 "with", "and", "or", "is", "are", "was", "were", "be",
                 "by", "from", "as", "this", "that", "it", "its"}
    content = [w for w in words if w not in stopwords and len(w) > 2]

    if not content:
        return "Unlabeled figure group"

    # Try to find a meaningful phrase
    joined = " ".join(content)
    if len(joined) > max_len:
        joined = joined[:max_len - 3] + "..."
    return joined


def _short_label(paper) -> str:
    if paper.title:
        words = paper.title.split()
        return " ".join(words[:4]) + ("..." if len(words) > 4 else "")
    return paper.source[:30]
