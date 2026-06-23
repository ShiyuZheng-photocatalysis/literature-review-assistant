"""Problem statement extraction and cross-paper analysis.

Extracts the research problem each paper addresses, then clusters
similar problems across papers.
"""

from ..embeddings import Embedder, cluster_texts
from ..utils import extract_sentences


def extract_problems(papers: list) -> list[dict]:
    """Extract research problems from each paper.

    Uses keyword-based + heuristic extraction on abstract, introduction,
    and conclusion sections.

    Returns list of {paper_label, problems: [{sentence, section, keyword}]}.
    """
    results = []
    for paper in papers:
        problems = _extract_one_paper_problems(paper)
        results.append({
            "paper_label": _short_label(paper),
            "paper_source": paper.source,
            "problems": problems,
            "problem_count": len(problems),
        })
    return results


def cluster_problems_across_papers(papers: list, embedder: Embedder = None,
                                   threshold: float = 0.72) -> dict:
    """Cluster similar problems across papers.

    Returns dict with:
        - problem_clusters: list of {label, problems, paper_count}
        - per_paper: list of per-paper problems (from extract_problems)
        - problem_matrix: paper x cluster presence matrix
    """
    if embedder is None:
        embedder = Embedder()

    per_paper = extract_problems(papers)

    # Collect all problem sentences
    all_problems = []  # List of (paper_idx, sentence)
    for i, pp in enumerate(per_paper):
        for prob in pp["problems"]:
            all_problems.append((i, prob["sentence"]))

    if not all_problems:
        return {"problem_clusters": [], "per_paper": per_paper, "problem_matrix": []}

    # Embed and cluster
    sentences = [p[1] for p in all_problems]
    embeddings = embedder.embed(sentences)
    raw_clusters = cluster_texts(embeddings, threshold=threshold, min_cluster_size=1)

    # Build problem clusters
    problem_clusters = []
    for idx_list in raw_clusters:
        papers_in = set()
        combined = ""
        items = []
        for idx in idx_list:
            paper_idx, sent = all_problems[idx]
            papers_in.add(paper_idx)
            combined += sent + " "
            items.append({
                "paper_label": per_paper[paper_idx]["paper_label"],
                "paper_idx": paper_idx,
                "text": sent[:400],
            })

        label = _summarize_problem(combined)
        problem_clusters.append({
            "label": label,
            "paper_count": len(papers_in),
            "problem_count": len(items),
            "items": items,
        })

    problem_clusters.sort(key=lambda c: c["paper_count"], reverse=True)

    # Build paper x cluster presence matrix
    paper_labels = [pp["paper_label"] for pp in per_paper]
    cluster_labels = [c["label"] for c in problem_clusters]
    matrix = []
    for i in range(len(papers)):
        row = []
        for c in problem_clusters:
            in_cluster = any(item["paper_idx"] == i for item in c["items"])
            row.append(1 if in_cluster else 0)
        matrix.append(row)

    return {
        "problem_clusters": problem_clusters,
        "per_paper": per_paper,
        "problem_matrix": {
            "paper_labels": paper_labels,
            "cluster_labels": cluster_labels,
            "matrix": matrix,
        },
    }


def compare_problem_focus(papers: list) -> list[dict]:
    """Identify what makes each paper's problem focus unique.

    Returns list of {paper_label, unique_focus, shared_focus}.
    """
    if len(papers) < 2:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Get problem-related text from each paper
    paper_texts = []
    for p in papers:
        text = p.abstract + " " + (p.get_section("introduction") or "")
        paper_texts.append(text)

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    tfidf = vectorizer.fit_transform(paper_texts)
    feature_names = vectorizer.get_feature_names_out()

    results = []
    for i, p in enumerate(papers):
        # Top TF-IDF terms for this paper
        row = tfidf[i].toarray().flatten()
        top_indices = row.argsort()[-10:][::-1]
        top_terms = [(feature_names[j], round(row[j], 4)) for j in top_indices if row[j] > 0]

        results.append({
            "paper_label": _short_label(p),
            "top_terms": top_terms,
            "problem_count": len(_extract_one_paper_problems(p)),
        })

    return results


def _extract_one_paper_problems(paper) -> list[dict]:
    """Extract problem statements from a single paper."""
    from ..pdf_processor import PROBLEM_KEYWORDS_EN, PROBLEM_KEYWORDS_ZH
    from ..pdf_processor import _split_sentences

    problems = []
    sections_to_check = [
        ("abstract", paper.abstract),
        ("introduction", paper.introduction),
        ("discussion", paper.discussion),
        ("conclusion", paper.conclusion),
    ]

    keywords = PROBLEM_KEYWORDS_EN
    if paper.language in ("zh", "mixed"):
        keywords = PROBLEM_KEYWORDS_EN + PROBLEM_KEYWORDS_ZH

    for section_name, section_text in sections_to_check:
        if not section_text:
            continue
        sentences = _split_sentences(section_text)
        for sent in sentences:
            sent_lower = sent.lower()
            for kw in keywords:
                if kw.lower() in sent_lower:
                    problems.append({
                        "sentence": sent.strip(),
                        "section": section_name,
                        "keyword": kw,
                    })
                    break

    # Deduplicate by sentence content
    seen = set()
    unique = []
    for p in problems:
        key = p["sentence"][:80].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _short_label(paper) -> str:
    if paper.title:
        words = paper.title.split()
        return " ".join(words[:4]) + ("..." if len(words) > 4 else "")
    return paper.source[:30]


def _summarize_problem(combined_text: str, max_len: int = 120) -> str:
    """Generate a short summary label for a problem cluster."""
    # Return first substantial sentence as the problem label
    sentences = extract_sentences(combined_text, min_length=15)
    if not sentences:
        return "Unlabeled problem"
    # Prefer sentences with problem keywords
    kw_sentences = []
    for s in sentences:
        for kw in ["challenge", "problem", "limitation", "remains", "however", "gap",
                    "缺乏", "困难", "挑战", "问题"]:
            if kw in s.lower():
                kw_sentences.append(s)
                break
    candidate = kw_sentences[0] if kw_sentences else sentences[0]
    if len(candidate) > max_len:
        candidate = candidate[:max_len - 3] + "..."
    return candidate
