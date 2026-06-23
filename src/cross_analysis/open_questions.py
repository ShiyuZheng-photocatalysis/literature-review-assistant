"""Open questions and unsolved problems aggregation.

Extracts and clusters "future work" and "open question" statements
from across papers to identify converging research gaps.
"""

from ..embeddings import Embedder, cluster_texts
from ..utils import extract_sentences
from ..pdf_processor import _split_sentences


def aggregate_open_questions(papers: list, embedder: Embedder = None,
                             threshold: float = 0.70) -> dict:
    """Aggregate and cluster open questions across papers.

    Args:
        papers: List of Paper objects.
        embedder: Embedder instance.
        threshold: Cosine similarity threshold for clustering.

    Returns dict with:
        - convergent_questions: list of questions identified by >= 2 papers
        - unique_questions: list of questions from only 1 paper
        - per_paper: per-paper open question lists
        - stats: summary statistics
    """
    if embedder is None:
        embedder = Embedder()

    # Extract open questions from each paper
    per_paper = []
    all_questions = []  # List of (paper_idx, question_text)

    for i, paper in enumerate(papers):
        questions = _extract_open_questions_one_paper(paper)
        per_paper.append({
            "paper_label": _short_label(paper),
            "paper_source": paper.source,
            "questions": questions,
            "count": len(questions),
        })
        for q in questions:
            all_questions.append((i, q))

    if not all_questions:
        return {
            "convergent_questions": [],
            "unique_questions": [],
            "per_paper": per_paper,
            "stats": {"total_papers": len(papers), "total_questions": 0, "convergent": 0},
        }

    # Cluster questions
    question_texts = [q[1] for q in all_questions]
    embeddings = embedder.embed(question_texts)
    raw_clusters = cluster_texts(embeddings, threshold=threshold, min_cluster_size=1)

    paper_labels = [_short_label(p) for p in papers]
    convergent = []
    unique = []

    for idx_list in raw_clusters:
        papers_in = set()
        items = []
        combined = ""

        for idx in idx_list:
            paper_idx, qtext = all_questions[idx]
            papers_in.add(paper_idx)
            combined += qtext + " "
            items.append({
                "paper_label": paper_labels[paper_idx],
                "paper_idx": paper_idx,
                "text": qtext[:400],
            })

        summary = _summarize_question(combined)
        entry = {
            "summary": summary,
            "paper_count": len(papers_in),
            "question_count": len(items),
            "items": items,
        }

        if len(papers_in) >= 2:
            convergent.append(entry)
        else:
            unique.append(entry)

    # Sort by paper count (convergent first, most cited first)
    convergent.sort(key=lambda q: q["paper_count"], reverse=True)
    unique.sort(key=lambda q: len(q["items"][0]["text"]), reverse=True)

    return {
        "convergent_questions": convergent,
        "unique_questions": unique,
        "per_paper": per_paper,
        "stats": {
            "total_papers": len(papers),
            "total_questions": len(all_questions),
            "convergent": len(convergent),
            "unique": len(unique),
        },
    }


def find_research_gaps(papers: list) -> list[dict]:
    """Identify research gaps mentioned across papers.

    Focuses on "not yet", "remains to be", "lack of" type statements
    that explicitly indicate what is missing in the field.
    """
    gap_patterns = [
        r"(?:has\s+not\s+(?:yet\s+)?been|have\s+not\s+(?:yet\s+)?been|remains?\s+(?:to\s+be|unclear|unknown|elusive|poorly\s+understood))",
        r"(?:lack\s+of|little\s+is\s+known|few\s+studies|no\s+(?:systematic|comprehensive|previous)\s+(?:study|investigation|work))",
        r"(?:further\s+(?:investigation|study|research|work)\s+(?:is|are)\s+(?:needed|required|necessary|warranted))",
        r"(?:有待|尚无|鲜有|缺乏|不足|仍需|尚需|亟待|亟需)",
    ]

    gaps = []
    paper_labels = [_short_label(p) for p in papers]

    for i, paper in enumerate(papers):
        sections = [paper.discussion, paper.conclusion, paper.introduction]
        for section_text in sections:
            if not section_text:
                continue
            sentences = _split_sentences(section_text)
            for sent in sentences:
                for pattern in gap_patterns:
                    import re
                    if re.search(pattern, sent, re.IGNORECASE):
                        clean = sent.strip()
                        if len(clean) > 30:
                            gaps.append({
                                "paper_label": paper_labels[i],
                                "paper_idx": i,
                                "text": clean[:500],
                            })
                            break

    # Group similar gaps
    if len(gaps) >= 2:
        _group_similar_gaps(gaps)

    return gaps


def _extract_open_questions_one_paper(paper) -> list[str]:
    """Extract open questions from a single paper."""
    future_patterns = [
        r"(?:future\s*work|further\s*(?:investigation|study|research|exploration|development))",
        r"(?:remains?\s*(?:to\s*be|unclear|unknown|elusive|an?\s*open|poorly\s*understood|challenging))",
        r"(?:warrants?\s*(?:further|additional|more|continued))",
        r"(?:open\s*question|outstanding\s*(?:question|problem|challenge|issue))",
        r"(?:有待|尚待|仍需|需进一步|值得进一步|尚未|尚不|仍不|未来工作|展望|未解决)",
        r"(?:beyond\s*the\s*scope|outside\s*the\s*scope|not\s*considered|not\s*addressed)",
        r"(?:needs?\s*(?:further|more|additional|to\s*be)\s*(?:investigation|study|research|explored|examined))",
    ]

    questions = []
    sections = [paper.discussion, paper.conclusion, paper.introduction]

    for section_text in sections:
        if not section_text:
            continue
        sentences = _split_sentences(section_text)
        for sent in sentences:
            import re
            for pattern in future_patterns:
                if re.search(pattern, sent, re.IGNORECASE):
                    clean = sent.strip()
                    if len(clean) > 30 and clean not in questions:
                        questions.append(clean)
                    break

    return questions


def _group_similar_gaps(gaps: list[dict]) -> None:
    """Add a grouping key to similar gaps."""
    if len(gaps) < 2:
        return

    embedder = Embedder()
    texts = [g["text"] for g in gaps]
    embeddings = embedder.embed(texts)
    clusters = cluster_texts(embeddings, threshold=0.70, min_cluster_size=1)

    for group_idx, idx_list in enumerate(clusters):
        for idx in idx_list:
            gaps[idx]["group_id"] = group_idx


def _short_label(paper) -> str:
    if paper.title:
        words = paper.title.split()
        return " ".join(words[:4]) + ("..." if len(words) > 4 else "")
    return paper.source[:30]


def _summarize_question(text: str, max_len: int = 150) -> str:
    """Generate a summary label for an open question."""
    sentences = extract_sentences(text, min_length=15)
    if not sentences:
        return "Unlabeled question"
    # Prefer the shortest meaningful sentence
    candidates = [s for s in sentences if 30 < len(s) < max_len]
    return candidates[0] if candidates else sentences[0][:max_len]
