"""Boilerplate / routine expression detection across papers.

Identifies near-identical or highly similar sentences that appear
in multiple papers, indicating templated writing patterns.
"""

from collections import defaultdict
from ..utils import extract_sentences, normalize_sentence
from ..embeddings import TfidfPhraseFinder, extract_shared_phrase


def find_boilerplate_phrases(papers: list, similarity_threshold: float = 0.65,
                             min_shared_length: int = 10) -> dict:
    """Find near-identical sentences and phrases shared across papers.

    Args:
        papers: List of Paper objects.
        similarity_threshold: TF-IDF cosine similarity threshold.
        min_shared_length: Minimum character length for shared phrases.

    Returns dict with:
        - by_section: dict of section_name -> list of phrase groups
        - summary: overall statistics
    """
    if len(papers) < 2:
        return {"by_section": {}, "summary": {"total_groups": 0, "total_pairs": 0}}

    sections = ["introduction", "related_work", "methods", "results", "discussion", "conclusion"]
    tfidf = TfidfPhraseFinder(char_ngram_range=(3, 6), max_features=8000)

    all_results = {}
    total_groups = 0
    total_pairs = 0

    for section_name in sections:
        # Gather all sentences from this section across all papers
        sent_paper_map = []  # List of (paper_idx, sentence)
        for i, paper in enumerate(papers):
            text = paper.get_section(section_name)
            if not text:
                continue
            sentences = extract_sentences(text, min_length=20)
            for sent in sentences:
                sent_paper_map.append((i, sent))

        if len(sent_paper_map) < 2:
            continue

        sentences = [s[1] for s in sent_paper_map]

        # Find similar pairs
        pairs = tfidf.find_similar_pairs(
            sentences,
            threshold=similarity_threshold,
            max_pairs=300,
        )

        if not pairs:
            continue

        # Group pairs into clusters (transitive grouping)
        groups = _group_similar_sentences(pairs, len(sentences))

        # Build output for this section
        section_groups = []
        for group in groups:
            # Extract the shared phrase
            texts_in_group = [sentences[idx] for idx in group]
            if len(set(texts_in_group)) < 2:
                continue

            # Find longest common substring across the group
            shared = _find_group_shared_phrase(texts_in_group, min_shared_length)

            papers_in_group = set()
            items = []
            for idx in group:
                paper_idx, sent = sent_paper_map[idx]
                papers_in_group.add(paper_idx)
                items.append({
                    "paper_idx": paper_idx,
                    "paper_label": _short_label(papers[paper_idx]),
                    "text": sent[:400],
                })

            if len(papers_in_group) >= 2:
                section_groups.append({
                    "shared_phrase": shared or "(similar wording)",
                    "paper_count": len(papers_in_group),
                    "sentence_count": len(items),
                    "items": items,
                })

        section_groups.sort(key=lambda g: g["paper_count"], reverse=True)
        all_results[section_name] = section_groups
        total_groups += len(section_groups)
        total_pairs += sum(g["sentence_count"] for g in section_groups)

    return {
        "by_section": all_results,
        "summary": {
            "total_groups": total_groups,
            "total_pairs": total_pairs,
        },
    }


def find_boilerplate_by_section(papers: list, section_name: str = "methods",
                                threshold: float = 0.65) -> list[dict]:
    """Convenience function to find boilerplate in a specific section."""
    result = find_boilerplate_phrases(papers, similarity_threshold=threshold)
    return result["by_section"].get(section_name, [])


def _group_similar_sentences(pairs: list[dict], n_sentences: int) -> list[list[int]]:
    """Group similar sentence indices into clusters using union-find."""
    parent = list(range(n_sentences))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in pairs:
        union(pair["idx_a"], pair["idx_b"])

    groups = defaultdict(list)
    for i in range(n_sentences):
        groups[find(i)].append(i)

    return [g for g in groups.values() if len(g) >= 2]


def _find_group_shared_phrase(texts: list[str], min_length: int) -> str:
    """Find the longest phrase shared by at least 2 texts in the group."""
    longest = ""
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            shared = extract_shared_phrase(texts[i], texts[j], min_length=min_length)
            if len(shared) > len(longest):
                longest = shared
    return longest


def _short_label(paper) -> str:
    if paper.title:
        words = paper.title.split()
        return " ".join(words[:4]) + ("..." if len(words) > 4 else "")
    return paper.source[:30]
