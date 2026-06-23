"""PDF text extraction and rule-based IMRaD section segmentation."""

import re
import io
import hashlib
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Section heading patterns (English + Chinese) ──────────────────────────

SECTION_PATTERNS = [
    # English numbered headings
    (r"^\s*(?:\d+[\.\)]\s*)?(?:I{1,3}[\.\)]\s*)?(?:Abstract)\s*$", "abstract"),
    (r"^\s*(?:\d+[\.\)]\s*)?(?:I{1,3}[\.\)]\s*)?(?:Introduction|Intro\.)\s*$", "introduction"),
    (r"^\s*(?:\d+[\.\)]\s*)?(?:I{1,3}[\.\)]\s*)?(?:Related\s*Work|Background|Literature\s*Review|Prior\s*Work)\s*$", "related_work"),
    (r"^\s*(?:\d+[\.\)]\s*)?(?:I{1,3}[\.\)]\s*)?(?:Method(?:ology|s)?|Approach|Experimental\s*(?:Design|Setup|Section)|Computational\s*Details|Theory|Formulation|Model)\s*$", "methods"),
    (r"^\s*(?:\d+[\.\)]\s*)?(?:I{1,3}[\.\)]\s*)?(?:Results?(?:\s*and\s*Discussion)?|Experiments?(?:\s*and\s*Results?)?)\s*$", "results"),
    (r"^\s*(?:\d+[\.\)]\s*)?(?:I{1,3}[\.\)]\s*)?(?:Discussion|Analysis)\s*$", "discussion"),
    (r"^\s*(?:\d+[\.\)]\s*)?(?:I{1,3}[\.\)]\s*)?(?:Conclusion|Summary|Concluding\s*Remarks)\s*$", "conclusion"),
    # Chinese headings
    (r"^\s*(?:\d+[\.\)]\s*)?摘要\s*$", "abstract"),
    (r"^\s*(?:\d+[\.\)]\s*)?引言|绪论|前言\s*$", "introduction"),
    (r"^\s*(?:\d+[\.\)]\s*)?相关工作|文献综述|研究背景\s*$", "related_work"),
    (r"^\s*(?:\d+[\.\)]\s*)?方法|实验方法|计算方法|理论|模型\s*$", "methods"),
    (r"^\s*(?:\d+[\.\)]\s*)?结果|实验结果|结果与讨论\s*$", "results"),
    (r"^\s*(?:\d+[\.\)]\s*)?讨论|分析\s*$", "discussion"),
    (r"^\s*(?:\d+[\.\)]\s*)?结论|总结\s*$", "conclusion"),
]

# Figure/table caption patterns
FIG_CAPTION_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:Fig(?:ure)?[\.\s]+\d+[\.\s:]+)(.+?)(?=\n\s*(?:Fig(?:ure)?[\.\s]+\d+|Table[\.\s]+\d+|$))", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:^|\n)\s*(?:图\s*\d+[\.\s:：]+)(.+?)(?=\n\s*(?:图\s*\d+|表\s*\d+|$))", re.DOTALL),
]

TABLE_CAPTION_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:Table[\.\s]+\d+[\.\s:]+)(.+?)(?=\n\s*(?:Fig(?:ure)?[\.\s]+\d+|Table[\.\s]+\d+|$))", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:^|\n)\s*(?:表\s*\d+[\.\s:：]+)(.+?)(?=\n\s*(?:图\s*\d+|表\s*\d+|$))", re.DOTALL),
]

# Problem-indicating keywords
PROBLEM_KEYWORDS_EN = [
    "challenge", "remains unclear", "open question", "limitation",
    "however", "nevertheless", "remains elusive", "poorly understood",
    "needs further", "warrants further", "remains to be", "yet to be",
    "unsolved", "unresolved", "bottleneck", "obstacle", "drawback",
    "shortcoming", "gap", "remains a challenge", "remains challenging",
    "not well understood", "little is known", "lack of", "insufficient",
    "has not been", "have not been", "no systematic", "few studies",
]

PROBLEM_KEYWORDS_ZH = [
    "挑战", "尚不清楚", "尚未解决", "有待", "仍不清楚",
    "瓶颈", "不足", "缺陷", "困难", "难题", "障碍",
    "鲜有", "缺乏", "尚无", "未解决", "尚未明确",
    "有待进一步", "尚需", "亟待", "亟需",
]


@dataclass
class FigureInfo:
    """Extracted figure metadata."""
    index: int
    caption: str
    page: int
    bbox: tuple  # (x0, y0, x1, y1) on page
    image_bytes: Optional[bytes] = None
    discussion_paragraphs: list = field(default_factory=list)


@dataclass
class Paper:
    """Structured representation of an academic paper."""
    source: str  # file path or arXiv ID
    title: str = ""
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    doi: str = ""
    # Section contents
    abstract: str = ""
    introduction: str = ""
    related_work: str = ""
    methods: str = ""
    results: str = ""
    discussion: str = ""
    conclusion: str = ""
    # Full text and metadata
    full_text: str = ""
    figures: list = field(default_factory=list)  # List[FigureInfo]
    tables: list = field(default_factory=list)  # List of captions
    references_text: str = ""
    # Computed
    text_hash: str = ""
    language: str = ""  # "en", "zh", "mixed"

    def all_section_text(self) -> str:
        """Concatenate all section text for embedding."""
        sections = [
            self.abstract, self.introduction, self.related_work,
            self.methods, self.results, self.discussion, self.conclusion
        ]
        return "\n\n".join(s for s in sections if s)

    def get_section(self, name: str) -> str:
        """Get a named section."""
        mapping = {
            "abstract": self.abstract,
            "introduction": self.introduction,
            "related_work": self.related_work,
            "methods": self.methods,
            "results": self.results,
            "discussion": self.discussion,
            "conclusion": self.conclusion,
        }
        return mapping.get(name, "")


def extract_paper_from_pdf(filepath_or_bytes, source_name: str = "") -> Paper:
    """Extract and segment a paper from a PDF file.

    Args:
        filepath_or_bytes: Path to PDF file, or bytes of PDF content.
        source_name: Identifier for the paper source.

    Returns:
        Paper object with segmented sections.
    """
    if isinstance(filepath_or_bytes, (str, Path)):
        doc = fitz.open(str(filepath_or_bytes))
    else:
        doc = fitz.open(stream=filepath_or_bytes, filetype="pdf")

    paper = Paper(source=source_name)
    all_lines = []
    figure_infos = []

    for page_idx, page in enumerate(doc):
        # Extract text blocks with position info
        blocks = page.get_text("dict")["blocks"]
        page_text_lines = []

        for block in blocks:
            if block["type"] != 0:  # Not text
                continue
            for line in block["lines"]:
                text = "".join([span["text"] for span in line["spans"]])
                if text.strip():
                    font_sizes = [span["size"] for span in line["spans"] if span["text"].strip()]
                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10
                    page_text_lines.append({
                        "text": text.strip(),
                        "font_size": avg_font_size,
                        "bbox": line["bbox"],
                    })

        # Extract images from page
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            if base_image and base_image.get("width", 0) > 100 and base_image.get("height", 0) > 100:
                figure_infos.append({
                    "page": page_idx,
                    "image_bytes": base_image["image"],
                    "bbox": (0, 0, base_image["width"], base_image["height"]),
                    "xref": xref,
                })

        all_lines.extend(page_text_lines)

    doc.close()

    # Concatenate full text
    paper.full_text = "\n".join([l["text"] for l in all_lines])
    paper.text_hash = hashlib.md5(paper.full_text.encode()).hexdigest()

    # Detect language
    paper.language = _detect_language(paper.full_text)

    # Segment into sections
    _segment_sections(paper, all_lines)

    # Extract figure captions
    _extract_figure_captions(paper, figure_infos)

    # Extract metadata
    _extract_metadata(paper)

    return paper


def _detect_language(text: str) -> str:
    """Detect whether text is English, Chinese, or mixed."""
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    total_chars = len(re.sub(r"\s", "", text))
    if total_chars == 0:
        return "en"
    ratio = chinese_chars / total_chars
    if ratio > 0.5:
        return "zh"
    elif ratio > 0.2:
        return "mixed"
    return "en"


def _segment_sections(paper: Paper, lines: list) -> None:
    """Segment paper text into IMRaD sections using heading patterns."""
    # Find section boundaries by matching heading patterns
    boundaries = []  # List of (line_index, section_name, line_text)

    for i, line in enumerate(lines):
        text = line["text"].strip()
        # Only consider lines with larger font or short text as potential headings
        if len(text) > 50:
            continue
        for pattern, section_name in SECTION_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                boundaries.append((i, section_name, text))
                break

    # If no structured headings found, try font-size based approach
    if len(boundaries) < 2:
        boundaries = _find_headings_by_font(lines)

    # Map boundaries to sections
    if not boundaries:
        # No structure detected; put everything in the paper as-is
        paper.introduction = paper.full_text
        return

    section_texts = _collect_section_texts(lines, boundaries)
    for name, text in section_texts.items():
        _set_section(paper, name, text)

    # If some sections are empty, try to infer from keyword-based splitting
    _fill_missing_sections(paper, lines)


def _find_headings_by_font(lines: list) -> list:
    """Use font-size heuristics to find section headings."""
    if not lines:
        return []

    font_sizes = [l["font_size"] for l in lines]
    max_font = max(font_sizes)
    # A heading is a short line with font size notably larger than body text
    body_font = sorted(set(font_sizes))[len(set(font_sizes)) // 2]

    boundaries = []
    for i, line in enumerate(lines):
        text = line["text"].strip()
        if len(text) > 60:
            continue
        if line["font_size"] > body_font * 1.05:
            # Try to classify this heading
            for pattern, section_name in SECTION_PATTERNS:
                if re.match(pattern, text, re.IGNORECASE):
                    boundaries.append((i, section_name, text))
                    break
    return boundaries


def _collect_section_texts(lines: list, boundaries: list) -> dict:
    """Collect text for each section given boundary positions."""
    section_texts = {}
    boundaries.append((len(lines), "END", ""))

    for idx in range(len(boundaries) - 1):
        start, name, _ = boundaries[idx]
        end, _, _ = boundaries[idx + 1]
        # Collect lines belonging to this section
        section_lines = []
        for j in range(start + 1, end):
            text = lines[j]["text"].strip()
            if text:
                section_lines.append(text)
        if section_lines:
            if name in section_texts:
                section_texts[name] += "\n" + "\n".join(section_lines)
            else:
                section_texts[name] = "\n".join(section_lines)

    return section_texts


def _fill_missing_sections(paper: Paper, lines: list) -> None:
    """Try keyword-based inference for missing sections."""
    text = paper.full_text
    paragraphs = text.split("\n\n")

    # Simple keyword-based fallback
    method_kw = [
        "method", "approach", "algorithm", "implementation",
        "setup", "protocol", "procedure"
    ]
    result_kw = [
        "result", "finding", "observation", "performance",
        "accuracy", "precision", "demonstrate", "show"
    ]
    discussion_kw = [
        "discuss", "implication", "interpret", "explain",
        "mechanism", "insight", "suggest"
    ]
    conclusion_kw = [
        "conclusion", "summary", "conclude", "summarize",
        "future work", "outlook", "perspective"
    ]

    para_scores = []
    for i, para in enumerate(paragraphs):
        para_lower = para.lower()
        scores = {
            "methods": sum(1 for kw in method_kw if kw in para_lower),
            "results": sum(1 for kw in result_kw if kw in para_lower),
            "discussion": sum(1 for kw in discussion_kw if kw in para_lower),
            "conclusion": sum(1 for kw in conclusion_kw if kw in para_lower),
        }
        para_scores.append((i, scores))

    # Assign paragraphs to best-matching empty sections
    for section_name in ["methods", "results", "discussion", "conclusion"]:
        if _get_section(paper, section_name):
            continue
        best_paras = sorted(para_scores, key=lambda x: x[1].get(section_name, 0), reverse=True)
        top_paras = [paragraphs[p[0]] for p in best_paras[:5] if p[1].get(section_name, 0) > 0]
        if top_paras:
            _set_section(paper, section_name, "\n\n".join(top_paras))


def _extract_figure_captions(paper: Paper, figure_infos: list) -> None:
    """Extract figure captions and match them to extracted images."""
    text = paper.full_text

    # Extract captions
    captions = []
    for pattern in FIG_CAPTION_PATTERNS:
        for match in pattern.finditer(text):
            captions.append(match.group(1).strip())

    captions = captions[:len(figure_infos)] if figure_infos else captions

    # Match captions to images (simple index-based matching)
    for i, caption in enumerate(captions):
        fig_info = figure_infos[i] if i < len(figure_infos) else {"page": 0, "image_bytes": None, "bbox": (0, 0, 0, 0)}
        paper.figures.append(FigureInfo(
            index=i + 1,
            caption=caption,
            page=fig_info["page"],
            bbox=fig_info.get("bbox", (0, 0, 0, 0)),
            image_bytes=fig_info.get("image_bytes"),
        ))

    # If no captions found, still add figures from images
    if not captions and figure_infos:
        for i, fi in enumerate(figure_infos):
            paper.figures.append(FigureInfo(
                index=i + 1,
                caption=f"Figure {i + 1}",
                page=fi["page"],
                bbox=fi.get("bbox", (0, 0, 0, 0)),
                image_bytes=fi.get("image_bytes"),
            ))

    # Also extract table captions
    for pattern in TABLE_CAPTION_PATTERNS:
        for match in pattern.finditer(text):
            paper.tables.append(match.group(1).strip())

    # Find discussion paragraphs that reference each figure
    _find_figure_discussions(paper, text)


def _find_figure_discussions(paper: Paper, text: str) -> None:
    """Find paragraphs that discuss each figure."""
    paragraphs = text.split("\n\n")
    for fig in paper.figures:
        fig_refs = [
            f"Figure {fig.index}", f"Fig. {fig.index}",
            f"Figure{fig.index}", f"Fig{fig.index}",
            f"图{fig.index}", f"图 {fig.index}",
        ]
        for para in paragraphs:
            if any(ref in para for ref in fig_refs):
                fig.discussion_paragraphs.append(para.strip())


def _extract_metadata(paper: Paper) -> None:
    """Extract title, authors, DOI from the paper text."""
    text = paper.full_text
    lines = text.split("\n")[:30]  # Metadata is usually in the first 30 lines

    # Try to identify title (typically the first substantive line, larger font)
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 20 and not stripped.startswith(("http", "doi:", "DOI:", "arXiv:")):
            paper.title = stripped
            break

    # Look for DOI
    doi_match = re.search(r"(?:DOI|doi):\s*(\S+)", text[:2000])
    if doi_match:
        paper.doi = doi_match.group(1)

    # Look for arXiv ID
    arxiv_match = re.search(r"arXiv:(\d+\.\d+)", text[:2000])
    if arxiv_match and not paper.source.startswith("10."):
        paper.source = f"arXiv:{arxiv_match.group(1)}"

    # Year
    year_match = re.search(r"(?:©|Copyright|Published).*?(20\d{2})", text[:2000])
    if year_match:
        paper.year = int(year_match.group(1))


def _get_section(paper: Paper, name: str) -> str:
    mapping = {
        "abstract": paper.abstract,
        "introduction": paper.introduction,
        "related_work": paper.related_work,
        "methods": paper.methods,
        "results": paper.results,
        "discussion": paper.discussion,
        "conclusion": paper.conclusion,
    }
    return mapping.get(name, "")


def _set_section(paper: Paper, name: str, text: str) -> None:
    mapping = {
        "abstract": "abstract",
        "introduction": "introduction",
        "related_work": "related_work",
        "methods": "methods",
        "results": "results",
        "discussion": "discussion",
        "conclusion": "conclusion",
    }
    if name in mapping:
        setattr(paper, mapping[name], text)


def extract_problem_statements(paper: Paper) -> list[dict]:
    """Extract problem statements from a paper.

    Returns list of {sentence, section, confidence}.
    """
    problems = []
    # Check abstract, intro, discussion, conclusion
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

    return problems


def extract_open_questions(paper: Paper) -> list[str]:
    """Extract open questions and future work statements."""
    future_patterns = [
        r"(?:future\s*work|further\s*(?:investigation|study|research)|remains?\s*to\s*be|warrants?\s*further|open\s*question|outstanding\s*question)",
        r"(?:有待|尚待|仍需|需进一步|值得进一步|尚未|尚不|仍不|未来|展望)",
    ]

    questions = []
    sections_to_check = [
        paper.discussion,
        paper.conclusion,
        paper.introduction,
    ]

    for section_text in sections_to_check:
        if not section_text:
            continue
        sentences = _split_sentences(section_text)
        for sent in sentences:
            for pattern in future_patterns:
                if re.search(pattern, sent, re.IGNORECASE):
                    clean = sent.strip()
                    if len(clean) > 30 and clean not in questions:
                        questions.append(clean)
                    break

    return questions


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitting that handles both English and Chinese."""
    # Replace newlines
    text = text.replace("\n", " ")
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    # Also split Chinese sentences
    result = []
    for s in sentences:
        parts = re.split(r'(?<=[。！？])', s)
        result.extend(p.strip() for p in parts if p.strip() and len(p.strip()) > 5)
    return result
