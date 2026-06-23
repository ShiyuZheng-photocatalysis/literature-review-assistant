"""PDF text extraction and rule-based IMRaD section segmentation.

Uses PyMuPDF (fitz) when available, falls back to pdfplumber.
"""

import re
import io
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try PyMuPDF first, fall back to pdfplumber
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    try:
        import pdfplumber
        HAS_PDFPLUMBER = True
    except ImportError:
        HAS_PDFPLUMBER = False

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

FIG_CAPTION_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:Fig(?:ure)?[\.\s]+\d+[\.\s:]+)(.+?)(?=\n\s*(?:Fig(?:ure)?[\.\s]+\d+|Table[\.\s]+\d+|$))", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:^|\n)\s*(?:图\s*\d+[\.\s:：]+)(.+?)(?=\n\s*(?:图\s*\d+|表\s*\d+|$))", re.DOTALL),
]

TABLE_CAPTION_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:Table[\.\s]+\d+[\.\s:]+)(.+?)(?=\n\s*(?:Fig(?:ure)?[\.\s]+\d+|Table[\.\s]+\d+|$))", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:^|\n)\s*(?:表\s*\d+[\.\s:：]+)(.+?)(?=\n\s*(?:图\s*\d+|表\s*\d+|$))", re.DOTALL),
]

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
    index: int
    caption: str
    page: int
    bbox: tuple = (0, 0, 0, 0)
    image_bytes: Optional[bytes] = None
    discussion_paragraphs: list = field(default_factory=list)


@dataclass
class Paper:
    source: str = ""
    title: str = ""
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    doi: str = ""
    abstract: str = ""
    introduction: str = ""
    related_work: str = ""
    methods: str = ""
    results: str = ""
    discussion: str = ""
    conclusion: str = ""
    full_text: str = ""
    figures: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    references_text: str = ""
    text_hash: str = ""
    language: str = ""

    def all_section_text(self) -> str:
        sections = [
            self.abstract, self.introduction, self.related_work,
            self.methods, self.results, self.discussion, self.conclusion
        ]
        return "\n\n".join(s for s in sections if s)

    def get_section(self, name: str) -> str:
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
    """Extract and segment a paper from a PDF file."""
    if HAS_FITZ:
        return _extract_with_fitz(filepath_or_bytes, source_name)
    elif HAS_PDFPLUMBER:
        return _extract_with_pdfplumber(filepath_or_bytes, source_name)
    else:
        raise ImportError(
            "Neither PyMuPDF nor pdfplumber is installed. "
            "Install one: pip install PyMuPDF  OR  pip install pdfplumber"
        )


def _extract_with_fitz(filepath_or_bytes, source_name: str = "") -> Paper:
    """Extract using PyMuPDF (fast, supports images)."""
    if isinstance(filepath_or_bytes, (str, Path)):
        doc = fitz.open(str(filepath_or_bytes))
    else:
        doc = fitz.open(stream=filepath_or_bytes, filetype="pdf")

    paper = Paper(source=source_name)
    all_lines = []
    figure_infos = []

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                text = "".join([span["text"] for span in line["spans"]])
                if text.strip():
                    font_sizes = [span["size"] for span in line["spans"] if span["text"].strip()]
                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10
                    all_lines.append({
                        "text": text.strip(),
                        "font_size": avg_font_size,
                    })
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            if base_image and base_image.get("width", 0) > 100 and base_image.get("height", 0) > 100:
                figure_infos.append({
                    "page": page_idx,
                    "image_bytes": base_image["image"],
                    "xref": xref,
                })
    doc.close()

    paper.full_text = "\n".join([l["text"] for l in all_lines])
    paper.text_hash = hashlib.md5(paper.full_text.encode()).hexdigest()
    paper.language = _detect_language(paper.full_text)
    _segment_sections(paper, all_lines)
    _extract_figure_captions(paper, figure_infos)
    _extract_metadata(paper)
    return paper


def _extract_with_pdfplumber(filepath_or_bytes, source_name: str = "") -> Paper:
    """Extract using pdfplumber (pure Python, always works)."""
    if isinstance(filepath_or_bytes, (str, Path)):
        pdf = pdfplumber.open(str(filepath_or_bytes))
    else:
        pdf = pdfplumber.open(io.BytesIO(filepath_or_bytes))

    paper = Paper(source=source_name)
    all_text_parts = []

    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text_parts.append(text)

    pdf.close()

    paper.full_text = "\n".join(all_text_parts)
    paper.text_hash = hashlib.md5(paper.full_text.encode()).hexdigest()
    paper.language = _detect_language(paper.full_text)

    # Build simple lines for section segmentation
    simple_lines = [{"text": line.strip(), "font_size": 10}
                    for line in paper.full_text.split("\n") if line.strip()]
    _segment_sections(paper, simple_lines)
    _extract_figure_captions(paper, [])
    _extract_metadata(paper)
    return paper


def _detect_language(text: str) -> str:
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
    boundaries = []
    for i, line in enumerate(lines):
        text = line["text"].strip()
        if len(text) > 50:
            continue
        for pattern, section_name in SECTION_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                boundaries.append((i, section_name, text))
                break

    if len(boundaries) < 2:
        boundaries = _find_headings_by_font(lines)

    if not boundaries:
        paper.introduction = paper.full_text
        return

    section_texts = _collect_section_texts(lines, boundaries)
    for name, text in section_texts.items():
        _set_section(paper, name, text)
    _fill_missing_sections(paper, lines)


def _find_headings_by_font(lines: list) -> list:
    if not lines:
        return []
    font_sizes = [l["font_size"] for l in lines]
    body_font = sorted(set(font_sizes))[len(set(font_sizes)) // 2]
    boundaries = []
    for i, line in enumerate(lines):
        text = line["text"].strip()
        if len(text) > 60:
            continue
        if line["font_size"] > body_font * 1.05:
            for pattern, section_name in SECTION_PATTERNS:
                if re.match(pattern, text, re.IGNORECASE):
                    boundaries.append((i, section_name, text))
                    break
    return boundaries


def _collect_section_texts(lines: list, boundaries: list) -> dict:
    section_texts = {}
    boundaries.append((len(lines), "END", ""))
    for idx in range(len(boundaries) - 1):
        start, name, _ = boundaries[idx]
        end, _, _ = boundaries[idx + 1]
        section_lines = []
        for j in range(start + 1, end):
            text = lines[j]["text"].strip()
            if text:
                section_lines.append(text)
        if section_lines:
            section_texts[name] = "\n".join(section_lines)
    return section_texts


def _fill_missing_sections(paper: Paper, lines: list) -> None:
    text = paper.full_text
    paragraphs = text.split("\n\n")
    method_kw = ["method", "approach", "algorithm", "implementation", "setup", "protocol", "procedure"]
    result_kw = ["result", "finding", "observation", "performance", "accuracy", "precision", "demonstrate", "show"]
    discussion_kw = ["discuss", "implication", "interpret", "explain", "mechanism", "insight", "suggest"]
    conclusion_kw = ["conclusion", "summary", "conclude", "summarize", "future work", "outlook", "perspective"]

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

    for section_name in ["methods", "results", "discussion", "conclusion"]:
        if paper.get_section(section_name):
            continue
        best_paras = sorted(para_scores, key=lambda x: x[1].get(section_name, 0), reverse=True)
        top_paras = [paragraphs[p[0]] for p in best_paras[:5] if p[1].get(section_name, 0) > 0]
        if top_paras:
            _set_section(paper, section_name, "\n\n".join(top_paras))


def _extract_figure_captions(paper: Paper, figure_infos: list) -> None:
    text = paper.full_text
    captions = []
    for pattern in FIG_CAPTION_PATTERNS:
        for match in pattern.finditer(text):
            captions.append(match.group(1).strip())

    captions = captions[:len(figure_infos)] if figure_infos else captions

    for i, caption in enumerate(captions):
        fi = figure_infos[i] if i < len(figure_infos) else {}
        paper.figures.append(FigureInfo(
            index=i + 1,
            caption=caption,
            page=fi.get("page", 0),
            image_bytes=fi.get("image_bytes"),
        ))

    if not captions and figure_infos:
        for i, fi in enumerate(figure_infos):
            paper.figures.append(FigureInfo(
                index=i + 1,
                caption=f"Figure {i + 1}",
                page=fi.get("page", 0),
                image_bytes=fi.get("image_bytes"),
            ))

    for pattern in TABLE_CAPTION_PATTERNS:
        for match in pattern.finditer(text):
            paper.tables.append(match.group(1).strip())

    _find_figure_discussions(paper, text)


def _find_figure_discussions(paper: Paper, text: str) -> None:
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
    text = paper.full_text
    lines = text.split("\n")[:30]
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 20 and not stripped.startswith(("http", "doi:", "DOI:", "arXiv:")):
            paper.title = stripped
            break
    doi_match = re.search(r"(?:DOI|doi):\s*(\S+)", text[:2000])
    if doi_match:
        paper.doi = doi_match.group(1)
    arxiv_match = re.search(r"arXiv:(\d+\.\d+)", text[:2000])
    if arxiv_match and not paper.source.startswith("10."):
        paper.source = f"arXiv:{arxiv_match.group(1)}"
    year_match = re.search(r"(?:©|Copyright|Published).*?(20\d{2})", text[:2000])
    if year_match:
        paper.year = int(year_match.group(1))


def extract_problem_statements(paper: Paper) -> list[dict]:
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
                    problems.append({"sentence": sent.strip(), "section": section_name, "keyword": kw})
                    break
    seen = set()
    unique = []
    for p in problems:
        key = p["sentence"][:80].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def extract_open_questions(paper: Paper) -> list[str]:
    future_patterns = [
        r"(?:future\s*work|further\s*(?:investigation|study|research)|remains?\s*to\s*be|warrants?\s*further|open\s*question|outstanding\s*question)",
        r"(?:有待|尚待|仍需|需进一步|值得进一步|尚未|尚不|仍不|未来|展望)",
    ]
    questions = []
    for section_text in [paper.discussion, paper.conclusion, paper.introduction]:
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
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    result = []
    for s in sentences:
        parts = re.split(r'(?<=[。！？])', s)
        result.extend(p.strip() for p in parts if p.strip() and len(p.strip()) > 5)
    return result


def _set_section(paper: Paper, name: str, text: str) -> None:
    mapping = {
        "abstract": "abstract", "introduction": "introduction",
        "related_work": "related_work", "methods": "methods",
        "results": "results", "discussion": "discussion",
        "conclusion": "conclusion",
    }
    if name in mapping:
        setattr(paper, mapping[name], text)
