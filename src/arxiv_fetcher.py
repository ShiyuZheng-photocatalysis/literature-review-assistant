"""ArXiv API client for fetching paper PDFs and metadata.

Uses the official arXiv API (no external dependencies beyond stdlib).
"""

import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path
from typing import Optional

ARXIV_API_URL = "https://export.arxiv.org/api/query"


def parse_arxiv_id(source: str) -> Optional[str]:
    """Extract arXiv ID from various input formats.

    Handles:
    - Direct IDs: "2301.12345", "hep-th/0302184"
    - URLs: "https://arxiv.org/abs/2301.12345"
    - PDF URLs: "https://arxiv.org/pdf/2301.12345.pdf"
    - "arXiv:2301.12345"
    """
    # URL patterns
    url_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/\s?#]+)", source)
    if url_match:
        return url_match.group(1).replace(".pdf", "")

    # arXiv: prefix
    prefix_match = re.search(r"arXiv:(\S+)", source)
    if prefix_match:
        return prefix_match.group(1).strip()

    # Direct ID (e.g., 2301.12345 or hep-th/0302184)
    direct_match = re.match(r"^(\d{4}\.\d{4,5}(?:v\d+)?|[\w-]+/\d{7}(?:v\d+)?)$", source.strip())
    if direct_match:
        return direct_match.group(1)

    return None


def fetch_paper_by_id(arxiv_id: str, download_pdf: bool = True) -> dict:
    """Fetch paper metadata and optionally PDF from arXiv.

    Returns dict with keys: title, authors, year, abstract, pdf_path,
    arxiv_id, doi, categories, published.
    """
    clean_id = arxiv_id.replace("arxiv:", "").strip()
    if "v" in clean_id:
        clean_id = clean_id.rsplit("v", 1)[0]

    # Query arXiv API
    params = urllib.parse.urlencode({
        "id_list": clean_id,
        "max_results": 1,
    })
    url = f"{ARXIV_API_URL}?{params}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LiteratureReviewAssistant/1.0"}
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read().decode("utf-8")

    # Parse Atom XML
    root = ET.fromstring(xml_data)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    entry = root.find(".//atom:entry", ns)
    if entry is None:
        raise ValueError(f"No results for arXiv ID: {clean_id}")

    # Extract metadata
    title = _get_text(entry, "atom:title", ns)
    abstract = _get_text(entry, "atom:summary", ns)
    published = _get_text(entry, "atom:published", ns)
    doi = _get_arxiv_doi(entry, ns)

    # Authors
    authors = []
    for author_elem in entry.findall("atom:author", ns):
        name = _get_text(author_elem, "atom:name", ns)
        if name:
            authors.append(name)

    # Categories
    categories = []
    for cat in entry.findall("atom:category", ns):
        term = cat.get("term", "")
        if term:
            categories.append(term)

    # Links
    pdf_url = ""
    abs_url = ""
    for link in entry.findall("atom:link", ns):
        href = link.get("href", "")
        if link.get("title") == "pdf":
            pdf_url = href
        elif link.get("rel") == "alternate":
            abs_url = href

    # Year
    year = None
    if published:
        year_match = re.match(r"(\d{4})", published)
        if year_match:
            year = int(year_match.group(1))

    # Clean title (remove newlines and extra whitespace)
    title = re.sub(r"\s+", " ", title).strip() if title else ""

    metadata = {
        "arxiv_id": clean_id,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract or "",
        "categories": categories,
        "published": published or "",
        "doi": doi,
        "pdf_path": None,
        "pdf_url": pdf_url,
    }

    if download_pdf and pdf_url:
        try:
            pdf_path = _download_pdf(pdf_url, clean_id)
            metadata["pdf_path"] = str(pdf_path)
        except Exception:
            metadata["pdf_path"] = None

    return metadata


def fetch_multiple_papers(arxiv_ids: list[str], download_pdf: bool = True,
                          progress_callback=None) -> list[dict]:
    """Fetch multiple papers from arXiv.

    Args:
        arxiv_ids: List of arXiv IDs or URLs.
        download_pdf: Whether to download PDFs.
        progress_callback: Optional callable(completed, total) for progress.

    Returns list of metadata dicts (same format as fetch_paper_by_id).
    """
    results = []
    total = len(arxiv_ids)

    for i, source in enumerate(arxiv_ids):
        aid = parse_arxiv_id(source)
        if not aid:
            results.append({"error": f"Could not parse arXiv ID from: {source}"})
            continue
        try:
            meta = fetch_paper_by_id(aid, download_pdf=download_pdf)
            results.append(meta)
        except Exception as e:
            results.append({"error": str(e), "arxiv_id": aid})

        if progress_callback:
            progress_callback(i + 1, total)

    return results


def _get_text(element: ET.Element, tag: str, ns: dict) -> str:
    """Get text content of a child element."""
    child = element.find(tag, ns)
    return child.text.strip() if child is not None and child.text else ""


def _get_arxiv_doi(entry: ET.Element, ns: dict) -> str:
    """Extract DOI from arXiv entry."""
    for link in entry.findall("atom:link", ns):
        href = link.get("href", "")
        if "doi.org" in href:
            return href.split("doi.org/")[-1]
    # Also check in the arXiv-specific namespace
    doi_elem = entry.find("arxiv:doi", ns)
    if doi_elem is not None and doi_elem.text:
        return doi_elem.text.strip()
    return ""


def _download_pdf(pdf_url: str, arxiv_id: str) -> Path:
    """Download PDF from arXiv to a temporary file."""
    req = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": "LiteratureReviewAssistant/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        pdf_data = resp.read()

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{arxiv_id.replace('/', '_')}.pdf",
        delete=False,
    )
    tmp.write(pdf_data)
    tmp.close()
    return Path(tmp.name)
