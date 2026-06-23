"""Text cleaning, caching, and parallel processing utilities."""

import re
import json
import hashlib
import pickle
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


# ── Text cleaning ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    if not text:
        return ""
    # Fix hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Normalize whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable characters except newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove repeated spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_sentence(sent: str) -> str:
    """Normalize a sentence for comparison (lowercase, strip punctuation)."""
    sent = sent.lower().strip()
    sent = re.sub(r"\s+", " ", sent)
    sent = re.sub(r"^\W+|\W+$", "", sent)
    return sent


def extract_sentences(text: str, min_length: int = 10) -> list[str]:
    """Split text into sentences, filtering very short ones."""
    if not text:
        return []
    text = text.replace("\n", " ")
    # Split on sentence-ending punctuation
    raw = re.split(r'(?<=[.!?。！？])\s+', text)
    sentences = []
    for s in raw:
        # Further split Chinese sentences
        parts = re.split(r'(?<=[。！？])', s)
        for p in parts:
            clean = p.strip()
            if len(clean) >= min_length:
                sentences.append(clean)
    return sentences


# ── Caching ────────────────────────────────────────────────────────────────

class CacheManager:
    """Simple file-based cache for processed papers and analysis results."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prefix: str, identifier: str) -> str:
        h = hashlib.md5(identifier.encode()).hexdigest()[:12]
        return f"{prefix}_{h}"

    def get(self, prefix: str, identifier: str) -> Optional[dict]:
        path = self.cache_dir / f"{self._key(prefix, identifier)}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
        return None

    def put(self, prefix: str, identifier: str, data: dict) -> None:
        path = self.cache_dir / f"{self._key(prefix, identifier)}.json"
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )

    def get_binary(self, prefix: str, identifier: str) -> Optional[bytes]:
        path = self.cache_dir / f"{self._key(prefix, identifier)}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        return None

    def put_binary(self, prefix: str, identifier: str, data) -> None:
        path = self.cache_dir / f"{self._key(prefix, identifier)}.pkl"
        path.write_bytes(pickle.dumps(data))

    def clear(self) -> None:
        for f in self.cache_dir.glob("*"):
            f.unlink()


# ── Parallel processing ────────────────────────────────────────────────────

def parallel_map(func, items: list, max_workers: int = 4, use_threads: bool = True) -> list:
    """Map a function over items in parallel.

    Args:
        func: Function to apply to each item.
        items: List of items.
        max_workers: Maximum number of parallel workers.
        use_threads: Use ThreadPoolExecutor if True, ProcessPoolExecutor if False.

    Returns list of results in the same order as items.
    """
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    results = [None] * len(items)

    with Executor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e

    return results


def batch_items(items: list, batch_size: int) -> list[list]:
    """Split items into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# ── Text statistics ────────────────────────────────────────────────────────

def text_stats(text: str) -> dict:
    """Compute basic statistics for a text."""
    if not text:
        return {"chars": 0, "words": 0, "sentences": 0, "paragraphs": 0}
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    sentences = extract_sentences(text)
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
    }
