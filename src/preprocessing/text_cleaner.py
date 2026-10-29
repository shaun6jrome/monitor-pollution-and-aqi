"""Text cleaning and normalization pipeline for unstructured environmental news.

Normalizes Unicode, collapses whitespace, strips unwanted markup, and removes boilerplate
while strictly preserving chemical formulas, pollutant names, locations, and numerical measurements.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Optional

# Regex for stripping standalone URLs and emails
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Boilerplate patterns common in syndicated news feeds
BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)\b(all rights reserved|read more at|click here to subscribe|terms of service|privacy policy)\b.*"),
    re.compile(r"(?i)\b(first published|source: [a-z0-9\.\s]+)\b.*$"),
]


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters using NFKC decomposition and unescape HTML entities."""
    if not text:
        return ""
    unescaped = html.unescape(text)
    # Normalize curly quotes and dashes
    unescaped = (
        unescaped.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return unicodedata.normalize("NFKC", unescaped)


def clean_text(text: Optional[str]) -> str:
    """Clean raw unstructured text while retaining technical environmental terms."""
    if not text or not isinstance(text, str):
        return ""

    normalized = normalize_unicode(text)

    # Remove URLs and email addresses
    cleaned = URL_REGEX.sub(" ", normalized)
    cleaned = EMAIL_REGEX.sub(" ", cleaned)

    # Strip news boilerplate lines
    lines = cleaned.splitlines()
    filtered_lines = []
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        is_boilerplate = False
        for bp in BOILERPLATE_PATTERNS:
            if bp.search(line_clean):
                is_boilerplate = True
                break
        if not is_boilerplate:
            filtered_lines.append(line_clean)

    cleaned = "\n".join(filtered_lines)

    # Collapse excessive whitespace and tabs while preserving paragraph breaks
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def normalize_text(text: Optional[str]) -> str:
    """Generate a single-line lowercased normalized string suitable for TF-IDF / similarity matching."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    # Replace newlines with spaces and lowercase
    single_line = re.sub(r"\s+", " ", cleaned).lower()
    return single_line.strip()
