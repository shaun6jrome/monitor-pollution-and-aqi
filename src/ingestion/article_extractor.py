"""Article extraction utility for public news pages.

Extracts readable textual content from article web pages using standard HTTP
requests and BeautifulSoup markup parsing. Respects timeouts, response size limits,
and paywalls by recording extraction status and falling back cleanly to metadata.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36 (DSE-Pollution-Research-Bot/1.0)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags to completely remove from consideration
STRIP_TAGS = [
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    "svg",
    "figure",
    "button",
]


class ArticleExtractor:
    """Extracts clean article text from permitted public web pages."""

    def __init__(self, timeout: int = 10, max_bytes: int = 1_000_000):
        self.timeout = timeout
        self.max_bytes = max_bytes

    def extract_from_html(self, html_content: str, url: str = "") -> Tuple[str, str]:
        """Parse raw HTML string and extract main body text.

        Returns (extracted_text, extraction_status).
        Status values: 'full_text', 'partial_text', 'title_only'.
        """
        if not html_content or not html_content.strip():
            return "", "title_only"

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove navigation, styles, scripts, and non-article boilerplate
            for tag in soup(STRIP_TAGS):
                tag.decompose()

            # Target primary article containers if available
            container = (
                soup.find("article")
                or soup.find("main")
                or soup.find(class_=re.compile(r"(article[-_]?body|story[-_]?content|post[-_]?body|entry[-_]?content)", re.I))
                or soup.find("body")
            )

            if not container:
                return "", "title_only"

            paragraphs = container.find_all("p")
            clean_paras = []
            for p in paragraphs:
                text = p.get_text(separator=" ", strip=True)
                # Filter out cookie disclaimers, share links, and trivial text
                if len(text) > 30 and not re.search(r"cookie|subscribe|sign up|rights reserved|advertisement", text, re.I):
                    clean_paras.append(text)

            body_text = "\n\n".join(clean_paras).strip()

            if len(body_text) >= 200:
                return body_text, "full_text"
            elif len(body_text) >= 60:
                return body_text, "partial_text"
            else:
                return "", "title_only"

        except Exception as e:
            logger.warning("HTML parsing failed for %s: %s", url, e)
            return "", "extraction_failed"

    def fetch_and_extract(self, url: str) -> Dict[str, str]:
        """Fetch URL content via HTTP and extract readable body text.

        Returns a dictionary with 'content_raw' and 'extraction_status'.
        """
        if not url or not url.startswith(("http://", "https://")):
            return {"content_raw": "", "extraction_status": "invalid_url"}

        parsed = urlparse(url)
        if not parsed.netloc:
            return {"content_raw": "", "extraction_status": "invalid_url"}

        try:
            response = requests.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
                stream=True,
                allow_redirects=True,
            )

            if response.status_code == 403 or response.status_code == 401:
                return {"content_raw": "", "extraction_status": "paywalled_or_restricted"}
            elif response.status_code != 200:
                return {"content_raw": "", "extraction_status": f"http_{response.status_code}"}

            # Check content length and type
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return {"content_raw": "", "extraction_status": "non_html_content"}

            raw_bytes = response.raw.read(self.max_bytes, decode_content=True)
            # Detect encoding or fallback to utf-8 with replacement
            encoding = response.encoding or "utf-8"
            html_content = raw_bytes.decode(encoding, errors="replace")

            text, status = self.extract_from_html(html_content, url=url)
            return {"content_raw": text, "extraction_status": status}

        except requests.Timeout:
            return {"content_raw": "", "extraction_status": "timeout"}
        except requests.RequestException as e:
            logger.info("Extraction request failed for %s: %s", url, e)
            return {"content_raw": "", "extraction_status": "request_failed"}
        except Exception as e:
            logger.error("Unexpected error extracting %s: %s", url, e)
            return {"content_raw": "", "extraction_status": "extraction_error"}
