"""GDELT DOC 2.0 API Client for retrieving real-time unstructured pollution news.

Integrates with https://api.gdeltproject.org/api/v2/doc/doc to fetch live news
articles across configurable environmental queries, handles local caching,
rate limiting, timestamp parsing, and incremental database ingestion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import config
from src.ingestion.article_extractor import ArticleExtractor
from src.storage.repository import PollutionRepository

logger = logging.getLogger(__name__)


def parse_gdelt_timestamp(seendate_str: Optional[str]) -> Optional[str]:
    """Parse GDELT seendate string (e.g. '20241029T143000Z' or '20241029143000') to ISO 8601 UTC."""
    if not seendate_str:
        return None

    clean = seendate_str.strip().replace("T", "").replace("Z", "")
    try:
        if len(clean) >= 14:
            dt = datetime.strptime(clean[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            return dt.isoformat()
        elif len(clean) >= 8:
            dt = datetime.strptime(clean[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
            return dt.isoformat()
    except (ValueError, TypeError) as e:
        logger.debug("Failed parsing GDELT date '%s': %s", seendate_str, e)

    return None


class GDELTClient:
    """Client for querying GDELT DOC 2.0 API with caching and error handling."""

    def __init__(
        self,
        repository: Optional[PollutionRepository] = None,
        cache_dir: Optional[Path] = None,
        timeout: int = config.GDELT_REQUEST_TIMEOUT,
        cache_ttl: int = config.GDELT_CACHE_TTL_SECONDS,
    ):
        self.repository = repository or PollutionRepository()
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.extractor = ArticleExtractor(timeout=8)

    def _get_cache_path(self, query: str, max_records: int) -> Path:
        """Generate safe deterministic cache file path for a query."""
        key = f"{query}_{max_records}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"gdelt_{digest}.json"

    def fetch_articles(
        self,
        query: str,
        max_records: int = config.GDELT_MAX_RECORDS,
        force_refresh: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Fetch news articles from GDELT DOC 2.0 API or cache.

        Returns (articles_list, metadata_dict).
        Metadata contains status, source, record counts, and errors if any.
        """
        cache_path = self._get_cache_path(query, max_records)
        now_ts = datetime.now(timezone.utc).timestamp()

        # Check local cache first unless force_refresh is requested
        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                cached_time = cached_data.get("cached_at_ts", 0)
                if now_ts - cached_time < self.cache_ttl:
                    articles = cached_data.get("articles", [])
                    meta = {
                        "status": "CACHED",
                        "source": "local_cache",
                        "cache_file": str(cache_path),
                        "cached_at": cached_data.get("cached_at"),
                        "total_retrieved": len(articles),
                        "query": query,
                    }
                    return articles, meta
            except Exception as e:
                logger.warning("Error reading GDELT cache %s: %s", cache_path, e)

        # Build GDELT DOC 2.0 query
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": min(max_records, 250),
            "format": "json",
            "sort": "DateDesc",
        }

        headers = {
            "User-Agent": "PollutionIntelligencePlatform/1.0 (Academic Research; Air Quality Study)",
            "Accept": "application/json, text/plain, */*",
        }

        try:
            response = requests.get(
                config.GDELT_BASE_URL,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                error_msg = f"GDELT API returned HTTP {response.status_code}: {response.text[:120]}"
                logger.error(error_msg)
                self.repository.log_retrieval("GDELT", query, "FAILED", error_message=error_msg)
                return [], {
                    "status": "FAILED",
                    "source": "api_error",
                    "error": error_msg,
                    "total_retrieved": 0,
                    "query": query,
                }

            # Parse JSON response
            text = response.text.strip()
            # GDELT occasionally returns plain text message on empty match or rate limit
            if not text or not text.startswith("{"):
                meta = {
                    "status": "NO_DATA",
                    "source": "api",
                    "message": "GDELT returned no structured JSON (possibly 0 results for query)",
                    "total_retrieved": 0,
                    "query": query,
                }
                self.repository.log_retrieval("GDELT", query, "NO_DATA", 0, 0, 0)
                return [], meta

            data = response.json()
            raw_articles = data.get("articles", [])

            # Write to cache
            cache_payload = {
                "query": query,
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "cached_at_ts": now_ts,
                "articles": raw_articles,
            }
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_payload, f, indent=2)
            except Exception as e:
                logger.warning("Could not write GDELT cache: %s", e)

            meta = {
                "status": "LIVE_SUCCESS",
                "source": "api",
                "total_retrieved": len(raw_articles),
                "query": query,
            }
            return raw_articles, meta

        except requests.Timeout:
            err = f"GDELT request timed out after {self.timeout}s"
            logger.warning(err)
            self.repository.log_retrieval("GDELT", query, "TIMEOUT", error_message=err)
            return [], {"status": "TIMEOUT", "error": err, "total_retrieved": 0, "query": query}
        except requests.RequestException as e:
            err = f"Network error contacting GDELT: {e}"
            logger.warning(err)
            self.repository.log_retrieval("GDELT", query, "NETWORK_ERROR", error_message=str(e))
            return [], {"status": "NETWORK_ERROR", "error": str(e), "total_retrieved": 0, "query": query}
        except Exception as e:
            err = f"Unexpected error during GDELT retrieval: {e}"
            logger.error(err)
            return [], {"status": "ERROR", "error": str(e), "total_retrieved": 0, "query": query}

    def ingest_query_results(
        self,
        query: str,
        max_records: int = config.GDELT_MAX_RECORDS,
        extract_content: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Ingest articles into the database idempotently.

        Records counts for total, new, and duplicate records.
        """
        raw_articles, meta = self.fetch_articles(query, max_records=max_records, force_refresh=force_refresh)

        total_retrieved = len(raw_articles)
        new_count = 0
        duplicate_count = 0

        for item in raw_articles:
            url = item.get("url", "").strip()
            title = item.get("title", "").strip()
            if not url or not title:
                continue

            domain = item.get("domain", "")
            seendate = item.get("seendate", "")
            published_at = parse_gdelt_timestamp(seendate)
            language = item.get("language", "en")

            content_raw = None
            content_norm = None
            extraction_status = "pending"

            if extract_content:
                ext_res = self.extractor.fetch_and_extract(url)
                content_raw = ext_res["content_raw"]
                extraction_status = ext_res["extraction_status"]
                content_norm = re.sub(r"\s+", " ", content_raw).strip() if content_raw else None

            article_id, is_new = self.repository.insert_article(
                url=url,
                title=title,
                source_domain=domain,
                published_at=published_at,
                search_query=query,
                language=language,
                content_raw=content_raw,
                content_normalized=content_norm,
                extraction_status=extraction_status,
            )

            if is_new:
                new_count += 1
            else:
                duplicate_count += 1

        # Log to database
        status_str = "SUCCESS" if meta.get("status") in ("LIVE_SUCCESS", "CACHED") else meta.get("status", "FAILED")
        self.repository.log_retrieval(
            source_name="GDELT",
            query=query,
            status=status_str,
            records_retrieved=total_retrieved,
            new_records=new_count,
            duplicate_records=duplicate_count,
            error_message=meta.get("error"),
        )

        return {
            "query": query,
            "status": meta.get("status"),
            "source": meta.get("source"),
            "total_retrieved": total_retrieved,
            "new_records": new_count,
            "duplicate_records": duplicate_count,
            "error": meta.get("error"),
        }
