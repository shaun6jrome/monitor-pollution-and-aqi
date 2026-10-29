"""SQLite Repository for the Pollution Intelligence Platform.

Handles schema initialization, idempotent ingestion, transactional queries,
and provenance tracking for articles, entities, discovered events, logs,
and model runs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config


def get_utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def generate_article_id(url: str) -> str:
    """Generate deterministic SHA-256 ID based on canonical URL."""
    clean_url = url.strip().lower()
    return "art_" + hashlib.sha256(clean_url.encode("utf-8")).hexdigest()[:24]


def generate_event_id(category: str, location: Optional[str], timestamp: str) -> str:
    """Generate deterministic event ID seed."""
    raw = f"{category}_{location or 'unknown'}_{timestamp}"
    return "evt_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class PollutionRepository:
    """Database repository providing thread-safe SQLite operations."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or config.SQLITE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return connection with foreign keys and Row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def init_schema(self) -> None:
        """Initialize database schema from schema.sql if tables do not exist."""
        schema_file = config.BASE_DIR / "database" / "schema.sql"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_file}")

        with open(schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    # ---------------------------------------------------------
    # Article Operations
    # ---------------------------------------------------------
    def insert_article(
        self,
        url: str,
        title: str,
        source_domain: Optional[str] = None,
        published_at: Optional[str] = None,
        search_query: Optional[str] = None,
        language: str = "en",
        content_raw: Optional[str] = None,
        content_normalized: Optional[str] = None,
        extraction_status: str = "pending",
        is_pollution_related: int = -1,
        classification_confidence: float = 0.0,
        classification_method: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Insert an article idempotently.

        Returns (article_id, is_new).
        """
        article_id = generate_article_id(url)
        now = get_utc_now_iso()

        sql = """
        INSERT INTO articles (
            id, url, title, source_domain, published_at, retrieved_at,
            search_query, language, content_raw, content_normalized,
            extraction_status, is_pollution_related, classification_confidence,
            classification_method, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            search_query = COALESCE(excluded.search_query, articles.search_query),
            content_raw = COALESCE(excluded.content_raw, articles.content_raw),
            content_normalized = COALESCE(excluded.content_normalized, articles.content_normalized),
            extraction_status = CASE
                WHEN articles.extraction_status = 'full_text' THEN 'full_text'
                ELSE excluded.extraction_status
            END,
            is_pollution_related = CASE
                WHEN articles.is_pollution_related != -1 THEN articles.is_pollution_related
                ELSE excluded.is_pollution_related
            END
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM articles WHERE url = ?", (url,))
            existing = cursor.fetchone()
            is_new = existing is None

            cursor.execute(
                sql,
                (
                    article_id,
                    url,
                    title,
                    source_domain,
                    published_at,
                    now,
                    search_query,
                    language,
                    content_raw,
                    content_normalized,
                    extraction_status,
                    is_pollution_related,
                    classification_confidence,
                    classification_method,
                    now,
                ),
            )
            conn.commit()
            return article_id, is_new

    def update_article_content(
        self,
        article_id: str,
        content_raw: str,
        content_normalized: str,
        extraction_status: str,
    ) -> None:
        """Update extracted text content and extraction status for an article."""
        sql = """
        UPDATE articles
        SET content_raw = ?, content_normalized = ?, extraction_status = ?
        WHERE id = ?
        """
        with self._get_connection() as conn:
            conn.execute(sql, (content_raw, content_normalized, extraction_status, article_id))
            conn.commit()

    def update_article_classification(
        self,
        article_id: str,
        is_pollution_related: int,
        confidence: float,
        method: str,
    ) -> None:
        """Update classification status for an article."""
        sql = """
        UPDATE articles
        SET is_pollution_related = ?, classification_confidence = ?, classification_method = ?
        WHERE id = ?
        """
        with self._get_connection() as conn:
            conn.execute(sql, (is_pollution_related, confidence, method, article_id))
            conn.commit()

    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Fetch article by ID."""
        sql = "SELECT * FROM articles WHERE id = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (article_id,)).fetchone()
            return dict(row) if row else None

    def list_articles(
        self,
        limit: int = 100,
        offset: int = 0,
        only_pollution: Optional[bool] = None,
        query_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List articles with optional filters."""
        query = "SELECT * FROM articles WHERE 1=1"
        params: List[Any] = []

        if only_pollution is True:
            query += " AND is_pollution_related = 1"
        elif only_pollution is False:
            query += " AND is_pollution_related = 0"

        if query_filter:
            query += " AND (title LIKE ? OR content_normalized LIKE ? OR source_domain LIKE ?)"
            pattern = f"%{query_filter}%"
            params.extend([pattern, pattern, pattern])

        query += " ORDER BY published_at DESC, created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def count_articles(self, only_pollution: Optional[bool] = None) -> int:
        """Count articles with optional filtering."""
        query = "SELECT COUNT(*) as cnt FROM articles WHERE 1=1"
        params: List[Any] = []
        if only_pollution is True:
            query += " AND is_pollution_related = 1"
        elif only_pollution is False:
            query += " AND is_pollution_related = 0"

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return row["cnt"] if row else 0

    # ---------------------------------------------------------
    # Entity Operations
    # ---------------------------------------------------------
    def insert_entity(
        self,
        article_id: str,
        entity_type: str,
        raw_text: str,
        normalized_value: str,
        confidence: float = 1.0,
        extraction_method: str = "rule_based",
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
    ) -> int:
        """Insert extracted entity with span attribution."""
        now = get_utc_now_iso()
        sql = """
        INSERT INTO entities (
            article_id, entity_type, raw_text, normalized_value,
            confidence, extraction_method, char_start, char_end, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                sql,
                (
                    article_id,
                    entity_type,
                    raw_text,
                    normalized_value,
                    confidence,
                    extraction_method,
                    char_start,
                    char_end,
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid or 0

    def get_entities_by_article(self, article_id: str) -> List[Dict[str, Any]]:
        """Fetch all entities extracted from an article."""
        sql = "SELECT * FROM entities WHERE article_id = ? ORDER BY id ASC"
        with self._get_connection() as conn:
            rows = conn.execute(sql, (article_id,)).fetchall()
            return [dict(r) for r in rows]

    # ---------------------------------------------------------
    # Event Operations
    # ---------------------------------------------------------
    def create_event(
        self,
        event_id: str,
        title: str,
        category: str,
        reported_location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        incident_start: Optional[str] = None,
        incident_end: Optional[str] = None,
        first_published_at: Optional[str] = None,
        last_published_at: Optional[str] = None,
        report_count: int = 1,
        distinct_domain_count: int = 1,
        confidence_score: float = 0.5,
        evidence_summary: Optional[str] = None,
        uncertainty_notes: Optional[str] = None,
        provenance_summary: Optional[str] = None,
    ) -> str:
        """Insert a newly discovered pollution event."""
        now = get_utc_now_iso()
        pub_start = first_published_at or now
        pub_end = last_published_at or now

        sql = """
        INSERT INTO events (
            id, title, category, reported_location, latitude, longitude,
            incident_start, incident_end, first_published_at, last_published_at,
            report_count, distinct_domain_count, confidence_score,
            evidence_summary, uncertainty_notes, provenance_summary,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
        """
        with self._get_connection() as conn:
            conn.execute(
                sql,
                (
                    event_id,
                    title,
                    category,
                    reported_location,
                    latitude,
                    longitude,
                    incident_start,
                    incident_end,
                    pub_start,
                    pub_end,
                    report_count,
                    distinct_domain_count,
                    confidence_score,
                    evidence_summary,
                    uncertainty_notes,
                    provenance_summary,
                    now,
                    now,
                ),
            )
            conn.commit()
            return event_id

    def add_article_to_event(
        self,
        event_id: str,
        article_id: str,
        similarity_score: float = 1.0,
        is_primary_source: int = 0,
        evidence_snippet: Optional[str] = None,
    ) -> None:
        """Link an article to an event with evidence and similarity score."""
        now = get_utc_now_iso()
        sql = """
        INSERT INTO event_articles (
            event_id, article_id, similarity_score, is_primary_source,
            evidence_snippet, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id, article_id) DO UPDATE SET
            similarity_score = excluded.similarity_score,
            evidence_snippet = COALESCE(excluded.evidence_snippet, event_articles.evidence_snippet)
        """
        with self._get_connection() as conn:
            conn.execute(
                sql,
                (
                    event_id,
                    article_id,
                    similarity_score,
                    is_primary_source,
                    evidence_snippet,
                    now,
                ),
            )
            # Update event report count and distinct domains
            conn.execute(
                """
                UPDATE events
                SET report_count = (
                    SELECT COUNT(*) FROM event_articles WHERE event_id = ?
                ),
                distinct_domain_count = (
                    SELECT COUNT(DISTINCT a.source_domain)
                    FROM event_articles ea
                    JOIN articles a ON ea.article_id = a.id
                    WHERE ea.event_id = ?
                ),
                updated_at = ?
                WHERE id = ?
                """,
                (event_id, event_id, now, event_id),
            )
            conn.commit()

    def list_events(
        self,
        category: Optional[str] = None,
        min_reports: int = 1,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List discovered events with optional category and report filters."""
        query = "SELECT * FROM events WHERE report_count >= ?"
        params: List[Any] = [min_reports]

        if category and category != "all":
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY last_published_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_event_with_articles(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Fetch event details along with all supporting articles."""
        with self._get_connection() as conn:
            event_row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
            if not event_row:
                return None

            event = dict(event_row)
            articles_query = """
            SELECT a.*, ea.similarity_score, ea.is_primary_source, ea.evidence_snippet
            FROM event_articles ea
            JOIN articles a ON ea.article_id = a.id
            WHERE ea.event_id = ?
            ORDER BY ea.similarity_score DESC
            """
            art_rows = conn.execute(articles_query, (event_id,)).fetchall()
            event["articles"] = [dict(r) for r in art_rows]
            return event

    def count_events(self) -> int:
        """Return total count of distinct events."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
            return row["cnt"] if row else 0

    # ---------------------------------------------------------
    # Retrieval Logs
    # ---------------------------------------------------------
    def log_retrieval(
        self,
        source_name: str,
        query: Optional[str],
        status: str,
        records_retrieved: int = 0,
        new_records: int = 0,
        duplicate_records: int = 0,
        error_message: Optional[str] = None,
    ) -> int:
        """Record an ingestion run in retrieval_logs."""
        now = get_utc_now_iso()
        sql = """
        INSERT INTO retrieval_logs (
            source_name, query, status, records_retrieved,
            new_records, duplicate_records, error_message, executed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                sql,
                (
                    source_name,
                    query,
                    status,
                    records_retrieved,
                    new_records,
                    duplicate_records,
                    error_message,
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid or 0

    def get_latest_retrieval_log(self, source_name: str = "GDELT") -> Optional[Dict[str, Any]]:
        """Get the most recent retrieval log for a source."""
        sql = """
        SELECT * FROM retrieval_logs
        WHERE source_name = ?
        ORDER BY id DESC LIMIT 1
        """
        with self._get_connection() as conn:
            row = conn.execute(sql, (source_name,)).fetchone()
            return dict(row) if row else None

    # ---------------------------------------------------------
    # Model Runs
    # ---------------------------------------------------------
    def log_model_run(
        self,
        run_id: str,
        model_type: str,
        target_pollutant: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        train_samples: int,
        test_samples: int,
        mae: float,
        rmse: float,
        r2: Optional[float] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_list: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        is_best_model: int = 0,
    ) -> None:
        """Persist ML model evaluation metrics and training metadata."""
        now = get_utc_now_iso()
        sql = """
        INSERT INTO model_runs (
            id, model_type, target_pollutant, train_start, train_end,
            test_start, test_end, train_samples, test_samples,
            mae, rmse, r2, hyperparameters_json, feature_list_json,
            model_path, is_best_model, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            mae = excluded.mae,
            rmse = excluded.rmse,
            r2 = excluded.r2,
            is_best_model = excluded.is_best_model
        """
        with self._get_connection() as conn:
            conn.execute(
                sql,
                (
                    run_id,
                    model_type,
                    target_pollutant,
                    train_start,
                    train_end,
                    test_start,
                    test_end,
                    train_samples,
                    test_samples,
                    mae,
                    rmse,
                    r2,
                    json.dumps(hyperparameters or {}),
                    json.dumps(feature_list or []),
                    model_path,
                    is_best_model,
                    now,
                ),
            )
            conn.commit()

    def list_model_runs(self, target_pollutant: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recorded model runs ordered by RMSE ascending."""
        query = "SELECT * FROM model_runs WHERE 1=1"
        params: List[Any] = []
        if target_pollutant:
            query += " AND target_pollutant = ?"
            params.append(target_pollutant)
        query += " ORDER BY rmse ASC, created_at DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ---------------------------------------------------------
    # CPCB Station Observations
    # ---------------------------------------------------------
    def insert_cpcb_observation(
        self,
        station_name: str,
        pollutant: str,
        value: float,
        unit: str,
        city: Optional[str] = None,
        state: Optional[str] = None,
        aqi_index: Optional[float] = None,
        observation_timestamp: Optional[str] = None,
        source_attribution: str = "CPCB / National AQI Portal",
    ) -> int:
        """Insert an official station observation."""
        now = get_utc_now_iso()
        obs_time = observation_timestamp or now
        sql = """
        INSERT INTO cpcb_observations (
            station_name, city, state, pollutant, value, unit,
            aqi_index, observation_timestamp, source_attribution, retrieved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                sql,
                (
                    station_name,
                    city,
                    state,
                    pollutant,
                    value,
                    unit,
                    aqi_index,
                    obs_time,
                    source_attribution,
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid or 0

    def get_latest_cpcb_observations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch most recent official station observations."""
        sql = """
        SELECT * FROM cpcb_observations
        ORDER BY observation_timestamp DESC, id DESC LIMIT ?
        """
        with self._get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            return [dict(r) for r in rows]
