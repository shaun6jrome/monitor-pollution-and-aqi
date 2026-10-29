-- Database Schema for Evidence-Aware Pollution Event Discovery and Air-Quality Prediction

PRAGMA foreign_keys = ON;

-- 1. Articles Table: Stores metadata, content, and classification for unstructured news
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    source_domain TEXT,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    search_query TEXT,
    language TEXT DEFAULT 'en',
    content_raw TEXT,
    content_normalized TEXT,
    extraction_status TEXT NOT NULL DEFAULT 'pending',
    is_pollution_related INTEGER DEFAULT -1,
    classification_confidence REAL DEFAULT 0.0,
    classification_method TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_pollution ON articles(is_pollution_related);
CREATE INDEX IF NOT EXISTS idx_articles_source_domain ON articles(source_domain);

-- 2. Entities Table: Extracted locations, dates, pollutants, and sources with provenance spans
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    normalized_value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    extraction_method TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entities_article_id ON entities(article_id);
CREATE INDEX IF NOT EXISTS idx_entities_type_norm ON entities(entity_type, normalized_value);

-- 3. Events Table: Distinct deduplicated real-world pollution incidents
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    reported_location TEXT,
    latitude REAL,
    longitude REAL,
    incident_start TEXT,
    incident_end TEXT,
    first_published_at TEXT NOT NULL,
    last_published_at TEXT NOT NULL,
    report_count INTEGER NOT NULL DEFAULT 1,
    distinct_domain_count INTEGER NOT NULL DEFAULT 1,
    confidence_score REAL NOT NULL DEFAULT 0.5,
    evidence_summary TEXT,
    uncertainty_notes TEXT,
    provenance_summary TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
CREATE INDEX IF NOT EXISTS idx_events_first_pub ON events(first_published_at);
CREATE INDEX IF NOT EXISTS idx_events_location ON events(reported_location);

-- 4. Event Articles Junction: Maps supporting evidence articles to discovered events
CREATE TABLE IF NOT EXISTS event_articles (
    event_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    similarity_score REAL,
    is_primary_source INTEGER DEFAULT 0,
    evidence_snippet TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (event_id, article_id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_event_articles_event ON event_articles(event_id);
CREATE INDEX IF NOT EXISTS idx_event_articles_article ON event_articles(article_id);

-- 5. Retrieval Logs: Records API sync events, new/duplicate counts, and health status
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    query TEXT,
    status TEXT NOT NULL,
    records_retrieved INTEGER DEFAULT 0,
    new_records INTEGER DEFAULT 0,
    duplicate_records INTEGER DEFAULT 0,
    error_message TEXT,
    executed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieval_logs_source ON retrieval_logs(source_name);
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_executed ON retrieval_logs(executed_at);

-- 6. Model Training and Evaluation Runs: Auditable record of ML experiments
CREATE TABLE IF NOT EXISTS model_runs (
    id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    target_pollutant TEXT NOT NULL,
    train_start TEXT,
    train_end TEXT,
    test_start TEXT,
    test_end TEXT,
    train_samples INTEGER NOT NULL,
    test_samples INTEGER NOT NULL,
    mae REAL NOT NULL,
    rmse REAL NOT NULL,
    r2 REAL,
    hyperparameters_json TEXT,
    feature_list_json TEXT,
    model_path TEXT,
    is_best_model INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_runs_model_type ON model_runs(model_type);
CREATE INDEX IF NOT EXISTS idx_model_runs_created_at ON model_runs(created_at);

-- 7. CPCB Official Station Observations: Live or latest government monitoring readings
CREATE TABLE IF NOT EXISTS cpcb_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_name TEXT NOT NULL,
    city TEXT,
    state TEXT,
    pollutant TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    aqi_index REAL,
    observation_timestamp TEXT NOT NULL,
    source_attribution TEXT NOT NULL,
    retrieved_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cpcb_station ON cpcb_observations(station_name);
CREATE INDEX IF NOT EXISTS idx_cpcb_obs_time ON cpcb_observations(observation_timestamp);
