# System Architecture: Evidence-Aware Pollution Event Discovery and Air-Quality Prediction

## 1. Architectural Overview

The platform implements an end-to-end data science ecosystem combining real-time unstructured news ingestion, multi-criteria natural language processing (NLP), duplicate-resistant event discovery, and historical pollutant concentration forecasting.

The architecture enforces strict separation between:
1. **Live Unstructured News Stream** (GDELT DOC 2.0 API)
2. **Official Ambient Monitoring Feeds** (Central Pollution Control Board / data.gov.in)
3. **Historical Empirical Sensor Benchmark** (UCI Machine Learning Repository Air Quality Dataset)
4. **Predictive Inference Engine** (Trained regression models)

---

## 2. Mermaid System Architecture Diagram

```mermaid
flowchart TD
    subgraph DataSources["External Data Sources"]
        GDELT["GDELT DOC 2.0 API (Global News Stream)"]
        UCI["UCI Machine Learning Repository (9,358 Sensor Observations)"]
        CPCB["CPCB / data.gov.in (Official Station Feeds)"]
    end

    subgraph Ingestion["Ingestion & In-Memory / Disk Caching"]
        GClient["GDELT Client & JSON Cache (TTL=3600s)"]
        Extractor["Respectful Article Extractor (HTML Parser)"]
        CPCBClient["CPCB Connector & Status Auditor"]
        SQLiteDB[("SQLite Persistent Storage\n(articles, events, entities, model_runs)")]
    end

    subgraph NLP["Unstructured Text Processing Pipeline"]
        Cleaner["Text Cleaner (Unicode NFKC, Boilerplate Strip)"]
        Classifier["Relevance Classifier (TF-IDF + Logistic Regression)"]
        NER["Entity Extractor (Locations, Pollutants, Dates, Sources)"]
    end

    subgraph Intelligence["Core Innovation: Event Discovery & Provenance"]
        Embedder["Semantic Embeddings (Sentence Transformers / TF-IDF Fallback)"]
        Matcher["Multi-Criteria Matcher (Semantic + Location + 72h Window + Category)"]
        ProvAudit["Provenance & Wire Syndication Analyzer"]
        Aggregator["Distinct Event Aggregator"]
    end

    subgraph Forecasting["Historical ML Forecasting Engine"]
        DatasetLoader["UCI Loader (-200 Sentinel Conversion)"]
        FE["Feature Engineering (Diurnal/Seasonal Cyclical, Past Lags)"]
        Split["Strict Chronological Split (80% Train / 20% Test)"]
        Train["Regression Models (Linear Regression, Random Forest, XGBoost)"]
        Eval["Held-Out Evaluation (MAE, RMSE, R^2, Residuals)"]
        Artifacts["Model Artifacts (models/*.joblib)"]
    end

    subgraph Interface["Streamlit Environmental Intelligence Dashboard"]
        Overview["Page A: Overview & Telemetry"]
        NewsUI["Page B: News Intelligence"]
        EventUI["Page C: Event Explorer & Audit Trail"]
        GeoUI["Page D: Geographic Breakdown"]
        MLLab["Page E: ML Laboratory"]
        PredictUI["Page F: Interactive Forecasting"]
        InnovUI["Page G: Innovation Evaluation"]
        DocUI["Page H & I: Documentation & Data Exports"]
    end

    GDELT --> GClient
    GClient --> Extractor
    Extractor --> SQLiteDB
    CPCB --> CPCBClient --> SQLiteDB

    SQLiteDB --> Cleaner
    Cleaner --> Classifier
    Classifier --> NER
    NER --> Embedder
    Embedder --> Matcher
    Matcher --> ProvAudit
    ProvAudit --> Aggregator
    Aggregator --> SQLiteDB

    UCI --> DatasetLoader --> FE --> Split --> Train --> Eval
    Eval --> SQLiteDB
    Train --> Artifacts

    SQLiteDB --> Interface
    Artifacts --> PredictUI
```

---

## 3. Workflow Progression Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit Dashboard
    participant GDELT as GDELT DOC 2.0 API
    participant Cache as Local Disk Cache
    participant DB as SQLite Database
    participant NLP as NLP & Deduplication Pipeline
    participant ML as ML Forecasting Pipeline

    User->>UI: Request news refresh for query ("industrial emissions")
    UI->>Cache: Check for valid unexpired query cache
    alt Cache Hit (TTL < 3600s)
        Cache-->>UI: Return cached article records
    else Cache Miss / Force Refresh
        UI->>GDELT: Query API with rate-limited HTTP GET
        GDELT-->>UI: Return JSON article payload
        UI->>Cache: Persist raw response to disk
        UI->>DB: Idempotently insert articles (SHA-256 URL hashing)
    end

    UI->>NLP: Run classification, NER, and multi-criteria clustering
    NLP->>NLP: Filter non-pollution news (TF-IDF + Logistic Regression)
    NLP->>NLP: Extract locations, pollutants, and temporal anchors
    NLP->>NLP: Group reports within 72h window and matching geography
    NLP->>DB: Update distinct event records & syndication provenance
    DB-->>UI: Render updated event clusters and evidence audit cards

    User->>UI: Request Model Training
    UI->>ML: Load UCI dataset, run chronological split, train models
    ML->>ML: Fit Linear Reg, Random Forest, XGBoost on identical test split
    ML->>DB: Log evaluation metrics (MAE, RMSE, R^2)
    DB-->>UI: Display model comparison table and best model selection
```

---

## 4. Key Subsystem Specifications

### 4.1 Ingestion & Idempotency
- **GDELT Client:** Implements exponential timeout handling (default 20s), bounded response sizes, and deterministic SHA-256 caching.
- **Article Extractor:** Respects robots constraints, HTTP 401/403 paywalls, and strips boilerplate navigation markup without losing pollutant formulas (`PM2.5`, `SO2`, `NOx`).
- **Idempotent Ingestion:** Articles are keyed by `art_` + SHA-256 of canonical URL. Repeated runs never duplicate database records.

### 4.2 NLP and Event Discovery
- **Relevance Classifier:** Supervised TF-IDF baseline with a transparent ground-truth labeling corpus distinguishing real industrial/agricultural pollution from sports, financial, and political distractions.
- **Entity Extractor:** Rule-based gazetteer and regex patterns for 30+ environmental hubs, chemical pollutants, and source taxonomies.
- **Deduplication Engine:** Integrates cosine similarity with geographic boundary matching, 72-hour temporal windows, and source syndication checks.

### 4.3 Machine Learning Forecasting
- **Dataset:** UCI Air Quality Multisensor dataset (9,358 hourly observations).
- **Target:** Carbon Monoxide true concentration `CO(GT)` in $mg/m^3$.
- **Validation:** Chronological split (80% train, 20% test). Transformation medians are calculated exclusively on training partitions to prevent data leakage.
