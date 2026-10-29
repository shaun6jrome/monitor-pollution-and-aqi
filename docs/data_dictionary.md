# Data Dictionary

This document details the schema, data types, physical units, missing value treatments, and semantics for all persistent SQLite tables and raw source datasets used in the platform.

---

## 1. SQLite Database Schema (`pollution_intelligence.db`)

### 1.1 `articles` Table
Stores unstructured news articles ingested from the GDELT DOC 2.0 API.

| Column Name | Data Type | Nullable | Description / Semantics |
| :--- | :--- | :--- | :--- |
| `id` | TEXT (PK) | No | Deterministic article ID (`art_` + 24-char SHA-256 of lowercase canonical URL). |
| `url` | TEXT (UNIQUE)| No | Canonical web URL of the news article. |
| `title` | TEXT | No | Original article headline text. |
| `source_domain` | TEXT | Yes | Extracted publisher domain (e.g., `thehindu.com`, `timesofindia.com`). |
| `published_at` | TEXT | Yes | Publication timestamp parsed to ISO 8601 UTC. |
| `retrieved_at` | TEXT | No | Ingestion timestamp recorded in ISO 8601 UTC. |
| `search_query` | TEXT | Yes | Search query topic that retrieved this article. |
| `language` | TEXT | Yes | Language code (default `en`). |
| `content_raw` | TEXT | Yes | Extracted raw body text from permitted HTTP requests. |
| `content_normalized` | TEXT | Yes | Cleaned and normalized text representation. |
| `extraction_status` | TEXT | No | Extraction outcome: `full_text`, `partial_text`, `title_only`, `paywalled_or_restricted`, `request_failed`. |
| `is_pollution_related`| INTEGER | No | Classifier output: `1` (pollution), `0` (not pollution), `-1` (uncertain). |
| `classification_confidence` | REAL | No | Posterior probability or confidence score in $[0.0, 1.0]$. |
| `classification_method` | TEXT | Yes | Active classification pipeline (`tfidf_logistic_regression` or `domain_lexicon_heuristic`). |
| `created_at` | TEXT | No | Database row creation timestamp (ISO 8601 UTC). |

---

### 1.2 `entities` Table
Stores domain entities extracted from article text with exact character spans.

| Column Name | Data Type | Nullable | Description / Semantics |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER (PK)| No | Autoincrementing primary key. |
| `article_id` | TEXT (FK) | No | Reference to `articles.id`. |
| `entity_type` | TEXT | No | Category: `LOCATION`, `POLLUTANT`, `SOURCE_CATEGORY`, `DATE_TIME`. |
| `raw_text` | TEXT | No | Exact verbatim substring as it appeared in the article text. |
| `normalized_value` | TEXT | No | Standardized gazetteer value (e.g. `Delhi, India`, `PM2.5`, `crop_burning`). |
| `confidence` | REAL | No | Extraction confidence score in $[0.0, 1.0]$. |
| `extraction_method` | TEXT | No | Extraction technique (`gazetteer_dictionary`, `regex_pattern`, `spacy_ner`). |
| `char_start` | INTEGER | Yes | Starting character index of entity span in text. |
| `char_end` | INTEGER | Yes | Ending character index of entity span in text. |
| `created_at` | TEXT | No | Timestamp of entity extraction. |

---

### 1.3 `events` Table
Stores distinct deduplicated real-world environmental pollution incidents.

| Column Name | Data Type | Nullable | Description / Semantics |
| :--- | :--- | :--- | :--- |
| `id` | TEXT (PK) | No | Deterministic event ID (`evt_` + 20-char SHA-256 hash). |
| `title` | TEXT | No | Representative headline summarizing the incident. |
| `category` | TEXT | No | Pollution taxonomy category (`crop_burning`, `industrial_emissions`, etc.). |
| `reported_location` | TEXT | Yes | Primary geographic location identified for the event. |
| `latitude` | REAL | Yes | Verified decimal latitude (null when textual only). |
| `longitude` | REAL | Yes | Verified decimal longitude (null when textual only). |
| `incident_start` | TEXT | Yes | Estimated incident initiation timestamp. |
| `incident_end` | TEXT | Yes | Estimated incident resolution or containment timestamp. |
| `first_published_at`| TEXT | No | Earliest observed article publication timestamp. |
| `last_published_at` | TEXT | No | Most recent article publication timestamp. |
| `report_count` | INTEGER | No | Number of distinct news reports clustered into this event. |
| `distinct_domain_count` | INTEGER | No | Number of unique publisher domains reporting the incident. |
| `confidence_score`| REAL | No | Aggregate evidence quality score in $[0.0, 1.0]$. |
| `evidence_summary` | TEXT | Yes | Textual synthesis of key pollutants, causes, and impacts. |
| `uncertainty_notes` | TEXT | Yes | Explicit documentation of data gaps or unverified claims. |
| `provenance_summary`| TEXT | Yes | Audit summary of reporting independence vs wire syndication. |
| `status` | TEXT | No | Lifecycle status (default `active`). |
| `created_at` | TEXT | No | Event discovery timestamp. |
| `updated_at` | TEXT | No | Timestamp of most recent article link or cluster refinement. |

---

### 1.4 `event_articles` Junction Table
Maps supporting news reports to discovered events.

| Column Name | Data Type | Nullable | Description / Semantics |
| :--- | :--- | :--- | :--- |
| `event_id` | TEXT (PK, FK) | No | Reference to `events.id`. |
| `article_id` | TEXT (PK, FK) | No | Reference to `articles.id`. |
| `similarity_score` | REAL | Yes | Cosine similarity score against event centroid or primary report. |
| `is_primary_source` | INTEGER | No | `1` if original seed article, `0` if corroborating report. |
| `evidence_snippet` | TEXT | Yes | Relevant headline or passage supporting inclusion. |
| `created_at` | TEXT | No | Timestamp association was established. |

---

### 1.5 `model_runs` Table
Records historical machine learning training runs, hyperparameters, and held-out evaluation metrics.

| Column Name | Data Type | Nullable | Description / Semantics |
| :--- | :--- | :--- | :--- |
| `id` | TEXT (PK) | No | Model run identifier (e.g., `run_RandomForestRegressor_CO(GT)`). |
| `model_type` | TEXT | No | Model algorithm name (`LinearRegression`, `RandomForestRegressor`, `XGBoostRegressor`). |
| `target_pollutant` | TEXT | No | Target variable modeled (`CO(GT)`). |
| `train_start` | TEXT | Yes | Starting timestamp of chronological training partition. |
| `train_end` | TEXT | Yes | Ending timestamp of chronological training partition. |
| `test_start` | TEXT | Yes | Starting timestamp of chronological test partition. |
| `test_end` | TEXT | Yes | Ending timestamp of chronological test partition. |
| `train_samples` | INTEGER | No | Number of training rows utilized. |
| `test_samples` | INTEGER | No | Number of held-out test rows evaluated. |
| `mae` | REAL | No | Mean Absolute Error evaluated on held-out test split. |
| `rmse` | REAL | No | Root Mean Squared Error evaluated on held-out test split. |
| `r2` | REAL | Yes | Coefficient of determination ($R^2$) on held-out test split. |
| `hyperparameters_json` | TEXT | Yes | Serialized JSON dictionary of model hyperparameters. |
| `feature_list_json` | TEXT | Yes | Serialized JSON list of input feature names. |
| `model_path` | TEXT | Yes | Path to serialized `.joblib` model artifact on disk. |
| `is_best_model` | INTEGER | No | `1` if selected as lowest RMSE model, `0` otherwise. |
| `created_at` | TEXT | No | Timestamp run was executed and logged. |

---

## 2. UCI Machine Learning Repository: Air Quality Dataset

Recorded at an Italian roadside monitoring station from March 10, 2004 to April 4, 2005 (9,358 hourly observations). Delimiter is `;` and sentinel missing value code is `-200`.

| Column Name | Original Field | Physical Units | Valid Range | Missingness Treatment | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Date` | Date | `DD/MM/YYYY` | 10/03/2004 to 04/04/2005 | None | Observation calendar date. |
| `Time` | Time | `HH.MM.SS` | 00.00.00 to 23.00.00 | None | Observation hourly timestamp. |
| `CO(GT)` | True CO | $mg/m^3$ | $0.1 - 11.9$ | `-200` converted to `NaN` (~18% missing) | Primary target: Reference analyzer Carbon Monoxide concentration. |
| `PT08.S1(CO)` | Sensor 1 | Sensor Resistance | $647 - 2040$ | `-200` converted to `NaN` (~4% missing) | Tin oxide metal oxide sensor response nominally targeted at CO. |
| `NMHC(GT)` | Non-Metanic HC | $\mu g/m^3$ | $7 - 1189$ | `-200` converted to `NaN` (~90% missing) | True Non-Metanic HydroCarbons concentration (reference analyzer). |
| `C6H6(GT)` | Benzene | $\mu g/m^3$ | $0.1 - 63.7$ | `-200` converted to `NaN` (~4% missing) | True Benzene concentration (reference analyzer). |
| `PT08.S2(NMHC)`| Sensor 2 | Sensor Resistance | $383 - 2214$ | `-200` converted to `NaN` (~4% missing) | Titania chemical sensor response nominally targeted at NMHC. |
| `NOx(GT)` | Nitrogen Oxides| $ppb$ | $2 - 1479$ | `-200` converted to `NaN` (~18% missing) | True Nitrogen Oxides concentration (reference analyzer). |
| `PT08.S3(NOx)` | Sensor 3 | Sensor Resistance | $322 - 2683$ | `-200` converted to `NaN` (~4% missing) | Tungsten oxide chemical sensor response nominally targeted at NOx. |
| `NO2(GT)` | Nitrogen Dioxide| $\mu g/m^3$ | $2 - 340$ | `-200` converted to `NaN` (~18% missing) | True Nitrogen Dioxide concentration (reference analyzer). |
| `PT08.S4(NO2)` | Sensor 4 | Sensor Resistance | $551 - 2775$ | `-200` converted to `NaN` (~4% missing) | Tungsten oxide chemical sensor response nominally targeted at NO2. |
| `PT08.S5(O3)` | Sensor 5 | Sensor Resistance | $221 - 2523$ | `-200` converted to `NaN` (~4% missing) | Indium oxide chemical sensor response nominally targeted at Ozone. |
| `T` | Temperature | $^\circ C$ | $-1.9 - 44.6$ | `-200` converted to `NaN` (~4% missing) | Ambient air temperature. |
| `RH` | Relative Humidity| $\%$ | $9.2 - 88.7$ | `-200` converted to `NaN` (~4% missing) | Relative humidity percentage. |
| `AH` | Absolute Humidity| $g/m^3$ | $0.18 - 2.23$ | `-200` converted to `NaN` (~4% missing) | Absolute ambient humidity. |
