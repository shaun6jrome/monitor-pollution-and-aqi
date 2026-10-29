# Academic Oral Defense Presentation Outline (10 Slides)

## Slide 1: Title and Problem Statement
- **Title:** Evidence-Aware Pollution Event Discovery and Air-Quality Prediction Using Unstructured Data
- **Domain:** Environmental Data Science / Natural Language Processing / Predictive Machine Learning
- **Problem Statement:** Ambient air quality monitoring stations provide accurate physical concentrations but cannot identify the human causes or episodic events (industrial leaks, stubble fires, waste burning) behind pollution spikes. Unstructured news contains causal evidence, but naive text processing suffers from massive duplication, geographic conflation, and lack of provenance.

---

## Slide 2: Motivation and Limitations of Existing Approaches
- **Physical Sensor Limits:** Sparse spatial density; cannot capture non-point emission events; unable to explain *why* a pollutant spiked.
- **Limitation of Keyword-Counting:** Treating every news article as a distinct event inflates incident frequency by 100%+ due to syndicated wire reporting (PTI, ANI, Reuters).
- **Limitation of Semantic-Only Grouping:** Vector similarity alone collapses separate incidents occurring in different cities (e.g. Mumbai vs Delhi) or weeks apart into single false clusters.

---

## Slide 3: Objectives and Proposed Contribution
- **Primary Objective:** Build an end-to-end data science ecosystem integrating real-time news retrieval, multi-criteria incident clustering, provenance auditing, and empirical pollutant regression.
- **Central Innovation:** A multi-dimensional event discovery engine combining semantic cosine similarity, geographic gazetteer verification, a 72-hour temporal event window, and wire syndication auditing.
- **Forecasting Objective:** Chronological baseline and non-linear regression modeling (Linear Regression, Random Forest, XGBoost) on historical sensor data with zero data leakage.

---

## Slide 4: Data Sources and Ingestion Architecture
- **Source A (Unstructured News):** GDELT DOC 2.0 API; configurable environmental queries; SHA-256 URL hashing; local JSON caching (TTL=3600s); respectful content extraction.
- **Source B (Historical Empirical Sensors):** UCI Machine Learning Repository Air Quality dataset (9,358 hourly observations; Italian roadside deployment; 5 metal oxide chemical sensors; -200 sentinel missingness).
- **Source C (Official Government Feeds):** Central Pollution Control Board (CPCB) via data.gov.in real-time AQI API.
- **Data Governance Principle:** Complete separation between historical calibration, live official telemetry, and modern unstructured news.

---

## Slide 5: Unstructured Data NLP Pipeline
- **Text Preprocessing:** NFKC Unicode normalization; HTML entity unescaping; regex-based advertising and boilerplate removal; preservation of chemical notations (PM2.5, SO2, NOx).
- **Relevance Classification:** Supervised TF-IDF vectorizer + Logistic Regression baseline; transparent benchmark corpus; inspectable domain lexicon fallback.
- **Entity Extraction:** Gazetteer covering 30+ environmental monitoring hubs; chemical pollutant patterns; 8-class emission source taxonomy; temporal anchor extraction.

---

## Slide 6: Core Innovation — Duplicate-Resistant Event Discovery
- **Embedding Pipeline:** Sentence Transformers (`all-MiniLM-L6-v2`) with unit-normalized vectors and TF-IDF fallback.
- **Multi-Criteria Fusion:**
  1. Cosine similarity threshold ($\ge 0.68$).
  2. Geographic boundary compatibility (rejects conflicting cities).
  3. 72-hour rolling temporal window ($|\Delta t| \le 72$ hrs).
  4. Emission category alignment.
- **Provenance Auditing:** Identifies cross-domain wire syndication ($\text{sim} \ge 0.85$); calculates calibrated evidence quality scores ($0.50 - 0.90$).

---

## Slide 7: Machine Learning Algorithms & Experimental Design
- **Target Variable:** Carbon Monoxide true concentration `CO(GT)` in $mg/m^3$.
- **Leakage Prevention:** Strict chronological split (80% train / 20% test); temporal cyclical features (sin/cos of hour and month); lag variables ($t-1, t-2, t-3, t-24$); rolling statistics (`closed='left'`); imputation medians computed strictly on training data.
- **Models Compared:**
  1. Linear Regression (standardized baseline)
  2. Random Forest Regressor (100 trees, depth 12)
  3. XGBoost Regressor (gradient boosted trees, lr=0.08, subsample=0.85)

---

## Slide 8: Actual Experimental Results and Dashboard Demonstration
- **Innovation Benchmark Results:**
  - Baseline 1 (Keyword Count): 10 discovered events (True: 5; Precision: 0.00, F1: 0.00).
  - Baseline 2 (Semantic Only): 6 discovered events (False merges across cities; F1: 0.72).
  - Proposed Multi-Criteria: 5 discovered events (Exact match to ground truth; Precision: 1.00, Recall: 1.00, F1: 1.00).
- **ML Regression Results:**
  - Random Forest and XGBoost substantially outperform the linear baseline on held-out test data.
- **Streamlit Interface:** 9 interactive pages featuring real-time GDELT sync, event provenance drilldowns, geographic breakdowns, and audited prediction interfaces.

---

## Slide 9: Limitations and Future Work
- **Spatial/Temporal Alignment:** Honest documentation that 2004 Italian sensor observations cannot be combined with 2024 Indian news for direct predictive augmentation without scientific fraud.
- **Text Geocoding Constraints:** Locations are preserved as verified textual strings rather than unverified fabricated coordinates.
- **Future Directions:** Collocated deployment pairing municipal CAAQMS stations with hyperlocal vernacular news and social grievance redressal portals.

---

## Slide 10: Conclusion and References
- **Summary:** Successfully implemented an end-to-end evidence-aware environmental intelligence ecosystem that resolves news duplication and reliably models pollutant concentrations.
- **Key Deliverables:** Fully functional Python/Streamlit platform, SQLite provenance database, 25-question viva guide, and 24-section comprehensive technical report.
- **Selected References:**
  - GDELT Project DOC 2.0 API Documentation (2024).
  - De Vito et al., *Sensors and Actuators B: Chemical* (2008) — UCI Air Quality Dataset.
  - Central Pollution Control Board (CPCB) National AQI Technical Manual (2014).
