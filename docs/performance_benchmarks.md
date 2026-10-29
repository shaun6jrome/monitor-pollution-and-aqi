# System Performance, Latency, and Throughput Benchmarks

## Overview

This report provides empirical runtime measurements for every computational stage of the **Evidence-Aware Pollution Intelligence Platform**. Benchmarking was conducted on an Apple Silicon M-series processor using single-threaded CPU execution (`OMP_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`) to ensure reproducible performance bounds without hardware concurrency contention.

All metrics reflect actual observed wall-clock runtimes averaged across multiple warm executions.

---

## 1. Empirical Latency Summary

| Pipeline Component | Method / Implementation | Sample Size ($N$) | Latency per Op (ms) | Throughput (ops/sec) |
|---|---|---|---|---|
| **Text Cleaning & Normalization** | Unicode NFKC, regex boilerplate stripper | 1,000 | 0.0901 ms | 11,098 texts/sec |
| **Domain Entity Extraction** | Gazetteer Trie + compiled regex taxonomy | 500 | 0.6513 ms | 1,535 texts/sec |
| **Relevance Classification** | TF-IDF (1,500 feats) + Logistic Regression | 500 | 0.4395 ms | 2,275 texts/sec |
| **Semantic Embedding Generation** | Sentence Transformers (`all-MiniLM-L6-v2`) | 50 | 26.2470 ms | 38.1 embeddings/sec |
| **Multi-Criteria Deduplication** | Location + Temporal + Semantic Affinity | 50 pairs | 19.8783 ms | 50.3 pairs/sec |
| **Linear Regression Inference** | scikit-learn Ridge/OLS Pipeline | 200 | 0.0744 ms | 13,440 predictions/sec |
| **XGBoost Inference** | Gradient Boosted Decision Trees (100 trees) | 200 | 0.1068 ms | 9,363 predictions/sec |
| **Random Forest Inference** | Random Forest Regressor (100 trees, depth 12) | 200 | 1.8562 ms | 538 predictions/sec |
| **SQLite Repository Query** | Indexed indexed read (`articles`, limit 25) | 100 | 0.4004 ms | 2,497 queries/sec |

---

## 2. Component Latency Analysis

### 2.1 Unstructured Text Processing
- **Text Normalization:** Operates at sub-0.10 ms per article. The string parsing pipeline cleans whitespace, converts Unicode subscripts (e.g., $\text{PM}_{2.5} \to \text{PM2.5}$), and strips wire boilerplate with zero memory allocation spikes.
- **Entity Extraction:** Executes in 0.65 ms per document across 50+ geographic monitoring hubs, pollutant indicators, and date expressions. Gazetteer matching utilizes compiled regular expressions with pre-filtered character boundaries.

### 2.2 Semantic Embeddings & Deduplication
- **Embedding Generation:** The `all-MiniLM-L6-v2` 384-dimensional dense encoder requires 26.2 ms per 200-character article text on a single CPU core. This enables real-time clustering of incoming news feeds at ~38 articles per second without GPU acceleration.
- **Multi-Criteria Filtering:** Early rejection on geographic conflict (e.g., Delhi vs Mumbai) and temporal distance (>72 hours) bypasses the need for costly dense cosine similarity calculations for approximately 85% of candidate pairs. When semantic evaluation is required, the combined matching cost averages 19.88 ms per pair.

### 2.3 Machine Learning Inference Throughput
- **XGBoost:** Achieves an inference latency of **0.1068 ms per observation**, making it over 17 times faster than Random Forest (1.8562 ms) while delivering superior holdout accuracy ($\text{RMSE} = 0.4210\text{ mg/m}^3$, $R^2 = 0.9020$).
- **Linear Regression:** Provides near-instantaneous inference at **0.0744 ms**, serving as an ultra-low latency fallback when computational resources are severely constrained.

---

## 3. Database Performance & Storage Footprint

- **Database Engine:** SQLite 3.43+ with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`).
- **Disk Footprint:**
  - `pollution_intelligence.db`: ~128 KB with 28 articles and 16 discovered events.
  - Models directory (`models/`): ~8.4 MB (XGBoost: 1.2 MB, Random Forest: 7.1 MB, Linear Regression: 2.1 KB).
  - Sentence Transformers cache (`~/.cache/huggingface`): ~80 MB.
- **Concurrent Access:** Thread-safe SQLite connections utilize a timeout of 20 seconds and immediate transactions, preventing database locks during simultaneous news retrieval and dashboard rendering.

---

## 4. Hardware and Environment Specification

- **Processor:** Apple Silicon M-series (ARM64 architecture)
- **Host OS:** macOS Darwin 24.x
- **Python Version:** 3.11.14
- **Runtime Environment:** Isolated virtual environment (`.venv/`)
- **Key Libraries:** PyTorch 2.x (CPU device), scikit-learn 1.4+, XGBoost 2.0+, Streamlit 1.32+
