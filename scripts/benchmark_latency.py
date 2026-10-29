"""Empirical Latency and Throughput Benchmarking Script.

Measures the wall-clock execution time of every core pipeline component:
1. Text cleaning and Unicode NFKC normalization
2. Domain entity extraction (gazetteer and regex)
3. Supervised pollution classification inference
4. Semantic sentence embedding generation (CPU)
5. Multi-criteria deduplication matching
6. ML concentration forecasting model inference
7. SQLite repository read/write operations

Generates verifiable metrics for the viva defense and documentation.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from src.forecasting.predict import ModelPredictor
from src.intelligence.deduplication import match_event_candidates
from src.intelligence.embeddings import SemanticEmbeddingManager
from src.preprocessing.event_classifier import PollutionClassifier
from src.preprocessing.location_extractor import EntityExtractor
from src.preprocessing.text_cleaner import clean_text
from src.storage.repository import PollutionRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_pipeline_components() -> Dict[str, Any]:
    """Execute latency benchmark across pipeline modules."""
    results = {}
    sample_text = (
        "Dense toxic smog blanketed Delhi NCR on Friday as the Air Quality Index "
        "worsened past 450 into the hazardous category. High concentrations of PM2.5 "
        "and PM10 were recorded at Anand Vihar, while industrial units in Ghaziabad "
        "faced temporary closure orders under GRAP Stage IV."
    )

    # 1. Text Cleaning
    n_clean = 1000
    t0 = time.perf_counter()
    for _ in range(n_clean):
        _ = clean_text(sample_text)
    t_clean = (time.perf_counter() - t0) / n_clean * 1000.0
    results["text_cleaning_ms"] = round(t_clean, 4)
    logger.info("Text cleaning: %.4f ms / sample", t_clean)

    # 2. Entity Extraction
    extractor = EntityExtractor(use_spacy=False)
    n_ner = 500
    t0 = time.perf_counter()
    for _ in range(n_ner):
        _ = extractor.extract_all(sample_text)
    t_ner = (time.perf_counter() - t0) / n_ner * 1000.0
    results["entity_extraction_ms"] = round(t_ner, 4)
    logger.info("Entity extraction: %.4f ms / sample", t_ner)

    # 3. Relevance Classification
    classifier = PollutionClassifier()
    n_clf = 500
    t0 = time.perf_counter()
    for _ in range(n_clf):
        _ = classifier.classify_text(title="Delhi smog alert", content=sample_text)
    t_clf = (time.perf_counter() - t0) / n_clf * 1000.0
    results["classification_inference_ms"] = round(t_clf, 4)
    logger.info("Classification inference: %.4f ms / sample", t_clf)

    # 4. Semantic Embedding Generation
    emb_mgr = SemanticEmbeddingManager()
    n_emb = 50
    t0 = time.perf_counter()
    for _ in range(n_emb):
        _ = emb_mgr.encode_texts([sample_text[:200]])
    t_emb = (time.perf_counter() - t0) / n_emb * 1000.0
    results["embedding_generation_ms"] = round(t_emb, 4)
    logger.info("Embedding generation (%s): %.4f ms / sample", emb_mgr.active_method, t_emb)

    # 5. Multi-Criteria Deduplication Matching
    art_a = {
        "title": "Delhi smog alert",
        "content_normalized": sample_text,
        "reported_location": "Delhi, India",
        "published_at": "2024-11-01T10:00:00Z",
    }
    art_b = {
        "title": "Delhi air pollution updates",
        "content_normalized": sample_text,
        "reported_location": "Delhi, India",
        "published_at": "2024-11-01T12:00:00Z",
    }
    n_match = 50
    t0 = time.perf_counter()
    for _ in range(n_match):
        _ = match_event_candidates(art_a, art_b, embedding_manager=emb_mgr)
    t_match = (time.perf_counter() - t0) / n_match * 1000.0
    results["candidate_matching_ms"] = round(t_match, 4)
    logger.info("Multi-criteria candidate matching: %.4f ms / pair", t_match)

    # 6. ML Model Inference
    for model_name in ["XGBoostRegressor", "RandomForestRegressor", "LinearRegression"]:
        predictor = ModelPredictor(model_name=model_name)
        if predictor.is_loaded:
            dummy_inputs = {
                "T": 18.5,
                "RH": 55.0,
                "AH": 1.1,
                "hour": 14,
                "month": 11,
                "CO(GT)_lag_1": 2.4,
            }
            n_infer = 200
            t0 = time.perf_counter()
            for _ in range(n_infer):
                _ = predictor.predict(dummy_inputs)
            t_infer = (time.perf_counter() - t0) / n_infer * 1000.0
            results[f"{model_name}_inference_ms"] = round(t_infer, 4)
            logger.info("%s inference: %.4f ms / sample", model_name, t_infer)

    # 7. Database Operations
    repo = PollutionRepository()
    t0 = time.perf_counter()
    n_db = 100
    for _ in range(n_db):
        _ = repo.list_articles(limit=25)
    t_db = (time.perf_counter() - t0) / n_db * 1000.0
    results["sqlite_query_latency_ms"] = round(t_db, 4)
    logger.info("SQLite 25-row query: %.4f ms", t_db)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SYSTEM LATENCY BENCHMARKS")
    print("=" * 60)
    res = benchmark_pipeline_components()
    print("\nBENCHMARK SUMMARY (ms / operation):")
    for k, v in res.items():
        print(f"  {k:<35}: {v} ms")
    print("=" * 60)
