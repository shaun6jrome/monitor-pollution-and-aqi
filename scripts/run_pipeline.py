"""End-to-End Execution Pipeline for the Pollution Intelligence Platform.

Orchestrates the complete data science workflow:
1. GDELT news retrieval and caching
2. Unstructured text cleaning, classification, and entity extraction
3. Multi-criteria duplicate-resistant event discovery
4. UCI dataset acquisition and validation
5. Chronological ML regression model training and evaluation
6. Innovation benchmark execution
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from scripts.download_dataset import download_uci_dataset
from src.forecasting.evaluate_models import format_model_comparison_table
from src.forecasting.train_models import train_and_evaluate_all_models
from src.ingestion.gdelt_client import GDELTClient
from src.intelligence.embeddings import SemanticEmbeddingManager
from src.intelligence.event_aggregator import EventAggregator
from src.intelligence.innovation_evaluator import evaluate_innovation_pipeline
from src.preprocessing.event_classifier import PollutionClassifier
from src.storage.repository import PollutionRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(force_refresh_news: bool = False) -> None:
    """Execute complete end-to-end pipeline."""
    repo = PollutionRepository()
    logger.info("Database initialized at %s", config.SQLITE_DB_PATH)

    # 1. News Ingestion
    logger.info("Stage 1: Ingesting news from GDELT DOC 2.0 API...")
    gdelt = GDELTClient(repository=repo)
    total_new = 0
    total_retrieved = 0

    queries = ["air pollution", "industrial emissions", "crop burning"]
    import time
    for idx, q in enumerate(queries):
        if idx > 0 and force_refresh_news:
            logger.info("Enforcing 6-second rate limit pause before GDELT query '%s'...", q)
            time.sleep(6)
        res = gdelt.ingest_query_results(q, max_records=25, force_refresh=force_refresh_news)
        total_retrieved += res["total_retrieved"]
        total_new += res["new_records"]
        logger.info("Query '%s': retrieved %d articles (%d new).", q, res["total_retrieved"], res["new_records"])

    # 2. Text Classification
    logger.info("Stage 2: Classifying article relevance...")
    classifier = PollutionClassifier()
    articles = repo.list_articles(limit=100)
    for art in articles:
        if art.get("is_pollution_related") == -1:
            c_res = classifier.classify_text(title=art["title"], content=art.get("content_normalized"))
            repo.update_article_classification(art["id"], c_res["label"], c_res["confidence"], c_res["method"])

    # 3. Multi-Criteria Event Discovery
    logger.info("Stage 3: Running multi-criteria event discovery...")
    emb_mgr = SemanticEmbeddingManager()
    aggregator = EventAggregator(repository=repo, embedding_manager=emb_mgr)
    disc_res = aggregator.discover_events_from_articles()
    logger.info("Discovered %d new events from %d articles.", disc_res["events_created"], disc_res["total_articles"])

    # 4. Dataset Acquisition & Machine Learning
    logger.info("Stage 4: Checking UCI Air Quality Dataset and Training Models...")
    if not config.UCI_RAW_CSV_PATH.exists():
        download_uci_dataset()

    ml_res = train_and_evaluate_all_models(repository=repo, save_models=True)
    summary_df = format_model_comparison_table(ml_res["models_summary"])

    # 5. Innovation Evaluation Benchmark
    logger.info("Stage 5: Running core innovation benchmark...")
    innov_res = evaluate_innovation_pipeline(embedding_manager=emb_mgr)

    # Print Final Pipeline Summary
    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total News Articles Ingested:    {repo.count_articles()}")
    print(f"Pollution-Related News Reports:  {repo.count_articles(only_pollution=True)}")
    print(f"Distinct Discovered Events:      {repo.count_events()}")
    print(f"Active Semantic Encoder:         {disc_res.get('active_embedding_method', getattr(emb_mgr, 'active_method', 'sentence_transformers'))}")
    print("-" * 70)
    print(f"Best Forecasting Model:          {ml_res['best_model']} (CO(GT) Target)")
    print(summary_df.to_string(index=False))
    print("-" * 70)
    print("Core Innovation: Multi-Criteria Deduplication vs Baselines")
    for row in innov_res.get("comparison_table", []):
        print(f" - {row.get('Method')}: {row.get('Discovered Events')} events (F1: {row.get('Pairwise F1')})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete pollution intelligence pipeline")
    parser.add_argument("--refresh-news", action="store_true", help="Force fresh news retrieval from GDELT API")
    args = parser.parse_args()

    run_pipeline(force_refresh_news=args.refresh_news)
