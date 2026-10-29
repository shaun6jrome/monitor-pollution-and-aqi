"""Configuration module for the Pollution Intelligence Platform.

Centralizes paths, database settings, API parameters, NLP thresholds,
and model hyperparameters. Loads configuration from environment variables
where applicable.
"""

from __future__ import annotations

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Ensure essential directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MODELS_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database
DEFAULT_DB_PATH = DATA_DIR / "pollution_intelligence.db"
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", str(DEFAULT_DB_PATH)))

# GDELT News Ingestion
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_REQUEST_TIMEOUT = int(os.getenv("GDELT_REQUEST_TIMEOUT", "20"))
GDELT_CACHE_TTL_SECONDS = int(os.getenv("GDELT_CACHE_TTL_SECONDS", "3600"))
GDELT_MAX_RECORDS = int(os.getenv("GDELT_MAX_RECORDS", "50"))

DEFAULT_POLLUTION_QUERIES = [
    "air pollution",
    "industrial emissions",
    "factory smoke",
    "construction dust",
    "crop burning",
    "vehicle emissions",
    "waste burning",
    "particulate pollution",
]

# Official Air Quality Portal / CPCB API
DATA_GOV_IN_API_KEY = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
CPCB_API_BASE_URL = os.getenv(
    "CPCB_API_BASE_URL",
    "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
)
CPCB_PORTAL_URL = "https://airquality.cpcb.gov.in/AQI_India/"

# NLP and Event Discovery
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.68"))
TEMPORAL_WINDOW_HOURS = 72  # Hours within which events can be considered co-occurring

EVENT_CATEGORIES = [
    "industrial_emissions",
    "vehicular_pollution",
    "construction_dust",
    "crop_burning",
    "waste_burning",
    "wildfire_smoke",
    "general_complaint",
    "other_uncertain",
]

# UCI Air Quality Dataset
UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
)
UCI_RAW_ZIP_PATH = RAW_DATA_DIR / "air_quality.zip"
UCI_RAW_CSV_PATH = RAW_DATA_DIR / "AirQualityUCI.csv"

# Target pollutant for regression
DEFAULT_FORECAST_TARGET = "CO(GT)"
RANDOM_STATE = 42
