"""UCI Air Quality Dataset Acquisition and Validation Script.

Downloads the official UCI Air Quality dataset (ID: 360) directly from the UCI ML
Repository, extracts AirQualityUCI.csv into data/raw/, and runs validation checks
to confirm 9,358 hourly observations and structure.
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import requests

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.forecasting.dataset_loader import load_and_validate_uci_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_zip_recursively(zip_bytes: bytes, target_csv_path: Path) -> bool:
    """Extract AirQualityUCI.csv from outer or nested zip archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer_zip:
        for name in outer_zip.namelist():
            if name.endswith("AirQualityUCI.csv"):
                with outer_zip.open(name) as src, open(target_csv_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                logger.info("Directly extracted %s to %s", name, target_csv_path)
                return True
            elif name.endswith(".zip"):
                # Nested zip found (common in UCI distributions)
                inner_bytes = outer_zip.read(name)
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
                    for inner_name in inner_zip.namelist():
                        if inner_name.endswith("AirQualityUCI.csv"):
                            with inner_zip.open(inner_name) as src, open(target_csv_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            logger.info("Extracted %s from inner archive %s", inner_name, name)
                            return True
    return False


def download_uci_dataset(force: bool = False) -> bool:
    """Download and extract the UCI Air Quality dataset."""
    target_csv = config.UCI_RAW_CSV_PATH
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if target_csv.exists() and not force:
        logger.info("Dataset already exists at %s. Validating structure...", target_csv)
        try:
            df, meta = load_and_validate_uci_dataset(target_csv)
            logger.info(
                "Validation passed: %d observations found (Expected size verified: %s)",
                meta["total_observations"],
                meta["is_expected_size"],
            )
            return True
        except Exception as e:
            logger.warning("Existing CSV failed validation (%s). Re-downloading...", e)

    url = config.UCI_DATASET_URL
    logger.info("Downloading UCI Air Quality dataset from %s ...", url)

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            logger.error("Download failed with HTTP status %d", response.status_code)
            print_manual_instructions()
            return False

        extracted = extract_zip_recursively(response.content, target_csv)
        if not extracted:
            logger.error("Could not locate 'AirQualityUCI.csv' inside downloaded archive.")
            print_manual_instructions()
            return False

        logger.info("Successfully saved dataset to %s", target_csv)
        df, meta = load_and_validate_uci_dataset(target_csv)
        logger.info(
            "Validation successful: %d observations from %s to %s.",
            meta["total_observations"],
            meta["start_time"],
            meta["end_time"],
        )
        return True

    except requests.RequestException as e:
        logger.error("Network error during download: %s", e)
        print_manual_instructions()
        return False
    except Exception as e:
        logger.error("Unexpected error acquiring dataset: %s", e)
        print_manual_instructions()
        return False


def print_manual_instructions() -> None:
    """Print clear instructions for manual dataset download."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Visit the official UCI ML Repository page:")
    print("   https://archive.ics.uci.edu/dataset/360/air+quality")
    print("2. Download the dataset zip file or direct 'AirQualityUCI.csv'.")
    print(f"3. Place the file 'AirQualityUCI.csv' into: {config.RAW_DATA_DIR}/AirQualityUCI.csv")
    print("4. Re-run this script or the application.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and validate UCI Air Quality Dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download even if CSV exists")
    args = parser.parse_args()

    success = download_uci_dataset(force=args.force)
    sys.exit(0 if success else 1)
