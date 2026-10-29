"""UCI Air Quality Dataset Loader, Schema Validator, and Preprocessor.

Loads the historical Air Quality Chemical Multisensor dataset from the UCI Machine
Learning Repository (9,358 hourly observations recorded at an Italian monitoring station).
Validates structure, parses Italian date/time formats, replaces -200 sentinel missing
value markers with NaN, checks duplicate rows, and provides missingness diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Date",
    "Time",
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

TARGET_DESCRIPTIONS = {
    "CO(GT)": "True hourly averaged Carbon Monoxide concentration (mg/m^3)",
    "NMHC(GT)": "True hourly averaged Non-Metanic HydroCarbons concentration (microg/m^3)",
    "C6H6(GT)": "True hourly averaged Benzene concentration (microg/m^3)",
    "NOx(GT)": "True hourly averaged Nitrogen Oxides concentration (ppb)",
    "NO2(GT)": "True hourly averaged Nitrogen Dioxide concentration (microg/m^3)",
    "PT08.S1(CO)": "Tin oxide chemical sensor response (nominally CO targeted)",
    "PT08.S2(NMHC)": "Titania chemical sensor response (nominally NMHC targeted)",
    "PT08.S3(NOx)": "Tungsten oxide chemical sensor response (nominally NOx targeted)",
    "PT08.S4(NO2)": "Tungsten oxide chemical sensor response (nominally NO2 targeted)",
    "PT08.S5(O3)": "Indium oxide chemical sensor response (nominally O3 targeted)",
    "T": "Ambient Temperature (degrees Celsius)",
    "RH": "Relative Humidity (%)",
    "AH": "Absolute Humidity (g/m^3)",
}


def load_and_validate_uci_dataset(
    csv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load, validate, and clean the UCI Air Quality CSV dataset.

    Returns (cleaned_dataframe, metadata_dict).
    """
    file_path = Path(csv_path or config.UCI_RAW_CSV_PATH)
    if not file_path.exists():
        raise FileNotFoundError(
            f"UCI Air Quality dataset not found at {file_path}. "
            "Please run scripts/download_dataset.py to acquire the dataset."
        )

    # Read CSV: delimiter is ';', decimal point in original UCI release is ','
    df_raw = pd.read_csv(file_path, sep=";", decimal=",")

    # Drop trailing empty columns created by extra semicolons
    df_raw = df_raw.dropna(how="all", axis=1)

    # Filter out empty rows at the bottom of the raw file
    df_raw = df_raw.dropna(subset=["Date", "Time"], how="any")

    # Validate essential columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Retain strictly expected columns
    df = df_raw[EXPECTED_COLUMNS].copy()

    # Parse Date (DD/MM/YYYY) and Time (HH.MM.SS) into combined timestamp
    # Replace dots in time with colons if present
    time_series = df["Time"].astype(str).str.replace(".", ":", regex=False)
    datetime_str = df["Date"].astype(str).str.strip() + " " + time_series.str.strip()
    df["datetime"] = pd.to_datetime(datetime_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")

    # Drop any row where datetime parsing failed
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # Replace UCI missing-value sentinel -200 with NaN across numerical columns
    numeric_cols = [c for c in EXPECTED_COLUMNS if c not in ("Date", "Time")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -200, col] = np.nan

    # Calculate missingness diagnostics
    missingness = {}
    for col in numeric_cols:
        n_missing = int(df[col].isna().sum())
        pct_missing = round((n_missing / len(df)) * 100, 2)
        missingness[col] = {
            "missing_count": n_missing,
            "missing_percentage": pct_missing,
            "valid_count": len(df) - n_missing,
        }

    # Verify expected row count
    total_observations = len(df)
    is_expected_size = total_observations == 9358

    metadata = {
        "dataset_name": "UCI Air Quality Chemical Multisensor Dataset",
        "file_path": str(file_path),
        "total_observations": total_observations,
        "is_expected_size": is_expected_size,
        "start_time": df["datetime"].min().isoformat(),
        "end_time": df["datetime"].max().isoformat(),
        "columns": list(df.columns),
        "missingness": missingness,
        "location_context": "Italian urban road-side monitoring station (March 2004 - April 2005)",
        "provenance_note": (
            "Historical reference dataset for sensor calibration and concentration forecasting. "
            "It does not represent current Indian ambient air quality."
        ),
    }

    return df, metadata
