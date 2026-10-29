# Data Storage Directory

This directory stores datasets, cached raw API responses, and processed feature sets for the pollution intelligence project.

## Directory Structure

- `raw/`: Stores pristine raw source files, including the UCI Air Quality dataset (`AirQualityUCI.csv`) and cached GDELT raw query outputs.
- `processed/`: Stores cleaned, preprocessed datasets, feature matrices, and evaluation splits.
- `cache/`: Stores transient API query cache records to prevent unnecessary repeated requests to external endpoints.

## Datasets

### 1. UCI Machine Learning Repository: Air Quality Dataset
- Source: [UCI ML Repository Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- Coverage: 9,358 hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device.
- Location: Field deployment in a significantly polluted area within an Italian city.
- Period: March 2004 to April 2005.
- Target: Sensor pollutant concentrations such as CO(GT), NMHC(GT), C6H6(GT), NOx(GT), NO2(GT).
- Important Note: This is an empirical historical monitoring dataset used for baseline and regression model benchmarking. It is distinct from live Indian ambient measurements.

### 2. GDELT DOC 2.0 API (Unstructured News Stream)
- Source: [GDELT Project DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- Content: Real-time and near-real-time global news coverage across multiple queries (air pollution, industrial emissions, crop burning, vehicle emissions, waste burning).
- Storage: Stored idempotently in the SQLite database (`pollution_intelligence.db`).
