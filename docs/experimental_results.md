# Experimental Evaluation Results and Statistical Analysis

This document compiles the empirical results for:
1. Historical Pollutant Concentration Forecasting on the UCI Air Quality Dataset.
2. Core Innovation Benchmark: Multi-Criteria Event Discovery vs Baselines.
3. Residual and Error Distribution Analysis.

---

## 1. Historical Pollutant Concentration Forecasting Benchmark

### 1.1 Experimental Configuration
- **Dataset:** UCI Machine Learning Repository Air Quality Dataset (ID: 360).
- **Target Variable:** Carbon Monoxide true concentration `CO(GT)` measured in $mg/m^3$.
- **Total Dataset Size:** 9,358 hourly observations.
- **Valid Non-Null Target Rows:** 7,674 observations (after filtering -200 sentinels).
- **Train/Test Splitting Strategy:** Strictly chronological split (first 80% train, subsequent 20% test).
- **Training Partition:** 6,139 hourly observations (March 10, 2004 to January 18, 2005).
- **Test Partition:** 1,535 hourly observations (January 18, 2005 to April 4, 2005).
- **Input Predictors:** 22 engineered features (sin/cos diurnal and seasonal encodings, target lags at $t-1, t-2, t-3, t-24$, 6-hour rolling aggregations, multi-chemical metal oxide sensors `PT08.S1` through `PT08.S5`, and ambient temperature/humidity).
- **Leakage Prevention:** Missing values are imputed using feature medians computed strictly on the training partition and transformed onto the test partition.

### 1.2 Model Comparison on Identical Chronological Test Set

| Model Architecture | Hyperparameters | Train MAE | Train RMSE | Test MAE ($mg/m^3$) | Test RMSE ($mg/m^3$) | Test $R^2$ Score | Model Selection |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Standard Scaler, OLS | 0.4082 | 0.5841 | 0.4412 | 0.6284 | 0.8120 | Linear Baseline |
| **Random Forest Regressor**| 100 trees, max depth 12 | 0.1754 | 0.2562 | 0.3245 | 0.4718 | 0.8942 | Non-linear Bagging |
| **XGBoost Regressor** | 100 trees, lr=0.08, depth 5| 0.1983 | 0.2814 | **0.3108** | **0.4521** | **0.9031** | **Selected Best (Lowest RMSE)** |

### 1.3 Key Observations
1. **Ensemble Superiority:** Both non-linear tree ensembles substantially outperform the linear model, reducing test RMSE from $0.6284\ mg/m^3$ to $0.4521\ mg/m^3$ (a 28.0% error reduction).
2. **Variance Explanation:** XGBoost explains over $90.3\%$ of the variance in held-out future Carbon Monoxide concentrations ($R^2 = 0.9031$).
3. **Generalization:** The gap between training RMSE ($0.2814$) and test RMSE ($0.4521$) demonstrates healthy model generalization without severe overfitting across the seasonal boundary.

---

## 2. Residual and Error Diagnostics (XGBoost Regressor)

Statistical analysis of prediction residuals ($e_i = y_i - \hat{y}_i$) on the 1,535 test observations:

| Diagnostic Metric | Value ($mg/m^3$) | Physical Interpretation |
| :--- | :--- | :--- |
| **Mean Residual (Bias)** | $+0.0214$ | Virtually zero systematic bias; model neither consistently over- nor under-predicts. |
| **Residual Standard Deviation** | $0.4516$ | High precision around true atmospheric concentration. |
| **Median Absolute Error** | $0.2180$ | 50% of all hourly predictions are accurate to within $0.22\ mg/m^3$. |
| **95th Percentile Absolute Error** | $0.9420$ | 95% of all predictions deviate by less than $0.95\ mg/m^3$. |
| **Maximum Observed Absolute Error** | $3.1250$ | Extreme peak errors occur during sudden anomalous episodic spikes. |

---

## 3. Core Innovation: Duplicate-Resistant Event Discovery Benchmark

### 3.1 Benchmark Dataset
Evaluated on a reviewed testbed comprising 10 news reports representing 5 distinct ground-truth environmental incidents:
- **Event 1:** Delhi post-Diwali hazardous smog spike (3 reports, including 2 syndicated wire clones).
- **Event 2:** Punjab agricultural stubble burning surge (2 reports).
- **Event 3:** Mumbai industrial chemical vapour leak (2 reports).
- **Event 4:** Delhi winter inversion smog episode 3 weeks later (2 reports — distinct temporal event).
- **Event 5:** Ghaziabad municipal landfill fire (1 report).

### 3.2 Quantitative Evaluation Comparison

| Evaluation Method | Discovered Events | Ground Truth Events | Pairwise Precision | Pairwise Recall | Pairwise F1 Score | Systematic Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1: Keyword Matching** | 10 | 5 | 0.0000 | 0.0000 | 0.0000 | Fails to cluster; 100% incident count inflation from wire syndication. |
| **Baseline 2: Semantic Similarity Only** | 6 | 5 | 0.6250 | 0.8333 | 0.7143 | Falsely merges Mumbai and Delhi reports due to shared generic smog phrasing. |
| **Proposed Multi-Criteria Discovery** | **5** | **5** | **1.0000** | **1.0000** | **1.0000** | **Perfect alignment; collapses duplicates while preserving separate cities and dates.** |

### 3.3 Systematic Failure Mode Analysis
1. **Why Baseline 1 Fails:** By counting every article that mentions "air pollution", articles syndicated across multiple news outlets (e.g. PTI copy published by *The Hindu* and *Hindustan Times*) are treated as independent physical incidents.
2. **Why Baseline 2 Fails:** Text vector cosine similarity between an article discussing "severe hazardous smog in Delhi" and "toxic chemical smog in Mumbai" exceeds $0.74$. Without geographic boundary constraints, the semantic clusterer collapses them into a single non-existent national incident.
3. **How the Proposed Engine Solves This:** Enforces strict geographic compatibility (rejects conflicting cities) and a 72-hour temporal event window (preventing the Nov 1 and Nov 25 Delhi smog episodes from merging).
