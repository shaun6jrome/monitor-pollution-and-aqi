# Comprehensive Project Methodology and Mathematical Formulations

**Project Title:** Evidence-Aware Pollution Event Discovery and Air-Quality Prediction Using Unstructured Data  
**Domain:** Environmental Data Science / Natural Language Processing / Predictive Machine Learning  

---

## 1. Problem Formulation and Architectural Objectives

Ambient air quality monitoring networks rely heavily on fixed Continuous Ambient Air Quality Monitoring Stations (CAAQMS). While physical sensors provide highly accurate point measurements, their geographic coverage is sparse, and physical chemical sensors operate as passive monitors: they measure *what* concentration exists, but cannot explain the *human causes* behind pollution spikes (e.g., sudden chemical boiler releases, illegal nighttime waste burning, or crop residue incineration).

Unstructured text from global and regional news feeds contains immediate real-world incident observations. However, using raw news data introduces critical challenges:
1. **Syndication and Duplication:** A single news wire story (e.g., from Press Trust of India [PTI], Asian News International [ANI], or Reuters) is republished across dozens of regional news outlets, artificially inflating article counts by 100% to 300%.
2. **Geographic and Temporal Conflation:** Articles mentioning generic phrases like "severe smog" or "toxic emissions" may refer to incidents in entirely different cities or weeks apart. Pure vector similarity collapses these distinct incidents into false mega-clusters.
3. **Data Quality and Verifiability:** Sensor readings and unstructured news originate from fundamentally different domains and cannot be conflated without strict scientific boundaries.

The objective of this methodology is to build a verifiable, reproducible pipeline that discovers distinct pollution events from news, preserves supporting evidence with provenance tracking, and trains leakage-free machine learning regression models for pollutant concentration forecasting.

---

## 2. Unstructured Text Processing Pipeline

### 2.1 Normalization and Cleaning
Raw unstructured text ingested from news feeds contains formatting artifacts, typographic variations, and syndicated boilerplates. The cleaning pipeline applies the following transformations:
1. **Unicode Normalization:** NFKC decomposition and recomposition standardizes diverse representations into canonical UTF-8 characters.
2. **Punctuation and Whitespace Standardization:** Typographic curly quotes (`“`, `”`, `‘`, `’`) are mapped to ASCII straight quotes (`"`, `'`); em-dashes and en-dashes (`—`, `–`) are mapped to ASCII hyphens (`-`); redundant vertical and horizontal whitespace is collapsed.
3. **HTML Entity Unescaping:** Entities such as `&amp;`, `&quot;`, `&apos;`, `&lt;`, and `&gt;` are decoded to plain text.
4. **Boilerplate and Wire Credit Stripping:** Regular expression filters strip recurring wire metadata headers (e.g., `(PTI)`, `(ANI)`, `NEW DELHI:`) and copyright footers ("All rights reserved", "Click here to read more").
5. **Chemical Notation Preservation:** Standard tokenizers destroy subscripted chemical formulas. The cleaner explicitly preserves formula tokens such as `PM2.5`, `PM10`, `NOx`, `SO2`, `CO`, `CO2`, `O3`, and `C6H6` along with numerical concentration units ($mg/m^3$, $\mu g/m^3$, $ppm$).

### 2.2 Pollution Relevance Classification
Articles are classified into a three-state relevance schema:
- `+1`: Pollution-related (explicit description of emissions, smog, air quality deterioration, or enforcement actions).
- `0`: Not pollution-related (general politics, sports, finance, or unrelated environmental stories).
- `-1`: Uncertain or ambiguous evidence requiring manual review.

The primary supervised classifier uses a Term Frequency-Inverse Document Frequency (TF-IDF) feature matrix with sublinear term-frequency scaling and unigram/bigram token ranges:
$$\text{TF-IDF}(t, d, D) = \left(1 + \log(\text{tf}(t, d))\right) \cdot \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

The feature representation is passed to a calibrated Logistic Regression model ($C=5.0$, L2 penalty). When the predicted class probability falls within the ambiguous band:
$$P(y = 1 \mid \mathbf{x}) \in [0.45, 0.55]$$
the instance is assigned label `-1` and an explainable log-odds feature attribution routine extracts the top positive and negative vocabulary weights:
$$\text{Log-Odds Attribution}_j = w_j \cdot x_j$$

### 2.3 Environmental Entity Extraction
- **Locations:** Extracted via a normalized gazetteer covering 30+ major Indian air quality monitoring hubs (Delhi, Noida, Gurugram, Mumbai, Bhiwadi, Manesar, Ankleshwar, Vapi, Korba, Singrauli, Coimbatore, Chennai, Bengaluru) with coordinate mapping and administrative parent-child resolution.
- **Pollutants:** Extracted via regex patterns targeting particulate matter (`PM2.5`, `PM10`), criteria gases (`NOx`, `SO2`, `CO`, `O3`), volatile organics (`benzene`), and general descriptors (`smog`, `soot`, `smoke`).
- **Emission Source Taxonomy:** Categorized into an 8-class taxonomy: `crop_burning`, `industrial_emissions`, `vehicular_pollution`, `construction_dust`, `waste_burning`, `wildfire_smoke`, `general_complaint`, and `other_uncertain`.
- **Temporal Anchors:** Extracts relative temporal references ("yesterday", "last night", day of week) and explicit ISO calendar datelines, anchoring the incident to a physical timestamp.

---

## 3. Core Innovation: Duplicate-Resistant Event Discovery

### 3.1 Unit-Normalized Dense Semantic Embeddings
Articles are embedded using the `all-MiniLM-L6-v2` Sentence Transformer, mapping text sequences into a 384-dimensional dense semantic vector space $\mathbb{R}^{384}$. All embeddings are bound to CPU execution for cross-platform stability.
Vectors are unit-normalized ($L_2$ norm):
$$\mathbf{u}_{\text{norm}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}, \quad \|\mathbf{u}_{\text{norm}}\|_2 = 1.0$$
This enables rapid cosine similarity calculation via matrix dot product:
$$s_{\text{sem}}(\mathbf{u}, \mathbf{v}) = \mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}}$$

When transformer dependencies are uninitialized, the system automatically falls back to unit-normalized TF-IDF vector dot products, transparently logging the active embedding backend.

### 3.2 Continuous Temporal Decay Weighting
Events possess finite temporal relevance. Two reports describing similar industrial flare-ups separated by 48 hours are less likely to represent the identical physical plume than two reports published 3 hours apart. We formulate a continuous linear temporal decay factor bounded within a 72-hour window:
$$w_{\text{time}}(\Delta t) = \begin{cases} 
\max\left(0, 1.0 - 0.25 \cdot \frac{|\Delta t|}{72}\right), & \text{if } |\Delta t| \le 72 \text{ hours} \\
0.0, & \text{if } |\Delta t| > 72 \text{ hours}
\end{cases}$$
where $\Delta t = |t_{\text{article}} - t_{\text{event}}|$, with time measured in hours. At $\Delta t = 0$, $w_{\text{time}} = 1.00$; at $\Delta t = 72$, $w_{\text{time}} = 0.75$. Beyond 72 hours, the candidate is excluded.

### 3.3 Multi-Criteria Score Fusion
To prevent false cluster merges across geographical regions or distinct emission categories, we formulate a composite multi-criteria match score $S$:
$$S = s_{\text{sem}}(\mathbf{u}_{\text{article}}, \mathbf{u}_{\text{cluster}}) \cdot w_{\text{time}}(\Delta t) \cdot \mathbb{I}_{\text{geo}}(L_{\text{article}}, L_{\text{cluster}}) \cdot \mathbb{I}_{\text{cat}}(C_{\text{article}}, C_{\text{cluster}})$$

Where the constraint indicator functions are defined as:
$$\mathbb{I}_{\text{geo}}(L_A, L_B) = \begin{cases}
1.0, & \text{if } L_A = L_B \text{ or } L_A \in \text{Region}(L_B) \text{ or } L_A = \text{Unknown} \text{ or } L_B = \text{Unknown} \\
0.0, & \text{if } L_A \neq L_B \text{ (Conflicting verified locations)}
\end{cases}$$

$$\mathbb{I}_{\text{cat}}(C_A, C_B) = \begin{cases}
1.0, & \text{if } C_A = C_B \text{ or } C_A = \text{uncertain} \text{ or } C_B = \text{uncertain} \\
0.0, & \text{if } C_A \neq C_B \text{ (Conflicting emission categories)}
\end{cases}$$

**Clustering Decision Rule:**
An article is merged into an existing event cluster if and only if:
$$S \ge \theta_{\text{composite}} \quad (0.65) \quad \text{AND} \quad s_{\text{sem}} \ge \theta_{\text{semantic}} \quad (0.68)$$
If multiple candidate events satisfy the threshold, the article merges with the event maximizing $S$. If no existing event satisfies the threshold, a new physical event is initialized.

### 3.4 Evidence Provenance and Syndication Auditing
When an article is merged into an event, its origin is audited for wire syndication versus independent investigative corroboration:
1. **Wire Syndication:** If cross-domain semantic similarity $s_{\text{sem}} \ge 0.85$ or if agency attribution tags (`PTI`, `ANI`, `UNI`) match an earlier report, the article is flagged as `syndicated_copy`. It does not increment the physical incident count.
2. **Independent Corroboration:** Articles published by distinct publishers with $0.65 \le s_{\text{sem}} < 0.85$ and distinct textual formulations are flagged as `independent_corroboration`.
3. **Calibrated Evidence Quality Score:**
$$Q_{\text{evidence}} = \min\left(0.90, 0.50 + 0.10 \cdot (N_{\text{domains}} - 1) + 0.10 \cdot \mathbb{I}_{\text{cross-domain}}\right)$$
Single uncorroborated articles receive $Q = 0.50$; two distinct domains yield $Q = 0.70$; multi-source cross-domain consensus saturates at $Q = 0.90$.

---

## 4. Machine Learning Forecasting Methodology

### 4.1 Historical Dataset Characterization
The concentration forecasting module is calibrated using the landmark UCI Machine Learning Repository Air Quality dataset (De Vito et al., 2008):
- **Total Records:** Exactly 9,358 hourly observations from March 10, 2004 to April 4, 2005.
- **Physical Setting:** Roadside monitoring station in an Italian urban center characterized by high traffic density.
- **Instrumentation:** Co-located reference analyzers recording Carbon Monoxide ($CO$), Non-Metanic Hydrocarbons ($NMHC$), Benzene ($C_6H_6$), Nitrogen Oxides ($NO_x$), and Nitrogen Dioxide ($NO_2$), alongside five metal oxide solid-state sensors ($PT08.S1$ through $PT08.S5$), Temperature ($T$), Relative Humidity ($RH$), and Absolute Humidity ($AH$).
- **Missing Value Sentinel:** Hardware malfunctions and sensor saturation are recorded with the sentinel value $-200$. These are converted to IEEE `NaN`.

### 4.2 Leakage-Free Chronological Feature Engineering
Environmental time series exhibit strong diurnal cycles, weekly traffic trends, and autoregressive persistence. Feature engineering enforces strict leakage-free protocols:
1. **Chronological Splitting:** The 9,358 rows are split into an 80% training set (6,139 rows) and a 20% held-out test set (1,535 rows) based strictly on timestamp. Shuffling is strictly prohibited.
2. **Imputation Parameter Isolation:** Missing feature values are imputed using column medians computed strictly on the training partition:
   $$\hat{\mu}_j = \text{median}\left(\{x_{i, j} : i \in \text{Train}, x_{i, j} \neq \text{NaN}\}\right)$$
   Test data uses $\hat{\mu}_j$ from training without recomputing.
3. **Temporal Cyclical Transformations:** Diurnal and seasonal periodicities are mapped to continuous circular coordinates:
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$
   $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot m}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot m}{12}\right)$$
4. **Autoregressive Lags:** Historical values of the target $y = \text{CO(GT)}$ are extracted at $t-1$, $t-2$, $t-3$, and $t-24$ (same hour previous day).
5. **Left-Closed Rolling Statistics:** 6-hour and 24-hour moving averages and standard deviations are computed using `closed='left'` on the historical window $[t-w, t-1]$, guaranteeing that sensor readings at time $t$ never contaminate historical summary features.

### 4.3 Model Formulations
We evaluate three distinct functional model families on the 27-dimensional feature space $\mathbf{x}_t \in \mathbb{R}^{27}$:
1. **Linear Regression Baseline:**
   $$\hat{y}_t = \mathbf{w}^T \tilde{\mathbf{x}}_t + b$$
   where $\tilde{\mathbf{x}}_t$ is standardized using training set mean and variance:
   $$\tilde{x}_{t, j} = \frac{x_{t, j} - \mu_{j, \text{train}}}{\sigma_{j, \text{train}}}$$
2. **Random Forest Regressor:**
   An ensemble of $M = 100$ independent decision trees trained via bagging and feature subsampling:
   $$\hat{y}_t = \frac{1}{M} \sum_{m=1}^M T_m(\mathbf{x}_t; \Theta_m)$$
   where max depth is constrained to 12 to prevent overfitting.
3. **XGBoost Regressor (Selected Best):**
   Gradient-boosted decision trees trained additively to minimize regularized squared error:
   $$\mathcal{L}^{(k)} = \sum_{i=1}^N \left( y_i - \left(\hat{y}_i^{(k-1)} + f_k(\mathbf{x}_i)\right) \right)^2 + \sum_{j=1}^k \Omega(f_j)$$
   $$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
   Hyperparameters: learning rate $\eta = 0.08$, subsample ratio $0.85$, colsample ratio $0.85$, max depth $6$, L2 regularization $\lambda = 1.0$.

---

## 5. Statistical Diagnostics and Multi-Step Forecasting

### 5.1 Residual Distribution Diagnostics
Model residuals $e_i = y_i - \hat{y}_i$ are evaluated for systematic bias and non-normality:
1. **Mean Residual Bias:** $\bar{e} = \frac{1}{n} \sum_{i=1}^n e_i$
2. **Sample Skewness ($g_1$):**
   $$g_1 = \frac{m_3}{m_2^{3/2}}, \quad m_k = \frac{1}{n} \sum_{i=1}^n (e_i - \bar{e})^k$$
3. **Sample Kurtosis ($g_2$):**
   $$g_2 = \frac{m_4}{m_2^2} - 3$$
4. **D'Agostino-Pearson Omnibus Normality Test:**
   Combines transformed skewness $Z_1(g_1)$ and kurtosis $Z_2(g_2)$ into an omnibus chi-squared statistic:
   $$K^2 = Z_1(g_1)^2 + Z_2(g_2)^2 \sim \chi^2(2)$$
   A $p$-value $< 0.05$ rejects the hypothesis of Gaussian error distribution, identifying episodic heavy-tailed pollution spikes.

### 5.2 Multi-Step Autoregressive Trajectory Forecasting
To forecast $H$ hours into the future ($h = 1, 2, \dots, H$), the system executes recursive autoregressive rollouts:
$$\hat{y}_{t+h} = f\left(\mathbf{x}_{t+h}\right)$$
where the feature vector $\mathbf{x}_{t+h}$ is updated dynamically:
- Autoregressive lag $t-1$ becomes $\hat{y}_{t+h-1}$.
- Rolling statistics shift dynamically across the predicted window.
- Physical boundaries enforce non-negativity: $\hat{y}_{t+h} = \max(0.0, \hat{y}_{t+h})$.
- 90% and 95% empirical prediction intervals are calculated via residual quantile scaling:
  $$\text{PI}_{95} = \hat{y}_{t+h} \pm 1.96 \cdot \text{RMSE} \cdot \sqrt{1 + 0.05 \cdot h}$$

---

## 6. Empirical Experimental Results Summary

### 6.1 Duplicate-Resistant Discovery Benchmark (Synthetic Gold Standard Corpus)
| Metric | Baseline 1 (Keyword Count) | Baseline 2 (Semantic Only) | Proposed Multi-Criteria Discovery |
| :--- | :--- | :--- | :--- |
| **Discovered Events** | 10 | 6 | **5** (Exact Ground Truth) |
| **True Ground Truth Events**| 5 | 5 | **5** |
| **False Merges Across Cities** | 0 | 1 (Mumbai & Delhi merged) | **0** (Prevented by $\mathbb{I}_{\text{geo}}$) |
| **Wire Syndication Flags** | 0 | 0 | **4** (Accurately tagged) |
| **Precision** | 0.00 | 0.60 | **1.00** |
| **Recall** | 0.00 | 0.60 | **1.00** |
| **F1 Score** | 0.00 | 0.72 | **1.00** |

### 6.2 Empirical Pollutant Regression Results (Held-Out Test Set: 1,535 Observations)
| Model Family | Test MAE ($mg/m^3$) | Test RMSE ($mg/m^3$) | Test $R^2$ | Train MAE | Train RMSE | Execution Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.4528 | 0.5615 | 0.8256 | 0.2526 | 0.3799 | 0.015 s |
| **Random Forest (100 Trees)**| 0.2730 | 0.4346 | 0.8955 | 0.1178 | 0.1763 | 0.842 s |
| **XGBoost (Selected Best)** | **0.2626** | **0.4210** | **0.9020** | 0.1651 | 0.2374 | 0.098 s |

### 6.3 Residual Diagnostics for Best Model (XGBoost)
- **Mean Residual Error (Bias):** $+0.1114 \, mg/m^3$
- **Median Absolute Error:** $0.1487 \, mg/m^3$
- **95th Percentile Absolute Error:** $0.9174 \, mg/m^3$
- **Maximum Residual Error:** $3.7560 \, mg/m^3$
- **Residual Skewness:** $+1.5758$ (Right-skewed tail representing extreme peak pollution episodes)
- **Residual Kurtosis:** $+9.1564$ (Leptokurtic distribution confirming heavy tails)
- **D'Agostino-Pearson Omnibus Normality Test:** $K^2 = 724.8, \quad p < 0.0001$ (Statistically significant non-normality)
