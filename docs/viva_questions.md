# Oral Examination (Viva Voce) Preparation Guide

This guide contains 25 comprehensive viva examination questions with model answers designed for an undergraduate Data Science Ecosystem (DSE) viva voce defense.

---

### Q1: What is unstructured data, and how does it differ from structured data?
**Answer:** Structured data resides in fixed schemas with predefined data types, tabular formats, and explicit relational keys (e.g. SQL databases, sensor time-series tables with columns like Timestamp, Temperature, and PM2.5). Unstructured data lacks a predefined conceptual data model or tabular organization; it comprises natural language text, news articles, press releases, images, or audio. In this project, GDELT news text represents unstructured data, which requires natural language processing (tokenization, entity extraction, and semantic vectorization) before it can be organized into structured event records.

---

### Q2: Why is news text useful for environmental pollution monitoring alongside physical sensors?
**Answer:** Physical CAAQMS sensors measure *concentrations* (e.g. $450\ \mu g/m^3$ of PM2.5) with high precision but cannot provide *causal context*. A sensor cannot explain whether a sudden spike was triggered by an illegal factory boiler blowdown, a municipal landfill fire, or agricultural stubble burning. Unstructured news reports supply immediate contextual evidence, identifying responsible facilities, human activities, geographic landmarks, and administrative actions that physical point monitors cannot measure directly.

---

### Q3: How is the GDELT DOC 2.0 API accessed and what are its key parameters?
**Answer:** The GDELT DOC 2.0 API (`https://api.gdeltproject.org/api/v2/doc/doc`) is accessed via standard HTTP GET requests. Key parameters include:
- `query`: Boolean search strings (e.g. `"air pollution"`, `"industrial emissions"`).
- `mode`: Set to `ArtList` to return article-level metadata.
- `maxrecords`: Number of records returned per query (up to 250).
- `format`: Set to `json` for programmatic parsing.
- `sort`: Set to `DateDesc` to retrieve the latest publications.

---

### Q4: How is the UCI Air Quality dataset used in this project?
**Answer:** The UCI dataset serves as the empirical historical sensor benchmark for regression modeling. It contains 9,358 hourly observations from an array of 5 metal oxide chemical sensors alongside temperature and humidity recorded in an Italian city between March 2004 and April 2005. It is used to calibrate and compare predictive algorithms (Linear Regression, Random Forest, XGBoost) to forecast true Carbon Monoxide concentrations (`CO(GT)`).

---

### Q5: Why are the UCI dataset, GDELT news, and CPCB feeds not interchangeable?
**Answer:** They occupy completely distinct temporal, geographic, and domain spaces:
1. **UCI Air Quality:** Historical sensor calibration data from Italy (2004–2005).
2. **GDELT News:** Contemporary global unstructured news stream (2024–present) describing episodic incidents.
3. **CPCB / data.gov.in:** Live official ambient monitoring station observations across Indian cities.
Conflating them—such as claiming that contemporary Indian news predicts 2004 Italian sensor readings—would constitute severe scientific fabrication.

---

### Q6: What is TF-IDF and how is it used in the platform?
**Answer:** Term Frequency-Inverse Document Frequency (TF-IDF) is a statistical measure that evaluates how important a word is to a document within a collection. Term Frequency measures local frequency, while Inverse Document Frequency penalizes universally common words:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$$
In this project, TF-IDF transforms article headlines and normalized text into numerical feature vectors for the baseline pollution relevance classifier (Logistic Regression) and serves as an automated embedding fallback.

---

### Q7: What are sentence embeddings and how do they differ from bag-of-words representations?
**Answer:** Bag-of-words and TF-IDF represent text as sparse, high-dimensional vectors based solely on exact word matches, failing to capture semantic synonyms (e.g. "toxic smog" and "hazardous haze" share zero vocabulary). Sentence embeddings (e.g., from Sentence Transformers such as `all-MiniLM-L6-v2`) project entire sentences into a dense, continuous semantic vector space (e.g. 384 dimensions) where texts with similar conceptual meaning reside close to one another regardless of exact wording.

---

### Q8: What is cosine similarity and how is it calculated?
**Answer:** Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space. For two vectors $u$ and $v$, it is calculated as:
$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
When embedding vectors are pre-normalized to unit Euclidean length ($\|u\|_2 = 1$), cosine similarity simplifies directly to the dot product $u \cdot v$, yielding a score in $[-1, 1]$ (or $[0, 1]$ for non-negative text representations).

---

### Q9: Why is semantic similarity alone insufficient for pollution event matching?
**Answer:** Two news articles describing "severe hazardous smog and poor air quality" will yield very high cosine similarity ($>0.80$) even if one occurred in **Mumbai** and the other in **Delhi**, or if one occurred in **November 2024** and the other occurred **four weeks later**. Semantic similarity captures *topic similarity*, not *incident identity*. Merging articles based solely on semantic similarity creates false-positive merges across distinct physical events.

---

### Q10: How does the proposed system detect duplicate and syndicated news reports?
**Answer:** Duplicate and syndicated reporting is detected by combining cross-domain similarity scoring with agency attribution analysis. When two articles from distinct domains exhibit near-verbatim text similarity ($\ge 0.85$) or shared wire credits (e.g. PTI, ANI, Reuters), the system flags them as syndicated copies of the same underlying story, preventing them from being counted as separate incidents.

---

### Q11: How does provenance analysis improve evidence quality?
**Answer:** Provenance traces the origins and reporting independence of supporting evidence. If five articles about a factory leak all originate from a single press release or wire feed copied verbatim across five aggregator websites, treating them as five independent confirmations is a false corroboration error. The provenance engine distinguishes true independent investigative reporting from syndicated repetition and adjusts the event's confidence score accordingly.

---

### Q12: What is the scientific difference between AQI and pollutant concentration?
**Answer:** 
- **Pollutant Concentration:** An objective, physically measured quantity expressing the mass or volume of a specific chemical species per unit volume of air (e.g. Carbon Monoxide in $mg/m^3$, PM2.5 in $\mu g/m^3$, NOx in $ppb$).
- **Air Quality Index (AQI):** A dimensionless, mathematically transformed composite index designed for public health risk communication. It maps non-linear pollutant concentrations onto standardized categories (e.g. 0–50 Good, 401–500 Severe) via country-specific sub-index piecewise linear breakpoints.

---

### Q13: Why compare Linear Regression, Random Forest, and XGBoost?
**Answer:** They represent three foundational algorithmic paradigms in predictive machine learning:
1. **Linear Regression:** A parametric, linear baseline providing maximum interpretability and establishing baseline performance.
2. **Random Forest:** A non-parametric bagging ensemble of decision trees that captures non-linear sensor response curves and interactions while resisting overfitting.
3. **XGBoost:** A gradient boosting ensemble that fits trees sequentially to pseudo-residuals, typically achieving the lowest prediction error on tabular sensor features.

---

### Q14: Why is a chronological train/test split mandatory for time-series forecasting?
**Answer:** Time-series observations exhibit temporal autocorrelation. A random K-fold shuffle would allow future observations ($t+1$) to appear in the training set while predicting past observations ($t$), leading to catastrophic *temporal lookahead data leakage*. Evaluation metrics from random splits are artificially optimistic and collapse in real-world deployment. A chronological split ensures the model is trained strictly on the past and tested on the unseen future.

---

### Q15: What is data leakage and how is it prevented in this implementation?
**Answer:** Data leakage occurs when information from outside the training partition influences model training. In this platform, leakage is prevented by:
1. Enforcing chronological train/test splitting.
2. Constructing lag and rolling features strictly using historical observations (`closed='left'`).
3. Calculating imputation medians strictly on the training partition and applying those saved parameters to the test partition.

---

### Q16: What do MAE and RMSE measure, and when is RMSE preferred?
**Answer:**
- **Mean Absolute Error (MAE):** The average magnitude of absolute errors: $\frac{1}{n}\sum |y_i - \hat{y}_i|$. It treats all errors proportionally and is robust to outliers.
- **Root Mean Squared Error (RMSE):** The square root of average squared errors: $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$. Because errors are squared before averaging, RMSE penalizes large prediction errors more severely. In air quality forecasting, missing a dangerous toxic pollutant spike is costly, making RMSE the preferred criterion for model selection.

---

### Q17: How is model selection performed in the ML Laboratory?
**Answer:** All candidate models are trained on the identical chronological training partition and evaluated against identical held-out test rows. The model that achieves the lowest held-out test RMSE is programmatically selected as `is_best_model=1` and stored in SQLite.

---

### Q18: How does the real-time workflow handle GDELT API failures or network downtime?
**Answer:** The `GDELTClient` implements configurable HTTP timeouts (default 20s), size limits, and local JSON disk caching (`TTL=3600s`). When the API is unreachable or returns HTTP errors (e.g. 503), the client logs the failure in `retrieval_logs`, gracefully serves the latest unexpired cached records, and visibly flags the data as cached or stale in the UI.

---

### Q19: What is the project's central proposed innovation?
**Answer:** The core innovation is **multi-criteria, duplicate-resistant pollution event discovery**. Unlike standard keyword counts that inflate incident frequency by counting every syndicated article, or semantic similarity models that falsely merge unrelated incidents across cities, the proposed engine integrates semantic cosine similarity, geographic gazetteer verification, a 72-hour temporal event window, and wire syndication auditing into a unified incident discovery pipeline.

---

### Q20: What experimental evidence demonstrates that the innovation is effective?
**Answer:** The innovation is benchmarked against a reviewed testbed containing syndicated wire duplicates, distinct city incidents, and separated temporal episodes. Baseline 1 (keyword count) overcounts events by 100%, and Baseline 2 (semantic only) falsely merges distinct episodes in Mumbai and Delhi. The proposed multi-criteria system achieves higher precision and F1 score by correctly collapsing duplicates while preserving geographically and temporally distinct incidents.

---

### Q21: What are the primary limitations of the current implementation?
**Answer:** 
1. **Spatial/Temporal Non-Alignment:** The UCI dataset reflects 2004–2005 Italian road measurements, while GDELT news represents contemporary incidents; they cannot be joined for direct cross-source training without scientific invalidity.
2. **Text Geocoding Limits:** Locations are extracted as textual entities (e.g. "Delhi") without precise street-level GPS coordinates.
3. **Media Reporting Bias:** News coverage favors capital cities and sensational crises over continuous low-level industrial emissions.

---

### Q22: Why does the prediction interface restrict input ranges and validate domain bounds?
**Answer:** Machine learning models are valid only within the domain boundaries established during training. Allowing unconstrained user inputs (e.g. negative sensor resistance or temperatures of 200°C) results in nonsensical extrapolation. The interface validates inputs against physical physical bounds and explicitly documents when training medians are substituted.

---

### Q23: How does the system handle missing values in the UCI dataset?
**Answer:** The dataset marks missing values with a $-200$ sentinel code. During ingestion, all $-200$ values are converted to standard `NaN`. Rows with missing target values are omitted from supervised training, while missing predictor variables are imputed using training medians fitted exclusively on the training split.

---

### Q24: What ethical and licensing considerations apply to unstructured news extraction?
**Answer:** News extraction must respect publisher intellectual property, copyright, and server capacity. The platform respects robots exclusions, HTTP 401/403 paywalls, sets explicit request timeouts, stores original canonical URLs and domain attributions, and extracts readable text strictly from permitted public web pages.

---

### Q25: How would this system be deployed in an operational environmental regulatory agency?
**Answer:** In an operational deployment, continuous official CAAQMS station feeds would be paired with local municipal news streams and social complaints from the same metropolitan jurisdiction. The event discovery engine would cluster complaints to generate automated anomaly alerts, dispatching field inspectors to investigate unmonitored emission sources.
