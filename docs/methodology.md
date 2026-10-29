# Project Methodology

## 1. Problem Formulation and Objectives

Ambient air pollution monitoring networks rely heavily on fixed continuous ambient air quality monitoring stations (CAAQMS). While physical sensors provide highly accurate point measurements, their geographic coverage is sparse, and sensor data alone cannot explain the *human causes* behind pollution spikes (e.g., sudden factory boiler releases, illegal nighttime waste burning, or crop residue incineration).

Unstructured text from global and regional news feeds contains immediate real-world incident observations. However, using raw news data introduces critical challenges:
1. **Syndication and Duplication:** A single news wire story (e.g. from PTI or ANI) is republished across dozens of newspapers, artificially inflating article counts.
2. **Geographic and Temporal Conflation:** Articles mentioning generic phrases like "severe smog" may refer to incidents in entirely different cities or weeks apart.
3. **Data Quality and Verifiability:** Sensor readings and unstructured news originate from fundamentally different domains and cannot be conflated without strict scientific boundaries.

The objective of this methodology is to build a verifiable pipeline that discovers distinct pollution events from news, preserves supporting evidence, and compares historical machine learning models for pollutant forecasting.

---

## 2. Unstructured Text Processing Pipeline

### 2.1 Normalization and Cleaning
- Unicode characters are decomposed and standardized via NFKC normalization.
- Curved quotation marks and typographic em-dashes are converted to ASCII equivalents.
- HTML entities (`&quot;`, `&amp;`, `&apos;`) are unescaped.
- Boilerplate advertising disclaimers ("All rights reserved", "Click here to subscribe") are removed using regular expressions.
- Chemical notations (`PM2.5`, `PM10`, `NOx`, `SO2`, `CO2`, `O3`) and numerical measurements are strictly preserved.

### 2.2 Pollution Relevance Classification
Articles are classified into:
- `1`: Pollution-related
- `0`: Not pollution-related
- `-1`: Uncertain or insufficient evidence

The primary model is a supervised TF-IDF vectorizer coupled with Logistic Regression ($C=1.0$). A curated training set of verified domain examples provides a transparent benchmark. A heuristic domain lexicon acts as an inspectable fallback when neural dependencies are uninitialized.

### 2.3 Entity Extraction
- **Locations:** Extracted via a normalized gazetteer covering major Indian air quality monitoring stations (Delhi, Noida, Gurugram, Mumbai, Punjab, etc.) with optional spaCy NER fallback.
- **Pollutants:** Extracted using case-insensitive regex patterns targeting particulate matter, nitrogen oxides, sulfur dioxide, ozone, volatile organic compounds, and toxic fumes.
- **Source Taxonomy:** Classified into 8 distinct categories (`crop_burning`, `industrial_emissions`, `vehicular_pollution`, `construction_dust`, `waste_burning`, `wildfire_smoke`, `general_complaint`, `other_uncertain`).
- **Temporal Anchors:** Identifies relative time expressions ("yesterday", "last night", day of week) and explicit calendar dates.

---

## 3. Core Innovation: Duplicate-Resistant Event Discovery

### 3.1 Embedding and Semantic Similarity
Text vectors are computed using Sentence Transformers (`all-MiniLM-L6-v2`) or an automated TF-IDF fallback. All vectors are unit-normalized ($L_2$ norm), allowing cosine similarity to be computed via dot product:

$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u_{norm} \cdot v_{norm}$$

### 3.2 Multi-Criteria Matching Engine
Two reports $A$ and $B$ are merged into a single event candidate only if all of the following criteria are satisfied:
1. **Semantic Similarity:** Cosine similarity score $\ge 0.68$.
2. **Geographic Compatibility:** Locations must match exactly or represent subregions (e.g. Noida within Uttar Pradesh). Conflicting cities (e.g. Delhi vs Mumbai) result in immediate rejection regardless of semantic similarity.
3. **Temporal Proximity:** Publication timestamps must fall within a 72-hour rolling window ($|\Delta t| \le 72$ hours).
4. **Source Category Alignment:** Reports must share compatible emission categories (e.g. stubble burning and industrial refinery emissions cannot be merged).

### 3.3 Evidence Provenance and Syndication Detection
When multiple articles support an event, the system audits cross-domain independence:
- **Wire Syndication:** Articles published across different domains with similarity $\ge 0.85$ or shared agency attribution tags (e.g. PTI, ANI, Reuters) are flagged as syndicated republication rather than independent corroboration.
- **Independent Corroboration:** Articles from distinct domains with moderate semantic similarity ($0.65 \le \text{sim} < 0.85$) and distinct text phrasing indicate independent investigations.
- **Evidence Quality Score:** Ranges from $0.50$ (single source) to $0.90$ (multi-domain independent corroboration).

---

## 4. Machine Learning Forecasting Methodology

### 4.1 Historical Dataset Context
The forecasting experiments utilize the UCI Machine Learning Repository Air Quality dataset.
- Total Observations: Exactly 9,358 hourly records.
- Location: Italian urban road-side monitoring station.
- Period: March 2004 to April 2005.
- Sensor Array: 5 metal oxide chemical sensors (tin oxide, titania, tungsten oxide, indium oxide) along with temperature, relative humidity, and absolute humidity.
- Missing Value Marker: $-200$ sentinel values are converted to `NaN`.

### 4.2 Leakage-Free Feature Engineering
To predict Carbon Monoxide concentration `CO(GT)` in $mg/m^3$:
1. **Diurnal and Seasonal Cyclical Encodings:**
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
   $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
2. **Historical Lags:** Target lags at $t-1$, $t-2$, $t-3$, and $t-24$ (previous day same hour).
3. **Rolling Statistics:** 6-hour rolling mean and rolling standard deviation computed strictly on historical observations (`closed='left'`) to avoid lookahead.
4. **Imputation:** Feature medians are calculated exclusively on the training split ($80\%$) and applied to the held-out test split ($20\%$).

### 4.3 Model Comparison
Three algorithms are trained and evaluated on identical chronological test observations:
1. **Linear Regression:** Standard ordinary least squares baseline with standard feature scaling.
2. **Random Forest Regressor:** Ensemble of 100 decision trees with max depth 12 and feature subsampling.
3. **XGBoost Regressor:** Gradient-boosted decision trees optimizing squared error with shrinkage rate 0.08 and subsampling 0.85.

Evaluation metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
