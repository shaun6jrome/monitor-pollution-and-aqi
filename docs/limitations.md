# Critical Analysis of Limitations and Data Governance Constraints

In accordance with strict scientific integrity and academic guidelines, this document explicitly details the scope boundaries, known gaps, and methodological limitations of the platform.

---

## 1. Spatial and Temporal Alignment Constraints (Section 10 Evaluation)

A foundational question explored in environmental data science is whether unstructured textual news signals can augment physical sensor forecasts.

### 1.1 Non-Alignment Between Data Sources
- **Historical Sensor Dataset (UCI):** Collected between March 2004 and April 2005 at an urban roadside station in an Italian city.
- **Unstructured News Feeds (GDELT):** Provides contemporary news articles (2024–present) covering predominantly Indian, South Asian, and global metropolitan incidents.
- **Scientific Conclusion:** It is scientifically invalid to join contemporary Indian news event signals with 2004 Italian sensor measurements and claim that news improves physical forecasting accuracy. Fabricating synthetic alignment or proxy features would violate the non-negotiable requirement of zero fabrication.
- **Formal Status:** Per prompt requirements (Section 10), the text-augmented forecasting experiment is formally documented as **NOT EVALUATED FOR DIRECT CROSS-SOURCE INFERENCE**, with the scientific rationale thoroughly explained.

---

## 2. Unstructured News Ingestion and Media Reporting Biases

1. **Metropolitan and Sensationalism Bias:** Media outlets disproportionately cover air quality crises in major capital cities (e.g., the National Capital Region of Delhi during post-Diwali winter smog) while severe rural biomass burning or localized industrial effluent discharge in smaller towns receives minimal reporting.
2. **Delayed Publication:** Media coverage of an environmental violation or chemical leak often lags several hours or days behind physical sensor detection. Unstructured news acts as an *evidence and causal discovery mechanism*, not an instantaneous low-latency physical telemetry sensor.
3. **Paywall and Anti-Scraping Restrictions:** In compliance with ethical scraping principles, the system does not bypass publisher paywalls or robots restrictions. Where full text cannot be legally retrieved via HTTP, the system safely falls back to article headlines and metadata.

---

## 3. Entity Extraction and Geocoding Boundaries

1. **Absence of Ground GPS Coordinates in Text:** News articles provide descriptive location names (e.g., "Noida Sector 62", "Kurla Industrial Area", "Sangrur District") rather than latitude and longitude pairs.
2. **Zero Coordinate Fabrication Policy:** The platform strictly refuses to assign artificial or pseudo-random coordinates to events. Where precise geocoding services are unconfigured, geographic distributions are presented as transparent frequency tables rather than unverified map pins.

---

## 4. Physical Sensor Characteristics and Drift

1. **Metal Oxide Chemical Sensor Drift:** The UCI dataset utilizes solid-state metal oxide chemical sensors ($SnO_2$, $WO_3$, $TiO_2$, $In_2O_3$). These sensors exhibit sensitivity cross-interference (e.g., humidity and temperature fluctuations alter sensor resistance alongside target gas concentration) and undergo chemical aging over time.
2. **Missing Sentinel Values:** Approximately 18% of target Carbon Monoxide observations in the UCI dataset are marked with the $-200$ missing value code. While median imputation fitted exclusively on training rows prevents forward lookahead, real-world deployment requires continuous sensor fault detection.
