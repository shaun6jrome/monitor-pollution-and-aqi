"""Pollution relevance classification pipeline.

Classifies unstructured news articles into:
- 1: Pollution-related
- 0: Not pollution-related
- -1: Uncertain / Insufficient evidence

Uses a transparent TF-IDF + Logistic Regression baseline trained on curated
domain examples, with a documented heuristic fallback when scikit-learn is uninitialized.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Curated reference training corpus with documented ground truth labels
REFERENCE_TRAINING_DATA: List[Tuple[str, int]] = [
    # Positive examples (1: Pollution-related)
    ("Delhi air quality plunges into severe category as thick toxic smog blankets capital", 1),
    ("Industrial emissions from chemical factories in industrial estate exceed permissible limits", 1),
    ("Farmers continue stubble burning in Punjab and Haryana causing severe particulate pollution", 1),
    ("Municipal corporation cracks down on illegal waste burning in open garbage dumps", 1),
    ("High levels of PM2.5 and PM10 recorded across continuous ambient air quality monitoring stations", 1),
    ("Vehicular exhaust and diesel emissions contribute to 35 percent of winter air pollution", 1),
    ("Construction sites ordered to halt operations due to severe dust pollution violations", 1),
    ("Massive landfill fire at Ghazipur releases toxic fumes and acrid smoke across residential colonies", 1),
    ("Pollution control board issues closure notices to polluting dyeing units emitting toxic effluents", 1),
    ("Thermal power plant fined for failing to install flue gas desulfurization units to cut SO2 emissions", 1),
    ("Ozone levels exceed national ambient air standards during afternoon hours across major intersections", 1),
    ("Brick kilns operating without zig-zag technology sealed by environmental task force", 1),
    ("Residents protest against cement factory releasing white particulate dust into residential area", 1),
    ("Smog tower fails to provide relief as AQI crosses 450 in National Capital Region", 1),
    ("Toxic chemical leak from industrial plant hospitalizes over twenty workers with breathing difficulty", 1),

    # Negative examples (0: Not pollution-related)
    ("National football championship scheduled to begin next month with twenty teams participating", 0),
    ("Stock market reaches record high led by banking and information technology shares", 0),
    ("New electric passenger vehicle launched with extended battery range and autonomous driving features", 0),
    ("State government announces financial budget focusing on rural infrastructure and healthcare", 0),
    ("Airline announces direct daily flights connecting Coimbatore and Singapore", 0),
    ("Researchers publish breakthrough paper on quantum computing error correction algorithms", 0),
    ("Central bank leaves benchmark repo rate unchanged amid easing food inflation", 0),
    ("Local school holds annual cultural festival and science exhibition showcasing student robotics", 0),
    ("City corporation launches municipal water supply upgrade project for suburban areas", 0),
    ("Police investigate commercial burglary at jewelry showroom in central business district", 0),
    ("National election commission releases voter turnout figures following general elections", 0),
    ("Software company opens new development center creating two thousand tech jobs", 0),
    ("Tourism department promotes heritage walking tours in ancient cultural districts", 0),
    ("Agricultural university develops high yield pest resistant variety of wheat crop", 0),
    ("Film festival announces award winners across international competition categories", 0),
]

POLLUTION_KEYWORDS = {
    "strong": [
        "air quality", "aqi", "pm2.5", "pm10", "smog", "stubble burning",
        "crop burning", "particulate matter", "toxic fumes", "industrial emissions",
        "sulfur dioxide", "nitrogen dioxide", "effluent", "flue gas", "hazardous air"
    ],
    "moderate": [
        "pollution", "emissions", "fumes", "smoke", "exhaust", "dust", "toxic",
        "cpcb", "environmental violation", "air pollution"
    ]
}


class PollutionClassifier:
    """Classifies whether unstructured text discusses real environmental pollution."""

    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        self._train_baseline()

    def _train_baseline(self) -> None:
        """Fit TF-IDF + Logistic Regression on the documented training set."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            texts = [item[0] for item in REFERENCE_TRAINING_DATA]
            labels = [item[1] for item in REFERENCE_TRAINING_DATA]

            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=500,
                lowercase=True,
                stop_words="english",
            )
            x_vec = self.vectorizer.fit_transform(texts)

            self.classifier = LogisticRegression(C=1.0, random_state=42)
            self.classifier.fit(x_vec, labels)
            self.is_trained = True
            logger.info("PollutionClassifier TF-IDF baseline trained on %d samples.", len(texts))
        except ImportError:
            logger.warning("scikit-learn not available. Falling back to heuristic classifier.")
            self.is_trained = False

    def classify_text(self, title: str, content: Optional[str] = None) -> Dict[str, Any]:
        """Classify article text into 1 (pollution), 0 (not pollution), or -1 (uncertain).

        Returns dictionary with label, confidence, method, and explanation.
        """
        combined = f"{title or ''} {content or ''}".strip().lower()
        if not combined:
            return {
                "label": -1,
                "confidence": 0.0,
                "method": "empty_input",
                "explanation": "No text content provided for classification.",
            }

        # 1. Supervised TF-IDF baseline if initialized
        if self.is_trained and self.classifier and self.vectorizer:
            try:
                x = self.vectorizer.transform([combined])
                probs = self.classifier.predict_proba(x)[0]
                prob_positive = float(probs[1])

                if prob_positive >= 0.65:
                    return {
                        "label": 1,
                        "confidence": round(prob_positive, 4),
                        "method": "tfidf_logistic_regression",
                        "explanation": f"Supervised model predicted pollution-related (P={prob_positive:.2f}).",
                    }
                elif prob_positive <= 0.35:
                    return {
                        "label": 0,
                        "confidence": round(1.0 - prob_positive, 4),
                        "method": "tfidf_logistic_regression",
                        "explanation": f"Supervised model predicted not pollution-related (P_pos={prob_positive:.2f}).",
                    }
                else:
                    return {
                        "label": -1,
                        "confidence": round(abs(prob_positive - 0.5) * 2, 4),
                        "method": "tfidf_logistic_regression",
                        "explanation": f"Uncertain prediction: probability {prob_positive:.2f} falls in ambiguity window [0.35, 0.65].",
                    }
            except Exception as e:
                logger.warning("TF-IDF classification error: %s. Using heuristic fallback.", e)

        # 2. Transparent Heuristic Fallback
        strong_matches = [w for w in POLLUTION_KEYWORDS["strong"] if w in combined]
        moderate_matches = [w for w in POLLUTION_KEYWORDS["moderate"] if w in combined]

        if strong_matches or len(moderate_matches) >= 2:
            conf = min(0.95, 0.60 + 0.15 * len(strong_matches) + 0.05 * len(moderate_matches))
            return {
                "label": 1,
                "confidence": round(conf, 4),
                "method": "domain_lexicon_heuristic",
                "explanation": f"Detected environmental pollution terms: {', '.join(strong_matches + moderate_matches)}.",
            }
        elif len(moderate_matches) == 1:
            return {
                "label": -1,
                "confidence": 0.50,
                "method": "domain_lexicon_heuristic",
                "explanation": f"Single weak keyword match '{moderate_matches[0]}' insufficient for confident classification.",
            }
        else:
            return {
                "label": 0,
                "confidence": 0.90,
                "method": "domain_lexicon_heuristic",
                "explanation": "No environmental pollution terms detected in text.",
            }
