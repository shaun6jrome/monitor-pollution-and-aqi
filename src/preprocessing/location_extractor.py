"""Entity extraction pipeline for pollution news.

Extracts locations, pollutant names, emission source categories, and temporal
expressions from text with exact character span attribution and normalized values.
Combines gazetteer pattern matching with optional spaCy NER fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Known geographic gazetteer focused on air quality monitoring hubs
KNOWN_LOCATIONS = {
    "delhi": "Delhi, India",
    "new delhi": "Delhi, India",
    "ncr": "National Capital Region, India",
    "noida": "Noida, Uttar Pradesh, India",
    "greater noida": "Greater Noida, Uttar Pradesh, India",
    "gurugram": "Gurugram, Haryana, India",
    "gurgaon": "Gurugram, Haryana, India",
    "ghaziabad": "Ghaziabad, Uttar Pradesh, India",
    "faridabad": "Faridabad, Haryana, India",
    "punjab": "Punjab, India",
    "haryana": "Haryana, India",
    "ludhiana": "Ludhiana, Punjab, India",
    "amritsar": "Amritsar, Punjab, India",
    "jalandhar": "Jalandhar, Punjab, India",
    "patiala": "Patiala, Punjab, India",
    "kanpur": "Kanpur, Uttar Pradesh, India",
    "lucknow": "Lucknow, Uttar Pradesh, India",
    "varanasi": "Varanasi, Uttar Pradesh, India",
    "agra": "Agra, Uttar Pradesh, India",
    "patna": "Patna, Bihar, India",
    "kolkata": "Kolkata, West Bengal, India",
    "mumbai": "Mumbai, Maharashtra, India",
    "pune": "Pune, Maharashtra, India",
    "nagpur": "Nagpur, Maharashtra, India",
    "ahmedabad": "Ahmedabad, Gujarat, India",
    "surat": "Surat, Gujarat, India",
    "jaipur": "Jaipur, Rajasthan, India",
    "bengaluru": "Bengaluru, Karnataka, India",
    "bangalore": "Bengaluru, Karnataka, India",
    "chennai": "Chennai, Tamil Nadu, India",
    "coimbatore": "Coimbatore, Tamil Nadu, India",
    "tirupur": "Tirupur, Tamil Nadu, India",
    "salem": "Salem, Tamil Nadu, India",
    "erode": "Erode, Tamil Nadu, India",
    "thoothukudi": "Thoothukudi, Tamil Nadu, India",
    "tuticorin": "Thoothukudi, Tamil Nadu, India",
    "hyderabad": "Hyderabad, Telangana, India",
    "visakhapatnam": "Visakhapatnam, Andhra Pradesh, India",
    "vizag": "Visakhapatnam, Andhra Pradesh, India",
    "chandigarh": "Chandigarh, India",
    "bhiwadi": "Bhiwadi, Rajasthan, India",
    "manesar": "Manesar, Haryana, India",
    "ankleshwar": "Ankleshwar, Gujarat, India",
    "vapi": "Vapi, Gujarat, India",
    "korba": "Korba, Chhattisgarh, India",
    "singrauli": "Singrauli, Madhya Pradesh, India",
    "talcher": "Talcher, Odisha, India",
    "asansol": "Asansol, West Bengal, India",
    "raniganj": "Raniganj, West Bengal, India",
    "dhanbad": "Dhanbad, Jharkhand, India",
    "moradabad": "Moradabad, Uttar Pradesh, India",
    "beijing": "Beijing, China",
    "lahore": "Lahore, Pakistan",
    "milan": "Milan, Italy",
}

# Chemical and particulate pollutants
POLLUTANT_PATTERNS = [
    (r"\bPM\s*2\.?5\b", "PM2.5"),
    (r"\bPM\s*10\b", "PM10"),
    (r"\bparticulate\s+matter\b", "Particulate Matter"),
    (r"\bNOx\b", "NOx"),
    (r"\bNO2\b|\bnitrogen\s+dioxide\b", "NO2"),
    (r"\bSO2\b|\bsulfur\s+dioxide\b|\bsulphur\s+dioxide\b", "SO2"),
    (r"\bCO\b|\bcarbon\s+monoxide\b", "CO"),
    (r"\bCO2\b|\bcarbon\s+dioxide\b", "CO2"),
    (r"\bO3\b|\bozone\b", "Ozone"),
    (r"\bsmog\b", "Smog"),
    (r"\bblack\s+carbon\b|\bsoot\b", "Black Carbon"),
    (r"\bvolatile\s+organic\s+compounds?\b|\bVOCs?\b", "VOC"),
    (r"\btoxic\s+(?:fumes|smoke|gas|emissions?)\b", "Toxic Fumes"),
    (r"\bfly\s+ash\b", "Fly Ash"),
]

# Source category heuristics
CATEGORY_PATTERNS = [
    (r"\b(?:crop|stubble|paddy|farm|parali)\s+(?:burn(?:ing)?|fires?)\b", "crop_burning"),
    (r"\b(?:waste|garbage|trash|landfill|plastic)\s+(?:burn(?:ing)?|fires?)\b", "waste_burning"),
    (r"\b(?:factory|plant|refinery|smelter|furnace|kiln|boiler|industrial)\s+(?:emissions?|smoke|fumes|chimney|pollution)\b", "industrial_emissions"),
    (r"\b(?:vehicle|vehicular|traffic|exhaust|diesel|tailpipe|automobile)\s+(?:emissions?|smoke|pollution)\b", "vehicular_pollution"),
    (r"\b(?:construction|demolition|unpaved|road|quarry)\s+(?:dust|debris)\b", "construction_dust"),
    (r"\b(?:wildfire|forest\s+fire|bushfire|brush\s+fire)\b", "wildfire_smoke"),
    (r"\b(?:air\s+quality|AQI|smog|hazardous\s+air|toxic\s+air|air\s+pollution)\b", "general_complaint"),
]

# Temporal expression patterns
DATE_PATTERNS = [
    (r"\b(?:yesterday|today|last\s+night|this\s+morning|past\s+week)\b", "RELATIVE_DATE"),
    (r"\b(?:on\s+)?(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", "DAY_OF_WEEK"),
    (r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b", "CALENDAR_DATE"),
    (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "NUMERIC_DATE"),
]


class EntityExtractor:
    """Extracts domain entities from environmental news text."""

    def __init__(self, use_spacy: bool = True):
        self.spacy_nlp = None
        if use_spacy:
            try:
                import spacy
                try:
                    self.spacy_nlp = spacy.load("en_core_web_sm")
                except Exception:
                    # Model not downloaded yet; fallback smoothly
                    self.spacy_nlp = None
            except ImportError:
                self.spacy_nlp = None

    def extract_locations(self, text: str) -> List[Dict[str, Any]]:
        """Extract locations using gazetteer and optional spaCy NER."""
        if not text:
            return []

        results: List[Dict[str, Any]] = []
        seen_spans = set()

        # 1. Gazetteer dictionary matching (exact, case-insensitive)
        for loc_key, norm_val in KNOWN_LOCATIONS.items():
            pattern = rf"\b{re.escape(loc_key)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    results.append(
                        {
                            "entity_type": "LOCATION",
                            "raw_text": match.group(0),
                            "normalized_value": norm_val,
                            "char_start": match.start(),
                            "char_end": match.end(),
                            "confidence": 0.95,
                            "extraction_method": "gazetteer_dictionary",
                        }
                    )

        # 2. spaCy NER if available
        if self.spacy_nlp:
            try:
                doc = self.spacy_nlp(text[:4000])  # limit length for speed
                for ent in doc.ents:
                    if ent.label_ in ("GPE", "LOC"):
                        span = (ent.start_char, ent.end_char)
                        if span not in seen_spans:
                            seen_spans.add(span)
                            raw_val = ent.text.strip()
                            norm = KNOWN_LOCATIONS.get(raw_val.lower(), raw_val)
                            results.append(
                                {
                                    "entity_type": "LOCATION",
                                    "raw_text": raw_val,
                                    "normalized_value": norm,
                                    "char_start": ent.start_char,
                                    "char_end": ent.end_char,
                                    "confidence": 0.85,
                                    "extraction_method": "spacy_ner",
                                }
                            )
            except Exception as e:
                logger.debug("spaCy location extraction skipped: %s", e)

        return results

    def extract_pollutants(self, text: str) -> List[Dict[str, Any]]:
        """Extract pollutant mentions and chemical species."""
        if not text:
            return []

        results: List[Dict[str, Any]] = []
        seen_spans = set()

        for pattern, norm_val in POLLUTANT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    results.append(
                        {
                            "entity_type": "POLLUTANT",
                            "raw_text": match.group(0),
                            "normalized_value": norm_val,
                            "char_start": match.start(),
                            "char_end": match.end(),
                            "confidence": 0.90,
                            "extraction_method": "regex_pattern",
                        }
                    )
        return results

    def extract_categories(self, text: str) -> List[Dict[str, Any]]:
        """Extract pollution event categories supported by evidence in the text."""
        if not text:
            return []

        results: List[Dict[str, Any]] = []
        seen_categories = set()

        for pattern, cat_name in CATEGORY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if cat_name not in seen_categories:
                    seen_categories.add(cat_name)
                    results.append(
                        {
                            "entity_type": "SOURCE_CATEGORY",
                            "raw_text": match.group(0),
                            "normalized_value": cat_name,
                            "char_start": match.start(),
                            "char_end": match.end(),
                            "confidence": 0.85,
                            "extraction_method": "source_taxonomy_rules",
                        }
                    )
        return results

    def extract_temporal_expressions(self, text: str) -> List[Dict[str, Any]]:
        """Extract relative and calendar temporal mentions."""
        if not text:
            return []

        results: List[Dict[str, Any]] = []
        seen_spans = set()

        for pattern, date_kind in DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    results.append(
                        {
                            "entity_type": "DATE_TIME",
                            "raw_text": match.group(0),
                            "normalized_value": match.group(0).strip(),
                            "char_start": match.start(),
                            "char_end": match.end(),
                            "confidence": 0.80,
                            "extraction_method": f"temporal_{date_kind.lower()}",
                        }
                    )
        return results

    def extract_all(self, text: str) -> List[Dict[str, Any]]:
        """Run complete extraction pipeline and aggregate entities."""
        entities = []
        entities.extend(self.extract_locations(text))
        entities.extend(self.extract_pollutants(text))
        entities.extend(self.extract_categories(text))
        entities.extend(self.extract_temporal_expressions(text))
        return entities
