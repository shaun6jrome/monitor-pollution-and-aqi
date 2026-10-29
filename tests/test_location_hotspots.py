"""Tests for industrial corridor and environmental hotspot entity extraction."""

import pytest
from src.preprocessing.location_extractor import EntityExtractor


@pytest.fixture
def extractor():
    return EntityExtractor(use_spacy=False)


def test_extract_industrial_hubs_bhiwadi_and_manesar(extractor):
    text = "Industrial emissions and boiler smoke elevated PM2.5 in Bhiwadi and Manesar auto corridor."
    locs = extractor.extract_locations(text)
    norm_vals = [l["normalized_value"] for l in locs]
    assert "Bhiwadi, Rajasthan, India" in norm_vals
    assert "Manesar, Haryana, India" in norm_vals


def test_extract_chemical_belt_ankleshwar_and_vapi(extractor):
    text = "Chemical factories in Ankleshwar and Vapi faced strict CPCB notices over VOC emissions."
    locs = extractor.extract_locations(text)
    norm_vals = [l["normalized_value"] for l in locs]
    assert "Ankleshwar, Gujarat, India" in norm_vals
    assert "Vapi, Gujarat, India" in norm_vals


def test_extract_tamil_nadu_industrial_belt(extractor):
    text = "Dyeing units in Tirupur, tanneries in Erode, and thermal plants in Tuticorin under scrutiny."
    locs = extractor.extract_locations(text)
    norm_vals = [l["normalized_value"] for l in locs]
    assert "Tirupur, Tamil Nadu, India" in norm_vals
    assert "Erode, Tamil Nadu, India" in norm_vals
    assert "Thoothukudi, Tamil Nadu, India" in norm_vals


def test_extract_coal_and_mining_corridors(extractor):
    text = "Fly ash from coal power complexes darkened skies over Korba, Singrauli, and Dhanbad."
    locs = extractor.extract_locations(text)
    norm_vals = [l["normalized_value"] for l in locs]
    assert "Korba, Chhattisgarh, India" in norm_vals
    assert "Singrauli, Madhya Pradesh, India" in norm_vals
    assert "Dhanbad, Jharkhand, India" in norm_vals
