"""Tests for CPCB CAAQMS station coordinate lookup table."""

from src.ingestion.cpcb_stations import (
    STATION_COORDINATES,
    CITY_CENTROIDS,
    get_station_coordinates,
)


def test_exact_station_coordinate_lookup():
    lat, lon = get_station_coordinates("SIDCO Kurichi, Coimbatore - TNPCB")
    assert lat is not None and lon is not None
    assert round(lat, 2) == 10.95
    assert round(lon, 2) == 76.97


def test_station_substring_coordinate_lookup():
    lat, lon = get_station_coordinates("Anand Vihar, Delhi")
    assert lat is not None and lon is not None
    assert round(lat, 2) == 28.65


def test_city_centroid_fallback():
    lat, lon = get_station_coordinates("Nonexistent Station 123", city_name="Coimbatore")
    assert lat is not None and lon is not None
    assert round(lat, 2) == 11.02
    assert round(lon, 2) == 76.96


def test_unmatched_returns_none():
    lat, lon = get_station_coordinates("Completely Unheard Location XYZ", city_name="Unknown")
    assert lat is None and lon is None
