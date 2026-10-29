"""Official CPCB CAAQMS Station Geographic Coordinate Gazetteer.

Maps prominent Continuous Ambient Air Quality Monitoring Stations (CAAQMS)
across Indian metropolitan and industrial corridors to verified GPS coordinates.
Used for geospatial plotting and distance calculations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

STATION_COORDINATES: Dict[str, Dict[str, Any]] = {
    # Coimbatore, Tamil Nadu
    "SIDCO Kurichi, Coimbatore - TNPCB": {
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "latitude": 10.9535,
        "longitude": 76.9744,
        "type": "Industrial CAAQMS",
    },
    "PSG Tech, Peelamedu, Coimbatore - TNPCB": {
        "city": "Coimbatore",
        "state": "Tamil Nadu",
        "latitude": 11.0247,
        "longitude": 77.0028,
        "type": "Urban CAAQMS",
    },

    # Delhi NCR
    "Anand Vihar, Delhi - DPCC": {
        "city": "Delhi",
        "state": "Delhi",
        "latitude": 28.6476,
        "longitude": 77.3158,
        "type": "Commercial / Transport CAAQMS",
    },
    "R K Puram, Delhi - DPCC": {
        "city": "Delhi",
        "state": "Delhi",
        "latitude": 28.5632,
        "longitude": 77.1869,
        "type": "Residential CAAQMS",
    },
    "Punjabi Bagh, Delhi - DPCC": {
        "city": "Delhi",
        "state": "Delhi",
        "latitude": 28.6741,
        "longitude": 77.1311,
        "type": "Residential / Commercial CAAQMS",
    },
    "ITO, Delhi - CPCB": {
        "city": "Delhi",
        "state": "Delhi",
        "latitude": 28.6317,
        "longitude": 77.2410,
        "type": "Traffic Intersection CAAQMS",
    },
    "Sector 62, Noida - UPPCB": {
        "city": "Noida",
        "state": "Uttar Pradesh",
        "latitude": 28.6245,
        "longitude": 77.3639,
        "type": "Industrial / IT CAAQMS",
    },
    "Vikas Sadan, Gurugram - HSPCB": {
        "city": "Gurugram",
        "state": "Haryana",
        "latitude": 28.4595,
        "longitude": 77.0266,
        "type": "Commercial CAAQMS",
    },

    # Mumbai, Maharashtra
    "Bandra Kurla Complex, Mumbai - MPCB": {
        "city": "Mumbai",
        "state": "Maharashtra",
        "latitude": 19.0657,
        "longitude": 77.8687,
        "type": "Commercial CAAQMS",
    },
    "Colaba, Mumbai - MPCB": {
        "city": "Mumbai",
        "state": "Maharashtra",
        "latitude": 18.9067,
        "longitude": 72.8147,
        "type": "Coastal CAAQMS",
    },
    "Worli, Mumbai - MPCB": {
        "city": "Mumbai",
        "state": "Maharashtra",
        "latitude": 19.0166,
        "longitude": 72.8166,
        "type": "Urban CAAQMS",
    },

    # Chennai, Tamil Nadu
    "Alandur, Chennai - TNPCB": {
        "city": "Chennai",
        "state": "Tamil Nadu",
        "latitude": 13.0012,
        "longitude": 80.2015,
        "type": "Commercial CAAQMS",
    },
    "Manali, Chennai - TNPCB": {
        "city": "Chennai",
        "state": "Tamil Nadu",
        "latitude": 13.1667,
        "longitude": 80.2667,
        "type": "Petrochemical Industrial CAAQMS",
    },

    # Bengaluru, Karnataka
    "BTM Layout, Bengaluru - KSPCB": {
        "city": "Bengaluru",
        "state": "Karnataka",
        "latitude": 12.9166,
        "longitude": 77.6101,
        "type": "Residential CAAQMS",
    },
    "City Railway Station, Bengaluru - KSPCB": {
        "city": "Bengaluru",
        "state": "Karnataka",
        "latitude": 12.9784,
        "longitude": 77.5695,
        "type": "Transport CAAQMS",
    },

    # Kolkata, West Bengal
    "Victoria Memorial, Kolkata - WBPCB": {
        "city": "Kolkata",
        "state": "West Bengal",
        "latitude": 22.5448,
        "longitude": 88.3426,
        "type": "Urban / Park CAAQMS",
    },
    "Fort William, Kolkata - WBPCB": {
        "city": "Kolkata",
        "state": "West Bengal",
        "latitude": 22.5535,
        "longitude": 88.3377,
        "type": "Urban CAAQMS",
    },

    # Hyderabad, Telangana
    "Sanathnagar, Hyderabad - TSPCB": {
        "city": "Hyderabad",
        "state": "Telangana",
        "latitude": 17.4563,
        "longitude": 78.4439,
        "type": "Industrial CAAQMS",
    },

    # Lucknow, Uttar Pradesh
    "Talkatora District Industries Center, Lucknow - UPPCB": {
        "city": "Lucknow",
        "state": "Uttar Pradesh",
        "latitude": 26.8336,
        "longitude": 80.8924,
        "type": "Industrial CAAQMS",
    },

    # Patna, Bihar
    "IGSC Planetarium Complex, Patna - BSPCB": {
        "city": "Patna",
        "state": "Bihar",
        "latitude": 25.6093,
        "longitude": 85.1376,
        "type": "Urban CAAQMS",
    },
}

# City centroid fallbacks when exact station is unmatched
CITY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "coimbatore": (11.0168, 76.9558),
    "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090),
    "noida": (28.5355, 77.3910),
    "gurugram": (28.4595, 77.0266),
    "mumbai": (19.0760, 72.8777),
    "chennai": (13.0827, 80.2707),
    "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "kolkata": (22.5726, 88.3639),
    "hyderabad": (17.3850, 78.4867),
    "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319),
    "patna": (25.5941, 85.1376),
    "ludhiana": (30.9010, 75.8573),
    "amritsar": (31.6340, 74.8723),
    "jaipur": (26.9124, 75.7873),
    "ahmedabad": (23.0225, 72.5714),
    "pune": (18.5204, 73.8567),
    "tirupur": (11.1085, 77.3411),
}


def get_station_coordinates(station_name: str, city_name: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
    """Resolve (latitude, longitude) for a CAAQMS station name or city fallback."""
    if not station_name:
        if city_name:
            norm_city = city_name.strip().lower()
            if norm_city in CITY_CENTROIDS:
                return CITY_CENTROIDS[norm_city]
        return None, None

    # Check exact match
    if station_name in STATION_COORDINATES:
        info = STATION_COORDINATES[station_name]
        return info["latitude"], info["longitude"]

    # Check substring match
    s_lower = station_name.lower()
    for name, info in STATION_COORDINATES.items():
        if name.lower() in s_lower or s_lower in name.lower():
            return info["latitude"], info["longitude"]

    # Fallback to city centroid
    if city_name:
        norm_city = city_name.strip().lower()
        if norm_city in CITY_CENTROIDS:
            return CITY_CENTROIDS[norm_city]

    for city_key, coords in CITY_CENTROIDS.items():
        if city_key in s_lower:
            return coords

    return None, None
