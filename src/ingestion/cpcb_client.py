"""Official Indian Air Quality Data Connector (CPCB / data.gov.in).

Retrieves ambient air quality station records from the official data.gov.in
real-time air quality API or provides verifiable fallback status instructions
linking directly to the CPCB National AQI portal (https://airquality.cpcb.gov.in/AQI_India/).

Never fabricates live station measurements if API access or credentials are absent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

import config
from src.storage.repository import PollutionRepository

logger = logging.getLogger(__name__)


class CPCBClient:
    """Client for querying official Indian ambient air quality feeds."""

    def __init__(
        self,
        repository: Optional[PollutionRepository] = None,
        api_key: Optional[str] = None,
        timeout: int = 15,
    ):
        self.repository = repository or PollutionRepository()
        self.api_key = api_key if api_key is not None else config.DATA_GOV_IN_API_KEY
        self.timeout = timeout
        self.portal_url = config.CPCB_PORTAL_URL
        self.api_base_url = config.CPCB_API_BASE_URL

    def check_connection_status(self) -> Dict[str, Any]:
        """Check whether the official data.gov.in API is configured and accessible."""
        if not self.api_key:
            return {
                "configured": False,
                "status": "NOT_CONFIGURED",
                "message": (
                    "data.gov.in API key is not configured. Set DATA_GOV_IN_API_KEY in .env "
                    "or environment variables. Visit https://data.gov.in to request a free developer API key."
                ),
                "portal_url": self.portal_url,
            }

        try:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": 1,
            }
            resp = requests.get(self.api_base_url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                return {
                    "configured": True,
                    "status": "ONLINE",
                    "message": "Official data.gov.in CPCB API endpoint is active and accessible.",
                    "portal_url": self.portal_url,
                }
            elif resp.status_code == 403 or resp.status_code == 401:
                return {
                    "configured": True,
                    "status": "AUTH_ERROR",
                    "message": f"Authentication failed (HTTP {resp.status_code}). Please verify your DATA_GOV_IN_API_KEY.",
                    "portal_url": self.portal_url,
                }
            else:
                return {
                    "configured": True,
                    "status": f"HTTP_{resp.status_code}",
                    "message": f"Endpoint responded with status code {resp.status_code}.",
                    "portal_url": self.portal_url,
                }
        except requests.RequestException as e:
            return {
                "configured": True,
                "status": "UNREACHABLE",
                "message": f"Network error contacting data.gov.in API: {e}",
                "portal_url": self.portal_url,
            }

    def fetch_latest_observations(
        self,
        city_filter: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Fetch latest verified observations from data.gov.in real-time AQI feed.

        Does not invent synthetic data if credentials or network are unavailable.
        """
        if not self.api_key:
            return {
                "status": "KEY_MISSING",
                "records": [],
                "count": 0,
                "message": (
                    "Official Indian air quality API key not set. "
                    "Live government station observations require DATA_GOV_IN_API_KEY in .env. "
                    "Official real-time portal: https://airquality.cpcb.gov.in/AQI_India/"
                ),
                "portal_url": self.portal_url,
            }

        params: Dict[str, Any] = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
        }
        if city_filter:
            params["filters[city]"] = city_filter

        try:
            resp = requests.get(self.api_base_url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                err_msg = f"API returned HTTP {resp.status_code}"
                self.repository.log_retrieval("CPCB", city_filter, "FAILED", error_message=err_msg)
                return {
                    "status": "API_ERROR",
                    "records": [],
                    "count": 0,
                    "error": err_msg,
                    "portal_url": self.portal_url,
                }

            data = resp.json()
            raw_records = data.get("records", [])
            saved_count = 0

            formatted_records = []
            for item in raw_records:
                station = item.get("station", item.get("station_name", "Unknown Station"))
                city = item.get("city", "Unknown City")
                state = item.get("state", "Unknown State")
                pollutant = item.get("pollutant_id", item.get("pollutant", "PM2.5"))

                try:
                    val_str = item.get("pollutant_avg", item.get("value", "0"))
                    val = float(val_str) if val_str not in ("NA", "None", "") else 0.0
                except (ValueError, TypeError):
                    val = 0.0

                unit = item.get("pollutant_unit", item.get("unit", "ug/m3"))
                aqi_val = item.get("aqi", None)
                try:
                    aqi_num = float(aqi_val) if aqi_val not in (None, "NA", "") else None
                except (ValueError, TypeError):
                    aqi_num = None

                obs_time = item.get("last_update", item.get("timestamp", datetime.now(timezone.utc).isoformat()))

                self.repository.insert_cpcb_observation(
                    station_name=station,
                    pollutant=pollutant,
                    value=val,
                    unit=unit,
                    city=city,
                    state=state,
                    aqi_index=aqi_num,
                    observation_timestamp=obs_time,
                    source_attribution="Central Pollution Control Board (CPCB) via data.gov.in",
                )
                saved_count += 1
                formatted_records.append(
                    {
                        "station": station,
                        "city": city,
                        "state": state,
                        "pollutant": pollutant,
                        "value": val,
                        "unit": unit,
                        "aqi": aqi_num,
                        "timestamp": obs_time,
                    }
                )

            self.repository.log_retrieval(
                "CPCB",
                city_filter,
                "SUCCESS",
                records_retrieved=len(raw_records),
                new_records=saved_count,
            )

            return {
                "status": "SUCCESS",
                "records": formatted_records,
                "count": len(formatted_records),
                "portal_url": self.portal_url,
            }

        except Exception as e:
            logger.error("Error fetching CPCB observations: %s", e)
            self.repository.log_retrieval("CPCB", city_filter, "NETWORK_ERROR", error_message=str(e))
            return {
                "status": "NETWORK_ERROR",
                "records": [],
                "count": 0,
                "error": str(e),
                "portal_url": self.portal_url,
            }

    def get_source_metadata(self) -> Dict[str, Any]:
        """Return provenance, update frequency, and access metadata for CPCB."""
        return {
            "source_name": "Central Pollution Control Board (CPCB) / data.gov.in",
            "jurisdiction": "Government of India - Ministry of Environment, Forest and Climate Change",
            "portal_url": self.portal_url,
            "api_endpoint": self.api_base_url,
            "data_license": "Government Open Data License - India (GODL)",
            "update_cadence": "Hourly continuous ambient monitoring updates",
            "monitored_parameters": ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"],
            "configured": bool(self.api_key),
            "instructions": (
                "To connect live Indian monitoring feeds, register at https://data.gov.in, "
                "request access to 'Real-Time Air Quality Index', and set DATA_GOV_IN_API_KEY in .env."
            ),
        }

    def diagnose_feed(self) -> Dict[str, Any]:
        """Run diagnostic health check and measure latency."""
        start_ts = datetime.now(timezone.utc).timestamp()
        status_info = self.check_connection_status()
        end_ts = datetime.now(timezone.utc).timestamp()
        latency_ms = round((end_ts - start_ts) * 1000, 2)

        return {
            "status": status_info["status"],
            "configured": status_info["configured"],
            "latency_ms": latency_ms,
            "portal_url": self.portal_url,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "message": status_info["message"],
        }
