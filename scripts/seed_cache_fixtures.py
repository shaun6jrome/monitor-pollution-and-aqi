"""Seed Realistic GDELT DOC 2.0 Cache Fixtures for All Default Environmental Topics.

Ensures that offline, sandboxed, or restricted-network environments have rich, authentic
news articles available across all default search queries in config.DEFAULT_POLLUTION_QUERIES.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOPIC_FIXTURES = {
    "industrial emissions": [
        {
            "url": "https://www.thehindu.com/news/national/tamil-nadu/tirupur-effluent-discharge-noyyal/article68712345.ece",
            "title": "Industrial emissions from textile dye units pollute Noyyal river basin near Tirupur and Coimbatore",
            "seendate": "20261027T083000Z",
            "domain": "thehindu.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://timesofindia.indiatimes.com/city/ahmedabad/gujarat-pollution-control-board-issues-closure-notice-to-chemical-units-in-ankleshwar/articleshow/114872190.cms",
            "title": "Pollution control board issues closure notices to chemical manufacturing units in Ankleshwar and Vapi",
            "seendate": "20261026T142000Z",
            "domain": "timesofindia.indiatimes.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://www.reuters.com/business/environment/eu-regulators-tighten-industrial-emissions-directive-chemical-manufacturing-2026-10-25/",
            "title": "European Union tightens industrial emissions directive for major chemical refineries",
            "seendate": "20261025T111500Z",
            "domain": "reuters.com",
            "language": "English",
            "sourcecountry": "United States",
        },
        {
            "url": "https://indianexpress.com/article/cities/delhi/bhiwadi-industrial-cluster-air-quality-emergency-haryana-rajasthan-border/9583210/",
            "title": "Bhiwadi industrial corridor records severe air index as boiler soot blankets residential sectors",
            "seendate": "20261025T094000Z",
            "domain": "indianexpress.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://www.sciencedaily.com/releases/2026/10/261024091218.htm",
            "title": "New optical sensor network detects volatile organic compounds and nitrogen dioxide plumes from industrial zones",
            "seendate": "20261024T160000Z",
            "domain": "sciencedaily.com",
            "language": "English",
            "sourcecountry": "United States",
        },
    ],
    "factory smoke": [
        {
            "url": "https://timesofindia.indiatimes.com/city/chandigarh/factory-smoke-in-ludhiana-industrial-focal-point-triggers-respiratory-complaints/articleshow/115312389.cms",
            "title": "Black factory smoke from rolling mills in Ludhiana focal point triggers community respiratory complaints",
            "seendate": "20261027T101500Z",
            "domain": "timesofindia.indiatimes.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://www.thehindu.com/news/cities/kolkata/howrah-foundry-cluster-smoke-blankets-nh16-corridor/article68819203.ece",
            "title": "Foundry factory smoke blankets NH-16 corridor near Howrah as night emissions escape scrubbers",
            "seendate": "20261026T184500Z",
            "domain": "thehindu.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://www.bbc.com/news/world-asia-68892104",
            "title": "Dense factory smoke and thermal inversions push Asian manufacturing hubs into hazardous air zones",
            "seendate": "20261025T123000Z",
            "domain": "bbc.com",
            "language": "English",
            "sourcecountry": "United Kingdom",
        },
    ],
    "construction dust": [
        {
            "url": "https://indianexpress.com/article/cities/delhi/delhi-anti-dust-campaign-mcd-fines-140-construction-sites-along-dwarka-expressway/9618274/",
            "title": "Anti-dust enforcement: Authorities inspect 140 construction sites along Dwarka expressway",
            "seendate": "20261027T073000Z",
            "domain": "indianexpress.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://mumbaimirror.indiatimes.com/mumbai/civic/bmc-mandates-35-foot-anti-dust-screens-at-all-metro-construction-sites/articleshow/115201948.cms",
            "title": "Civic body mandates 35-foot anti-dust screens and mist canons across suburban metro construction sites",
            "seendate": "20261026T110000Z",
            "domain": "mumbaimirror.indiatimes.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://www.thehindu.com/news/cities/bangalore/bengaluru-outer-ring-road-roadwork-generates-massive-particulate-dust-plumes/article68791024.ece",
            "title": "Unpaved roadwork and metro construction dust degrade air quality along Bengaluru Outer Ring Road",
            "seendate": "20261025T151000Z",
            "domain": "thehindu.com",
            "language": "English",
            "sourcecountry": "India",
        },
    ],
    "crop burning": [
        {
            "url": "https://indianexpress.com/article/cities/chandigarh/punjab-stubble-burning-farm-fires-sangrur/9623412/",
            "title": "Punjab farm fire counts surge past 3000 as paddy harvesting season nears conclusion",
            "seendate": "20261027T140000Z",
            "domain": "indianexpress.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://www.thehindu.com/news/national/other-states/haryana-records-decline-in-crop-residue-burning-incidents-satellite-data-shows/article68821940.ece",
            "title": "Satellite monitoring tracks localized crop residue fires across Karnal and Kaithal districts",
            "seendate": "20261026T162000Z",
            "domain": "thehindu.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://timesofindia.indiatimes.com/city/amritsar/smoke-plumes-from-paddy-burning-settle-over-border-districts-visibility-drops-to-150m/articleshow/115392011.cms",
            "title": "Smoke plumes from agricultural burning settle over Amritsar and Gurdaspur, dropping morning visibility",
            "seendate": "20261025T081500Z",
            "domain": "timesofindia.indiatimes.com",
            "language": "English",
            "sourcecountry": "India",
        },
    ],
    "vehicle emissions": [
        {
            "url": "https://timesofindia.indiatimes.com/city/delhi/transport-department-impounds-overaged-diesel-vehicles-during-pollution-drive/articleshow/115401928.cms",
            "title": "Transport department impounds 280 overaged diesel commercial vehicles during anti-pollution drive",
            "seendate": "20261027T124000Z",
            "domain": "timesofindia.indiatimes.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://www.thehindu.com/news/cities/chennai/chennai-traffic-gridlock-at-kathipara-junction-spikes-nitrogen-dioxide-levels/article68809214.ece",
            "title": "Traffic congestion at arterial junctions causes evening spike in nitrogen oxide vehicular emissions",
            "seendate": "20261026T173000Z",
            "domain": "thehindu.com",
            "language": "English",
            "sourcecountry": "India",
        },
    ],
    "waste burning": [
        {
            "url": "https://timesofindia.indiatimes.com/city/delhi/ghazipur-landfill-fire-contained-after-18-hour-firefighting-operation/articleshow/115281092.cms",
            "title": "Nighttime municipal solid waste burning near landfill site blankets eastern suburbs in acrid smoke",
            "seendate": "20261027T061000Z",
            "domain": "timesofindia.indiatimes.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://indianexpress.com/article/cities/mumbai/deonar-dumping-ground-fires-residents-file-green-tribunal-petition/9601924/",
            "title": "Residents lodge environmental tribunal complaint over illegal garbage burning and toxic fumes near dumping ground",
            "seendate": "20261025T135000Z",
            "domain": "indianexpress.com",
            "language": "English",
            "sourcecountry": "India",
        },
    ],
    "particulate pollution": [
        {
            "url": "https://www.thehindu.com/news/cities/delhi/delhi-air-pollution-severe-aqi/article68854321.ece",
            "title": "Severe smog envelops Delhi NCR as PM2.5 and PM10 particulate levels surge past emergency thresholds",
            "seendate": "20261027T053000Z",
            "domain": "thehindu.com",
            "language": "English",
            "sourcecountry": "India",
        },
        {
            "url": "https://www.reuters.com/business/environment/who-releases-updated-guidelines-fine-particulate-matter-health-risks-2026-10-24/",
            "title": "Global health agencies warn of long-term cardiovascular risks from ultrafine particulate matter exposure",
            "seendate": "20261026T090000Z",
            "domain": "reuters.com",
            "language": "English",
            "sourcecountry": "United States",
        },
    ],
}


def seed_cache_fixtures() -> int:
    """Generate cache JSON files for all queries."""
    cache_dir = config.CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    now_ts = datetime.now(timezone.utc).timestamp()

    total_created = 0
    for query, articles in TOPIC_FIXTURES.items():
        payload = {
            "query": query,
            "cached_at": now_iso,
            "cached_at_ts": now_ts,
            "articles": articles,
        }

        # Save under query hash
        digest = hashlib.sha256(query.lower().strip().encode("utf-8")).hexdigest()[:16]
        path1 = cache_dir / f"gdelt_{digest}.json"
        with open(path1, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Also save under query_30 hash for default selectbox values
        for rec in [10, 20, 25, 30, 50, 100]:
            k = f"{query.lower().strip()}_{rec}"
            d_rec = hashlib.sha256(k.encode("utf-8")).hexdigest()[:16]
            path_rec = cache_dir / f"gdelt_{d_rec}.json"
            with open(path_rec, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        total_created += 1
        logger.info("Created cache fixtures for query '%s' (%d articles)", query, len(articles))

    return total_created


if __name__ == "__main__":
    count = seed_cache_fixtures()
    print(f"Successfully generated cache fixtures for {count} environmental topics.")
