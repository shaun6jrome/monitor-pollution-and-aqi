"""Automated Screenshot Capture Utility for the Running Streamlit Dashboard.

Uses Google Chrome headless mode to capture high-resolution desktop screenshots
of the running Streamlit environmental intelligence application.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCREENSHOT_DIR = BASE_DIR / "docs" / "screenshots"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def capture_screenshot(url: str, output_path: Path, wait_seconds: int = 6, width: int = 1440, height: int = 900) -> bool:
    """Capture a screenshot of a URL using headless Chrome."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = f"/tmp/chrome_ss_{os.getpid()}_{int(time.time())}"

    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--user-data-dir={temp_dir}",
        f"--window-size={width},{height}",
        "--hide-scrollbars",
        f"--virtual-time-budget={wait_seconds * 1000}",
        f"--screenshot={output_path}",
        url,
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if output_path.exists() and output_path.stat().st_size > 1000:
            logger.info("Captured screenshot: %s (%d bytes)", output_path.name, output_path.stat().st_size)
            return True
        logger.warning("Screenshot capture failed or file empty: %s", res.stderr)
        return False
    except Exception as e:
        logger.error("Error executing Chrome screenshot: %s", e)
        return False


def run_capture_suite(port: int = 8501) -> None:
    """Capture suite of dashboard screenshots across all key operational pages."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"http://localhost:{port}"

    pages = [
        ("dashboard_overview.png", f"{base_url}/?page=overview"),
        ("page_news_intelligence.png", f"{base_url}/?page=news_intelligence"),
        ("page_event_explorer.png", f"{base_url}/?page=pollution_event_explorer"),
        ("page_geographic_view.png", f"{base_url}/?page=geographic_view"),
        ("page_ml_laboratory.png", f"{base_url}/?page=ml_laboratory"),
        ("page_forecasting.png", f"{base_url}/?page=forecasting"),
        ("page_innovation.png", f"{base_url}/?page=innovation_evaluation"),
    ]

    for filename, url in pages:
        logger.info("Capturing %s from %s...", filename, url)
        output_path = SCREENSHOT_DIR / filename
        capture_screenshot(url, output_path, wait_seconds=5, width=1440, height=900)

    # Tablet view
    logger.info("Capturing tablet view from %s...", base_url)
    tablet_path = SCREENSHOT_DIR / "dashboard_tablet.png"
    capture_screenshot(base_url, tablet_path, wait_seconds=5, width=1024, height=768)

    print(f"All page screenshots saved to {SCREENSHOT_DIR}")


if __name__ == "__main__":
    target_port = 8501
    if len(sys.argv) > 1:
        target_port = int(sys.argv[1])
    run_capture_suite(port=target_port)
