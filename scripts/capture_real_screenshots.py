"""Capture high-resolution screenshots of Streamlit pages using Chrome Remote Debugging and WebSockets.

Navigates to each Streamlit page, waits for React rendering to finish, and captures full-resolution PNG screenshots.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import websockets

BASE_DIR = Path(__file__).resolve().parent.parent
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCREENSHOT_DIR = BASE_DIR / "docs" / "screenshots"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def capture_page_cdp(ws_url: str, page_url: str, output_path: Path) -> bool:
    """Connect to Chrome via WebSocket CDP, navigate, wait for React, and capture screenshot."""
    try:
        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            # Navigate
            nav_cmd = {"id": 1, "method": "Page.navigate", "params": {"url": page_url}}
            await ws.send(json.dumps(nav_cmd))
            
            # Read messages until navigate completes
            msg = await ws.recv()
            
            # Wait 3.5 seconds for Streamlit React components and charts to mount
            await asyncio.sleep(3.5)

            # Capture screenshot
            shot_cmd = {"id": 2, "method": "Page.captureScreenshot", "params": {"format": "png"}}
            await ws.send(json.dumps(shot_cmd))
            
            resp_str = await ws.recv()
            data = json.loads(resp_str)
            while "result" not in data or "data" not in data["result"]:
                resp_str = await ws.recv()
                data = json.loads(resp_str)

            img_bytes = base64.b64decode(data["result"]["data"])
            with open(output_path, "wb") as f:
                f.write(img_bytes)

            logger.info("Successfully captured %s (%d bytes)", output_path.name, len(img_bytes))
            return True
    except Exception as e:
        logger.error("Error capturing %s: %s", output_path.name, e)
        return False


def main(port: int = 8501) -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    debug_port = 9222
    user_data = f"/tmp/chrome_cdp_{os.getpid()}"

    chrome_cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-proxy-server",
        f"--remote-debugging-port={debug_port}",
        f"--user-data-dir={user_data}",
        "--window-size=1440,900",
        "about:blank",
    ]

    logger.info("Launching Chrome with remote debugging on port %d...", debug_port)
    proc = subprocess.Popen(chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    try:
        # Get active tab websocket URL
        url_list = f"http://127.0.0.1:{debug_port}/json/list"
        req = urllib.request.urlopen(url_list, timeout=5)
        tabs = json.loads(req.read().decode())
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        logger.info("Connected to Chrome via CDP: %s", ws_url)

        pages = [
            ("dashboard_overview.png", f"http://localhost:{port}/?page=overview"),
            ("page_news_intelligence.png", f"http://localhost:{port}/?page=news_intelligence"),
            ("page_event_explorer.png", f"http://localhost:{port}/?page=pollution_event_explorer"),
            ("page_geographic_view.png", f"http://localhost:{port}/?page=geographic_view"),
            ("page_ml_laboratory.png", f"http://localhost:{port}/?page=ml_laboratory"),
            ("page_forecasting.png", f"http://localhost:{port}/?page=forecasting"),
            ("page_innovation.png", f"http://localhost:{port}/?page=innovation_evaluation"),
        ]

        for filename, url in pages:
            out_file = SCREENSHOT_DIR / filename
            asyncio.run(capture_page_cdp(ws_url, url, out_file))

        logger.info("All screenshots captured and saved to %s", SCREENSHOT_DIR)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
