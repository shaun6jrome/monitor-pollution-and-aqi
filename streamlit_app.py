"""Streamlit Community Cloud Entrypoint Alias.

Imports and executes the main environmental intelligence application from app.py.
Ensures both 'app.py' and 'streamlit_app.py' work as deployment entrypoints on Streamlit Cloud.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Execute app.py in __main__ namespace
app_file = BASE_DIR / "app.py"
runpy.run_path(str(app_file), run_name="__main__")
