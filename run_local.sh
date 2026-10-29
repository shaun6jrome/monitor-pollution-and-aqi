#!/usr/bin/env bash
# One-click startup script for the Evidence-Aware Pollution Intelligence Platform

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "================================================================="
echo "   Launching Evidence-Aware Pollution Intelligence Platform      "
echo "================================================================="

# Check if .venv exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment .venv not found. Creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Kill any lingering streamlit process on port 8501
if lsof -ti:8501 >/dev/null 2>&1; then
    echo "Port 8501 is occupied. Freeing port 8501..."
    kill -9 $(lsof -ti:8501) 2>/dev/null || true
    sleep 1
fi

echo "Opening dashboard at http://localhost:8501 ..."
streamlit run app.py --server.port 8501
