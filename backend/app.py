#!/usr/bin/env python3
"""
Entry point launcher for Agent-Forge Web Server.
Runs Uvicorn on http://localhost:8001
"""
import sys
from pathlib import Path

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import uvicorn

if __name__ == "__main__":
    print("Starting Agent-Forge Backend Server at http://127.0.0.1:8001")
    uvicorn.run("agentforge.api.server:app", host="127.0.0.1", port=8001, reload=True)
