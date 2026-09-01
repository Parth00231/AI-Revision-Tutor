#!/usr/bin/env python3
"""
Server entry point.
Run: uv run python server.py
Then open: http://localhost:8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=["."],
    )
