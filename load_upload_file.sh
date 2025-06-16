#!/bin/bash

# This script is used to run the FastAPI application with Uvicorn.
# It uses the 'uv' command to run Uvicorn with specified parameters.

uv run -m uvicorn --host 0  --port 8080 app.upload_file:app --workers 4 --log-level info --reload