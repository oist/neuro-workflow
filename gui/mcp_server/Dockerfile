# Use official Python 3.13 slim image
FROM python:3.13-slim

# Set a working directory
WORKDIR /app

# Install build dependencies and cleanup to keep image small
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml so we can install deps
# COPY pyproject.toml /app/

# Install pip and dependencies listed in pyproject.toml using pip
# This extracts the dependencies manually because pyproject may be simple
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        "fastmcp>=2.12.4" \
        "httpx>=0.28.1" \
        "mcp>=1.16.0"

# Copy application files
COPY workflow_mcp.py /app/

# Default command
CMD ["python", "./workflow_mcp.py"]

