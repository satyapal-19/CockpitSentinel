# =============================================================================
# CockpitSentinel - Production Cloud & Dashboard Container
# =============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DASHBOARD_PORT=8000 \
    DATABASE_BACKEND=auto

# Install system dependencies for OpenCV and MediaPipe headless execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] pydantic pymongo

# Copy source code, configs, templates, and documentation
COPY src/ ./src/
COPY configs/ ./configs/
COPY test_reports/ ./test_reports/

# Install local package in editable or standard mode
RUN pip install --no-cache-dir -e .

# Expose FastAPI dashboard port
EXPOSE 8000

# Health check to ensure zero-downtime rolling updates
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Start FastAPI web dashboard
CMD ["uvicorn", "cockpit_sentinel.dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
