# =============================================================================
# CockpitSentinel - Production Cloud & Dashboard Container
# Optimized for Render.com, AWS App Runner, and Container PaaS
# =============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DATABASE_BACKEND=auto

# Install minimal OS dependencies for headless OpenCV & healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install optimized cloud dependencies (CPU PyTorch + OpenCV Headless)
COPY requirements-cloud.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-cloud.txt

# Copy application source code, configs, templates, and evaluation reports
COPY src/ ./src/
COPY configs/ ./configs/
COPY test_reports/ ./test_reports/
COPY weights/trained/yolo11s_distraction/best.pt ./weights/trained/yolo11s_distraction/best.pt
COPY weights/trained/yolo11s_balanced_distraction/best.pt ./weights/trained/yolo11s_balanced_distraction/best.pt

# Install cockpit-sentinel package metadata
RUN pip install --no-cache-dir --no-deps -e .

# Expose default port
EXPOSE 8000

# Health check to ensure zero-downtime rolling updates
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/status || exit 1

# Start FastAPI web dashboard dynamically binding to $PORT (Render sets PORT=10000)
CMD ["sh", "-c", "uvicorn cockpit_sentinel.dashboard.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
