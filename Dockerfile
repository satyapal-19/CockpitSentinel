# =============================================================================
# CockpitSentinel - Production Cloud & Dashboard Container
# Optimized for Render.com, AWS App Runner, and Container PaaS
# =============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DATABASE_BACKEND=auto \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    YOLO_OFFLINE=true \
    YOLO_AUTOINSTALL=false \
    YOLO_VERBOSE=false

# Install minimal OS dependencies for OpenCV, PyTorch & healthcheck (libgl1 fixes libGL.so.1 missing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install optimized cloud dependencies (CPU PyTorch + OpenCV Headless)
COPY requirements-cloud.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cloud.txt

# Pre-cache FaceNet weights and export lightweight ONNX for sub-100ms low-memory CPU inference
RUN mkdir -p /root/.cache/torch/checkpoints /root/.cache/torch/hub/checkpoints /app/weights/pretrained && \
    curl -fSL -o /root/.cache/torch/checkpoints/20180402-114759-vggface2.pt \
    https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt && \
    cp /root/.cache/torch/checkpoints/20180402-114759-vggface2.pt /root/.cache/torch/hub/checkpoints/20180402-114759-vggface2.pt && \
    python -c "import torch; from facenet_pytorch import InceptionResnetV1; m=InceptionResnetV1(pretrained='vggface2').eval(); d=torch.randn(1,3,160,160); p='/app/weights/pretrained/facenet.onnx'; exec('try:\n torch.onnx.export(m, d, p, opset_version=14, dynamo=False)\nexcept TypeError:\n torch.onnx.export(m, d, p, opset_version=14)')"

# Copy application source code, configs, templates, and evaluation reports
COPY src/ ./src/
COPY configs/ ./configs/
COPY test_reports/ ./test_reports/
COPY weights/pretrained/yolo11n.onnx ./weights/pretrained/yolo11n.onnx
COPY weights/trained/yolo11s_distraction/best.pt ./weights/trained/yolo11s_distraction/best.pt
COPY weights/trained/yolo11s_balanced_distraction/best.pt ./weights/trained/yolo11s_balanced_distraction/best.pt
COPY weights/trained/yolo11s_balanced_distraction/best.onnx ./weights/trained/yolo11s_balanced_distraction/best.onnx

# Install cockpit-sentinel package metadata
RUN pip install --no-cache-dir --no-deps -e .

# Expose default port
EXPOSE 8000

# Health check to ensure zero-downtime rolling updates
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/status || exit 1

# Start FastAPI web dashboard dynamically binding to $PORT (Render sets PORT=10000)
CMD ["sh", "-c", "uvicorn cockpit_sentinel.dashboard.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
