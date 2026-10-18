# =============================================================================
# CockpitSentinel - Production Cloud & Dashboard Container
# Optimized for Hugging Face Spaces, Render.com, AWS App Runner & Docker PaaS
# =============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
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

# Set up non-root user (UID 1000 required by Hugging Face Spaces security sandbox)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TORCH_HOME=/home/user/.cache/torch

WORKDIR /home/user/app

# Install optimized cloud dependencies (CPU PyTorch + OpenCV Headless)
COPY --chown=user:user requirements-cloud.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cloud.txt

# Pre-cache FaceNet weights and export lightweight ONNX for sub-100ms inference
RUN mkdir -p /home/user/.cache/torch/checkpoints /home/user/.cache/torch/hub/checkpoints /home/user/app/weights/pretrained && \
    curl -fSL -o /home/user/.cache/torch/checkpoints/20180402-114759-vggface2.pt \
    https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt && \
    cp /home/user/.cache/torch/checkpoints/20180402-114759-vggface2.pt /home/user/.cache/torch/hub/checkpoints/20180402-114759-vggface2.pt && \
    python -c "import torch; from facenet_pytorch import InceptionResnetV1; m=InceptionResnetV1(pretrained='vggface2').eval(); d=torch.randn(1,3,160,160); p='/home/user/app/weights/pretrained/facenet.onnx'; exec('try:\n torch.onnx.export(m, d, p, opset_version=14, dynamo=False)\nexcept TypeError:\n torch.onnx.export(m, d, p, opset_version=14)')"

# Copy application source code, configs, templates, and evaluation reports
COPY --chown=user:user src/ ./src/
COPY --chown=user:user configs/ ./configs/
COPY --chown=user:user test_reports/ ./test_reports/
COPY --chown=user:user weights/pretrained/yolo11n.onnx ./weights/pretrained/yolo11n.onnx
COPY --chown=user:user weights/trained/yolo11s_distraction/best.pt ./weights/trained/yolo11s_distraction/best.pt
COPY --chown=user:user weights/trained/yolo11s_balanced_distraction/best.pt ./weights/trained/yolo11s_balanced_distraction/best.pt
COPY --chown=user:user weights/trained/yolo11s_balanced_distraction/best.onnx ./weights/trained/yolo11s_balanced_distraction/best.onnx

# Install cockpit-sentinel package metadata
RUN pip install --no-cache-dir --no-deps -e .

# Ensure full ownership of the user directory
RUN chown -R user:user /home/user

# Switch to non-root user
USER user

# Expose standard cloud port (7860 for Hugging Face Spaces, Render $PORT fallback)
EXPOSE 7860

# Health check to ensure zero-downtime rolling updates
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/api/status || exit 1

# Start FastAPI web dashboard dynamically binding to $PORT (Hugging Face passes 7860, Render passes 10000)
CMD ["sh", "-c", "python -m uvicorn cockpit_sentinel.dashboard.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
