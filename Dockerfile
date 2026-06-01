# ============================================================
# Dockerfile — Clothing Classifier (multi-stage build)
# Base: python:3.11-slim
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# gcc needed for some pip compilations; libgomp1 for onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

# onnxruntime requires libgomp1 at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy only necessary artifacts — no training code, no notebooks
COPY app.py      .
COPY best.onnx   .
COPY labels.txt  .

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    ONNXRUNTIME_CPU_NUM_THREADS=2 \
    PORT=8000 \
    MODEL_PATH=/app/best.onnx \
    LABELS_PATH=/app/labels.txt \
    CONF_THRESHOLD=0.70

EXPOSE 8000

# Run as non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
