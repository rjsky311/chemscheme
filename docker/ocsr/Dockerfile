# ChemScheme OCSR Container (DEPRECATED — use split images instead)
#
# DEPRECATED: This combined image (~8.3 GB) bundles both PyTorch and TensorFlow.
# Prefer the dedicated split images for smaller downloads:
#   docker build -t chemscheme-molscribe docker/molscribe/  (~2-2.5 GB)
#   docker build -t chemscheme-decimer docker/decimer/       (~2-2.5 GB)
#
# This file is kept for backward compatibility with existing deployments.
#
# Build:  docker build -t chemscheme-ocsr docker/ocsr/
# Run:    docker run --rm -v /path/to/data:/data chemscheme-ocsr --tool decimer --image /data/mol.png
#
# Multi-stage build with CPU-only PyTorch (no NVIDIA CUDA stubs)
# Key: MolScribe requires torch<2.0, so we use torch 1.13.1+cpu

# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Global constraint: lock numpy<2 (rdkit-pypi + torchvision need it)
RUN echo "numpy<2" > /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# Install PyTorch 1.x CPU-only (MolScribe requires torch<2.0)
# --index-url ensures CPU-only wheels without NVIDIA stubs
RUN pip install --no-cache-dir \
    "torch<2.0" "torchvision<0.15" \
    --index-url https://download.pytorch.org/whl/cpu

# Install MolScribe WITHOUT deps to prevent torch CUDA reinstall from PyPI
# Then install its non-torch deps separately
RUN pip install --no-cache-dir --no-deps MolScribe
# MolScribe's exact deps (from setup.py): pin versions to match
RUN pip install --no-cache-dir \
    "albumentations==1.1.0" \
    "SmilesPE==0.0.3" \
    "OpenNMT-py==2.2.0" \
    "rdkit-pypi>=2021.03.2" \
    "timm==0.4.12" \
    pandas

# Install TensorFlow CPU + DECIMER + shared deps
RUN pip install --no-cache-dir \
    tensorflow-cpu \
    decimer \
    opencv-python-headless \
    huggingface_hub

# Pre-download model checkpoints at build time
COPY warm_models.py /tmp/warm_models.py
RUN python /tmp/warm_models.py && rm /tmp/warm_models.py

# ── Cleanup: strip development artifacts ──────────────────────────
RUN set -eux; \
    rm -rf /usr/local/lib/python3.10/site-packages/torch/include/; \
    rm -rf /usr/local/lib/python3.10/site-packages/torch/test/; \
    rm -rf /usr/local/lib/python3.10/site-packages/tensorflow/include/; \
    rm -rf /usr/local/lib/python3.10/site-packages/caffe2/; \
    find /usr/local/lib/python3.10/site-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true; \
    find /usr/local/lib/python3.10/site-packages -name '*.pyc' -delete 2>/dev/null || true; \
    find /usr/local/lib/python3.10/site-packages -type d -name '*.dist-info' -exec rm -rf {} + 2>/dev/null || true; \
    echo "Cleanup complete"

# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.10-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender1 libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy cleaned Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy model checkpoints to appuser-accessible location
COPY --from=builder /root/.cache/ /tmp/model-cache/
COPY --from=builder /root/.data/ /tmp/model-data/

COPY entrypoint.py /app/entrypoint.py
WORKDIR /app

# Drop root privileges for runtime security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data && \
    mv /tmp/model-cache/ /home/appuser/.cache/ && \
    mv /tmp/model-data/ /home/appuser/.data/ && \
    chown -R appuser:appuser /app /data /home/appuser/.cache /home/appuser/.data
USER appuser

ENV HF_HOME=/home/appuser/.cache/huggingface
ENV OSTIO_DATA=/home/appuser/.data

# VOLUME after chown so ownership is preserved (G7 fix)
VOLUME ["/data"]

ENTRYPOINT ["python", "/app/entrypoint.py"]
