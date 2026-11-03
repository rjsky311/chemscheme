# ChemScheme MolScribe Container (split from chemscheme-ocsr)
# CPU-only PyTorch 1.13 + MolScribe for molecule image recognition
#
# Build:  docker build -t chemscheme-molscribe docker/molscribe/
# Run:    docker run --rm -v /path/to/data:/data chemscheme-molscribe --image /data/mol.png
#
# Expected image size: ~2-2.5 GB (vs 8.3 GB combined)

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
RUN pip install --no-cache-dir \
    "torch<2.0" "torchvision<0.15" \
    --index-url https://download.pytorch.org/whl/cpu

# Install MolScribe WITHOUT deps to prevent torch CUDA reinstall from PyPI
RUN pip install --no-cache-dir --no-deps MolScribe
# MolScribe's exact deps (from setup.py): pin versions to match
RUN pip install --no-cache-dir \
    "albumentations==1.1.0" \
    "SmilesPE==0.0.3" \
    "OpenNMT-py==2.2.0" \
    "rdkit-pypi>=2021.03.2" \
    "timm==0.4.12" \
    pandas \
    huggingface_hub

# Pre-download model checkpoint at build time
COPY warm_models.py /tmp/warm_models.py
RUN python /tmp/warm_models.py && rm /tmp/warm_models.py

# ── Aggressive cleanup ────────────────────────────────────────────
RUN set -eux; \
    SP=/usr/local/lib/python3.10/site-packages; \
    # PyTorch development artifacts
    rm -rf $SP/torch/include/ $SP/torch/test/ $SP/torch/share/; \
    rm -rf $SP/caffe2/; \
    # Static libraries (not needed for inference)
    find $SP/torch/lib/ -name '*.a' -delete 2>/dev/null || true; \
    # Type stubs
    find $SP/torch/_C/ -name '*.pyi' -delete 2>/dev/null || true; \
    # Python cache + dist-info
    find $SP -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true; \
    find $SP -name '*.pyc' -delete 2>/dev/null || true; \
    find $SP -type d -name '*.dist-info' -exec rm -rf {} + 2>/dev/null || true; \
    # Test directories inside packages
    find $SP -type d -name tests -exec rm -rf {} + 2>/dev/null || true; \
    find $SP -type d -name test -exec rm -rf {} + 2>/dev/null || true; \
    # Strip shared libraries
    find $SP -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    echo "Cleanup complete"

# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.10-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender1 libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy cleaned Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy MolScribe model checkpoint to appuser-accessible location
COPY --from=builder /root/.cache/ /tmp/model-cache/

COPY entrypoint.py /app/entrypoint.py
WORKDIR /app

# Drop root privileges for runtime security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data && \
    mv /tmp/model-cache/ /home/appuser/.cache/ && \
    chown -R appuser:appuser /app /data /home/appuser/.cache
USER appuser

# HuggingFace will find the cache under $HOME/.cache/huggingface
ENV HF_HOME=/home/appuser/.cache/huggingface

# VOLUME after chown so ownership is preserved
VOLUME ["/data"]

ENTRYPOINT ["python", "/app/entrypoint.py"]
