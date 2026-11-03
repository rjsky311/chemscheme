# ChemScheme MolNexTR Container (S68 — closes S67 Part 5a / MARCUS-spec parity)
# CPU-only PyTorch 2.x + MolNexTR for molecule image recognition
#
# MolNexTR (Chen et al., 2024): ConvNext + ViT + Transformer-decoder OCSR;
# 67% SMILES accuracy on real chemistry papers — highest single-tool open-source
# in the MARCUS ensemble (RSC Digital Discovery, Oct 2025).
#
# Build:  docker build -t chemscheme-molnextr docker/molnextr/
# Run:    docker run --rm -v /path/to/data:/data chemscheme-molnextr --image /data/mol.png
#
# Expected image size: ~2.5-3 GB (PyTorch CPU + MolNexTR weights)

# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Lock numpy<2 (rdkit-pypi compatibility); transformers torch<3 baseline
RUN echo "numpy<2" > /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# PyTorch CPU-only — MolNexTR upstream uses torch>=2.0
RUN pip install --no-cache-dir \
    "torch>=2.0,<3.0" "torchvision>=0.15,<1.0" \
    --index-url https://download.pytorch.org/whl/cpu

# MolNexTR core dependencies (ConvNext via timm; transformers; image utilities)
RUN pip install --no-cache-dir \
    "timm>=0.9" \
    "transformers>=4.30,<5.0" \
    "rdkit-pypi>=2022.09" \
    "Pillow>=10.0" \
    "opencv-python-headless>=4.8" \
    "albumentations>=1.3" \
    "einops>=0.6" \
    "numpy<2"

# MolNexTR is research code (no PyPI release). Install from upstream git.
# Pin a specific commit when production-deploying — `main` is for first-build smoke only.
# Ref: Chen et al., Org Process Res Dev / arxiv 2410.10103
RUN pip install --no-cache-dir --no-deps \
    "git+https://github.com/CYF2000127/MolNexTR.git@main" \
    || echo "WARNING: MolNexTR git install failed — entrypoint will surface as runtime error"

# Cleanup to shrink image
RUN SP=$(python -c "import sys; print(sys.path[-1])"); \
    find $SP -name '*.pyc' -delete 2>/dev/null || true; \
    find $SP -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true; \
    find $SP -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true; \
    find $SP -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    echo "Cleanup complete"

# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.10-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender1 libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

COPY entrypoint.py /app/entrypoint.py
WORKDIR /app

# Drop root for runtime security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data && \
    chown -R appuser:appuser /app /data /home/appuser
USER appuser

# MolNexTR weights cached under HOME (~/.cache/molnextr or HF cache)
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface

VOLUME ["/data"]

ENTRYPOINT ["python", "/app/entrypoint.py"]
