# ChemScheme DECIMER Container (split from chemscheme-ocsr)
# CPU-only TensorFlow + DECIMER for molecule image recognition
#
# Build:  docker build -t chemscheme-decimer docker/decimer/
# Run:    docker run --rm -v /path/to/data:/data chemscheme-decimer --image /data/mol.png
#
# Expected image size: ~2-2.5 GB (vs 8.3 GB combined)

# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Global constraint: lock numpy<2 (rdkit-pypi needs it)
RUN echo "numpy<2" > /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# Install TensorFlow CPU + shared deps first
RUN pip install --no-cache-dir \
    tensorflow-cpu \
    opencv-python-headless \
    "rdkit-pypi>=2021.03.2"

# Install DECIMER with --no-deps to prevent it pulling GPU tensorflow (G1 fix)
RUN pip install --no-cache-dir --no-deps decimer

# Pre-download model at build time
COPY warm_models.py /tmp/warm_models.py
RUN python /tmp/warm_models.py && rm /tmp/warm_models.py

# ── Aggressive cleanup ────────────────────────────────────────────
RUN set -eux; \
    SP=/usr/local/lib/python3.10/site-packages; \
    # TensorFlow development artifacts
    rm -rf $SP/tensorflow/include/; \
    rm -rf $SP/tensorflow/compiler/; \
    rm -rf $SP/tensorflow/lite/; \
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

# Copy DECIMER model cache to appuser-accessible location
COPY --from=builder /root/.data/ /tmp/model-data/

COPY entrypoint.py /app/entrypoint.py
WORKDIR /app

# Drop root privileges for runtime security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data && \
    mv /tmp/model-data/ /home/appuser/.data/ && \
    chown -R appuser:appuser /app /data /home/appuser/.data
USER appuser

# DECIMER looks for models under ~/.data
ENV OSTIO_DATA=/home/appuser/.data

# VOLUME after chown so ownership is preserved
VOLUME ["/data"]

ENTRYPOINT ["python", "/app/entrypoint.py"]
