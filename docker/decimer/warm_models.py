"""Warm DECIMER model cache during Docker build.

Triggers the model download so it's baked into the image
and doesn't need to be fetched at runtime.
"""

from __future__ import annotations

import sys
import traceback


def warm_decimer() -> None:
    """Trigger DECIMER model download."""
    try:
        print("Warming DECIMER model...", flush=True)
        # DECIMER downloads on first import of the predictor
        from DECIMER import predict_SMILES  # noqa: F401

        print("DECIMER model cached.", flush=True)
    except Exception:
        print("WARNING: DECIMER warm-up failed (non-fatal):", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    warm_decimer()
    print("Model warm-up complete.", flush=True)
    sys.exit(0)
