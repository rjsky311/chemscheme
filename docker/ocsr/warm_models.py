"""Warm model caches during Docker build.

Downloads MolScribe + DECIMER model checkpoints so they're baked
into the image and don't need to be fetched at runtime.
"""

from __future__ import annotations

import sys
import traceback


def warm_molscribe() -> None:
    """Download MolScribe checkpoint from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download

        print("Downloading MolScribe checkpoint...", flush=True)
        ckpt = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
        print(f"MolScribe checkpoint cached at {ckpt}", flush=True)
    except Exception:
        print("WARNING: MolScribe warm-up failed (non-fatal):", flush=True)
        traceback.print_exc()


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
    warm_molscribe()
    warm_decimer()
    print("Model warm-up complete.", flush=True)
    sys.exit(0)
