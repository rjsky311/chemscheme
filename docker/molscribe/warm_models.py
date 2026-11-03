"""Warm MolScribe model cache during Docker build.

Downloads the checkpoint so it's baked into the image
and doesn't need to be fetched at runtime.
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


if __name__ == "__main__":
    warm_molscribe()
    print("Model warm-up complete.", flush=True)
    sys.exit(0)
