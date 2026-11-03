#!/usr/bin/env python3
"""Entrypoint for MolScribe OCSR container.

Usage:
    python entrypoint.py --image /data/input.png

Prints SMILES to stdout on success. Errors go to stderr with exit code 1.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress ML framework logs that would pollute stdout SMILES output
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")


def run_molscribe(image_path: str) -> str:
    """Run MolScribe on a single image file."""
    import torch
    from huggingface_hub import hf_hub_download
    from molscribe import MolScribe

    ckpt = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
    device = torch.device("cpu")
    model = MolScribe(ckpt, device=device)
    output = model.predict_image_file(image_path)
    smiles = output.get("smiles", "")
    if not smiles:
        raise RuntimeError("MolScribe returned empty SMILES")
    return smiles


def main() -> int:
    parser = argparse.ArgumentParser(description="MolScribe OCSR runner")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image file",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        return 1

    try:
        smiles = run_molscribe(str(image_path))
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(smiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
