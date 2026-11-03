#!/usr/bin/env python3
"""Unified entrypoint for OCSR tools (MolScribe + DECIMER).

Usage:
    python entrypoint.py --tool molscribe --image /data/input.png
    python entrypoint.py --tool decimer --image /data/input.png

Prints SMILES to stdout on success. Errors go to stderr with exit code 1.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress ML framework logs that would pollute stdout SMILES output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

TOOLS = ("molscribe", "decimer")


def run_molscribe(image_path: str) -> str:
    """Run MolScribe on a single image file."""
    import torch
    from huggingface_hub import hf_hub_download
    from molscribe import MolScribe

    # Download checkpoint from HuggingFace (cached after first download)
    ckpt = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
    device = torch.device("cpu")
    model = MolScribe(ckpt, device=device)
    output = model.predict_image_file(image_path)
    smiles = output.get("smiles", "")
    if not smiles:
        raise RuntimeError("MolScribe returned empty SMILES")
    return smiles


def run_decimer(image_path: str) -> str:
    """Run DECIMER on a single image file."""
    from DECIMER import predict_SMILES

    smiles = predict_SMILES(image_path)
    if not smiles:
        raise RuntimeError("DECIMER returned empty SMILES")
    return smiles


def main() -> int:
    parser = argparse.ArgumentParser(description="OCSR tool runner")
    parser.add_argument(
        "--tool",
        required=True,
        choices=TOOLS,
        help="Which OCSR tool to run",
    )
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
        if args.tool == "molscribe":
            smiles = run_molscribe(str(image_path))
        else:
            smiles = run_decimer(str(image_path))
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        # RuntimeError: empty SMILES / model failure
        # ValueError: invalid input data
        # OSError: file I/O / model checkpoint issues
        # ImportError: missing OCSR package
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(smiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
