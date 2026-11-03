#!/usr/bin/env python3
"""Entrypoint for DECIMER OCSR container.

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

# Suppress TensorFlow logs that would pollute stdout SMILES output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")


def run_decimer(image_path: str) -> str:
    """Run DECIMER on a single image file."""
    from DECIMER import predict_SMILES

    smiles = predict_SMILES(image_path)
    if not smiles:
        raise RuntimeError("DECIMER returned empty SMILES")
    return smiles


def main() -> int:
    parser = argparse.ArgumentParser(description="DECIMER OCSR runner")
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
        smiles = run_decimer(str(image_path))
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(smiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
