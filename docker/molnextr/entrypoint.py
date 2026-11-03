#!/usr/bin/env python3
"""Entrypoint for MolNexTR OCSR container (S68 — closes S67 Part 5a).

Usage:
    python entrypoint.py --image /data/input.png

Prints SMILES to stdout on success. Errors go to stderr with exit code 1.

MolNexTR (Chen et al., 2024): ConvNext + ViT + Transformer-decoder.
MARCUS ensemble (RSC Digital Discovery, Oct 2025) — 67% SMILES accuracy
on real chemistry papers, highest single-tool open-source recognizer.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress noisy ML logs that would pollute stdout SMILES output
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")


def run_molnextr(image_path: str) -> str:
    """Run MolNexTR on a single image file → SMILES string.

    Imports the upstream MolNexTR API at call time so dockerfile-build failures
    surface as a clean runtime error instead of an import-time crash.
    """
    try:
        # MolNexTR upstream API — adjust if upstream renames the entry point.
        # Reference: https://github.com/CYF2000127/MolNexTR
        from molnextr import MolNexTR  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            f"MolNexTR Python package not importable: {exc}. "
            "Verify the docker build's `pip install git+...MolNexTR` step succeeded "
            "and the upstream module name matches the import."
        )
        raise RuntimeError(msg) from exc

    model = MolNexTR()
    smiles = model.predict_smiles(image_path)
    if not smiles:
        raise RuntimeError("MolNexTR returned empty SMILES")
    return str(smiles).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="MolNexTR OCSR runner")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image file (PNG/JPEG)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        return 1

    try:
        smiles = run_molnextr(str(image_path))
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(smiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
