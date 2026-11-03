"""4-tool OCSR benchmark framework (S68 Part 4 — closes S67 Part 5b).

Measures per-tool accuracy + coverage + confidence calibration on a fixed
ground-truth set. Produces a JSON report consumable by the user for keep/drop/
reweight decisions on Mathpix (whose 99% marketing accuracy was unverified
S67 Part 5b).

Real-recognizer execution requires Docker daemon + 3 docker images built:
``docker build -t chemscheme-{molscribe,decimer,molnextr} docker/{molscribe,decimer,molnextr}/``
then set ``CHEMSCHEME_OCSR_DOCKER=1`` and run
``chemscheme benchmark-ocsr --input-dir tests/fixtures/ocsr_benchmark/
--ground-truth tests/fixtures/ocsr_benchmark/ground_truth.json --output report.json``.

Mock mode (``--mock``) skips real-tool dependencies for framework smoke testing.

Reference: MARCUS ensemble methodology (RSC Digital Discovery, Oct 2025).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chemscheme.recognition.cross_validator import smiles_agree
from chemscheme.recognition.protocols import MoleculeRecognizer

logger = logging.getLogger(__name__)


def score_smiles_match(predicted: str, ground_truth: str) -> bool:
    """True iff predicted SMILES is canonically equal to ground_truth.

    Uses ``smiles_agree`` (RDKit canonical SMILES comparison). Empty or invalid
    predictions return False.
    """
    if not predicted or not ground_truth:
        return False
    return smiles_agree(predicted, ground_truth)


@dataclass(frozen=True)
class BenchmarkResult:
    """One row in the benchmark output: (tool, image) result + scoring."""

    tool: str
    image_id: str
    ground_truth: str
    predicted: str
    confidence: float
    match: bool
    elapsed_seconds: float
    error: str | None = None


def run_benchmark(
    images: dict[str, Path],
    ground_truth: dict[str, str],
    recognizers: dict[str, MoleculeRecognizer],
) -> list[BenchmarkResult]:
    """Run each recognizer on each image, score against ground_truth, collect results.

    Args:
        images: image_id → path mapping. Files are read as bytes per call.
        ground_truth: image_id → expected canonical SMILES.
        recognizers: tool_name → MoleculeRecognizer instance.

    Images without a ground_truth entry are silently skipped (no scoring
    possible). Per-call exceptions are recorded in the BenchmarkResult.error
    field, never raised.
    """
    results: list[BenchmarkResult] = []
    for image_id, image_path in images.items():
        if image_id not in ground_truth:
            logger.info("Skipping %s — no ground_truth entry", image_id)
            continue
        gt_smiles = ground_truth[image_id]
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            for tool_name in recognizers:
                results.append(
                    BenchmarkResult(
                        tool=tool_name,
                        image_id=image_id,
                        ground_truth=gt_smiles,
                        predicted="",
                        confidence=0.0,
                        match=False,
                        elapsed_seconds=0.0,
                        error=f"could not read image: {exc}",
                    )
                )
            continue

        for tool_name, recognizer in recognizers.items():
            t0 = time.monotonic()
            try:
                rec_result = recognizer.recognize(image_bytes)
                elapsed = time.monotonic() - t0
                predicted = rec_result.value
                confidence = rec_result.confidence
                match = score_smiles_match(predicted, gt_smiles)
                err: str | None = None
            except Exception as exc:
                elapsed = time.monotonic() - t0
                predicted = ""
                confidence = 0.0
                match = False
                err = str(exc)
            results.append(
                BenchmarkResult(
                    tool=tool_name,
                    image_id=image_id,
                    ground_truth=gt_smiles,
                    predicted=predicted,
                    confidence=confidence,
                    match=match,
                    elapsed_seconds=elapsed,
                    error=err,
                )
            )
    return results


def generate_report(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Aggregate per-tool metrics: accuracy, coverage, confidence calibration.

    accuracy = correct / (correct + incorrect_no_error) — excludes errored attempts
    coverage = (correct + incorrect_no_error) / total — fraction of images the tool answered
    avg_confidence_correct vs avg_confidence_incorrect — calibration probe
    errors = count of recognizer failures (e.g., docker down)
    """
    per_tool: dict[str, dict[str, Any]] = {}
    image_ids: set[str] = set()
    tools_seen: set[str] = set()

    for r in results:
        image_ids.add(r.image_id)
        tools_seen.add(r.tool)
        bucket = per_tool.setdefault(
            r.tool,
            {
                "correct": 0,
                "incorrect": 0,  # answered but wrong
                "errors": 0,
                "total": 0,
                "_conf_correct": [],
                "_conf_incorrect": [],
            },
        )
        bucket["total"] += 1
        if r.error is not None:
            bucket["errors"] += 1
            continue
        if r.match:
            bucket["correct"] += 1
            bucket["_conf_correct"].append(r.confidence)
        else:
            bucket["incorrect"] += 1
            bucket["_conf_incorrect"].append(r.confidence)

    # Finalize per-tool metrics
    final: dict[str, dict[str, Any]] = {}
    for tool, b in per_tool.items():
        attempted = b["correct"] + b["incorrect"]
        accuracy = b["correct"] / attempted if attempted > 0 else 0.0
        coverage = attempted / b["total"] if b["total"] > 0 else 0.0
        conf_c = b["_conf_correct"]
        conf_i = b["_conf_incorrect"]
        final[tool] = {
            "accuracy": accuracy,
            "coverage": coverage,
            "correct": b["correct"],
            "incorrect": b["incorrect"],
            "errors": b["errors"],
            "total": b["total"],
            "avg_confidence_correct": sum(conf_c) / len(conf_c) if conf_c else 0.0,
            "avg_confidence_incorrect": sum(conf_i) / len(conf_i) if conf_i else 0.0,
        }

    return {
        "per_tool": final,
        "summary": {
            "total_images": len(image_ids),
            "total_tools": len(tools_seen),
            "total_attempts": len(results),
        },
    }
