"""Reaction diagram parsing: RxnIM + RxnScribe adapters.

Parses full reaction scheme images to identify steps, reactants,
products, arrows, and condition regions. Cross-validates between
two architecturally different parsers.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from chemscheme.models import AgreementLevel
from chemscheme.recognition.runner import run_tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BBox:
    """Bounding box in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class ParsedStep:
    """A single reaction step parsed from a diagram."""

    reactant_boxes: tuple[BBox, ...]
    product_boxes: tuple[BBox, ...]
    arrow_bbox: BBox
    condition_bbox: BBox | None = None


@dataclass(frozen=True)
class DiagramParseResult:
    """Result of parsing a reaction diagram image."""

    steps: tuple[ParsedStep, ...]
    confidence: float
    source: str
    agreement: AgreementLevel
    needs_human_review: bool = False


@dataclass(frozen=True)
class CroppedRegion:
    """A cropped sub-image from a parsed diagram."""

    bbox: BBox
    role: str  # "reactant", "product", "condition"
    step_index: int


def _parse_json_output(raw: str, source: str) -> DiagramParseResult:
    """Parse JSON output from a diagram parser tool."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return DiagramParseResult(
            steps=(),
            confidence=0.0,
            source=source,
            agreement=AgreementLevel.SINGLE,
        )

    steps_data = data.get("steps", [])
    steps: list[ParsedStep] = []
    for s in steps_data:
        reactant_boxes = tuple(BBox(b[0], b[1], b[2], b[3]) for b in s.get("reactants", []))
        product_boxes = tuple(BBox(b[0], b[1], b[2], b[3]) for b in s.get("products", []))
        arrow = s.get("arrow", [0, 0, 10, 10])
        arrow_bbox = BBox(arrow[0], arrow[1], arrow[2], arrow[3])
        cond = s.get("condition")
        condition_bbox = BBox(cond[0], cond[1], cond[2], cond[3]) if cond else None
        steps.append(
            ParsedStep(
                reactant_boxes=reactant_boxes,
                product_boxes=product_boxes,
                arrow_bbox=arrow_bbox,
                condition_bbox=condition_bbox,
            )
        )

    confidence = 0.85 if steps else 0.0
    return DiagramParseResult(
        steps=tuple(steps),
        confidence=confidence,
        source=source,
        agreement=AgreementLevel.SINGLE,
    )


class RxnIMParser:
    """RxnIM reaction diagram parser via subprocess."""

    def __init__(self, cmd: str = "rxnim") -> None:
        self._cmd = cmd

    def recognize(self, image: bytes, image_format: str = "png") -> DiagramParseResult:
        """Parse reaction diagram from image bytes."""
        with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as tmp:
            tmp.write(image)
            tmp_path = tmp.name

        try:
            output = run_tool([self._cmd, "--image", tmp_path], timeout=120)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if output.returncode != 0:
            logger.warning("RxnIM failed: %s", output.stderr)
            return DiagramParseResult(
                steps=(),
                confidence=0.0,
                source="rxnim",
                agreement=AgreementLevel.SINGLE,
            )

        raw = output.stdout.decode("utf-8", errors="replace").strip()
        return _parse_json_output(raw, "rxnim")


class RxnScribeParser:
    """RxnScribe reaction diagram parser via subprocess."""

    def __init__(self, cmd: str = "rxnscribe") -> None:
        self._cmd = cmd

    def recognize(self, image: bytes, image_format: str = "png") -> DiagramParseResult:
        """Parse reaction diagram from image bytes."""
        with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as tmp:
            tmp.write(image)
            tmp_path = tmp.name

        try:
            output = run_tool([self._cmd, "--image", tmp_path], timeout=120)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if output.returncode != 0:
            logger.warning("RxnScribe failed: %s", output.stderr)
            return DiagramParseResult(
                steps=(),
                confidence=0.0,
                source="rxnscribe",
                agreement=AgreementLevel.SINGLE,
            )

        raw = output.stdout.decode("utf-8", errors="replace").strip()
        return _parse_json_output(raw, "rxnscribe")


def parse_diagram(
    image: bytes,
    parsers: list[RxnIMParser | RxnScribeParser] | None = None,
) -> DiagramParseResult:
    """Cross-validate diagram parsing between multiple parsers.

    Agreement is based on matching step counts. If step counts match,
    HIGH agreement; if they differ, LOW agreement with human review.

    Raises:
        ValueError: If no parsers provided.
    """
    if not parsers:
        msg = "At least one diagram parser is required"
        raise ValueError(msg)

    results = [p.recognize(image) for p in parsers]

    if len(results) == 1:
        r = results[0]
        conf = r.confidence if r.steps else 0.3
        return DiagramParseResult(
            steps=r.steps,
            confidence=conf,
            source=r.source,
            agreement=AgreementLevel.SINGLE,
        )

    # Cross-validate: compare step counts
    step_counts = [len(r.steps) for r in results]
    if len(set(step_counts)) == 1:
        # All agree on step count
        best = max(results, key=lambda r: r.confidence)
        return DiagramParseResult(
            steps=best.steps,
            confidence=best.confidence,
            source=best.source,
            agreement=AgreementLevel.HIGH,
        )
    else:
        # Disagreement on step count
        best = max(results, key=lambda r: r.confidence)
        return DiagramParseResult(
            steps=best.steps,
            confidence=best.confidence * 0.7,
            source=best.source,
            agreement=AgreementLevel.LOW,
            needs_human_review=True,
        )


def crop_regions(image: bytes, parse_result: DiagramParseResult) -> list[CroppedRegion]:
    """Extract cropped regions from a parsed diagram.

    Returns CroppedRegion objects for each reactant, product, and condition
    box. Actual pixel cropping deferred to image processing layer;
    this returns the logical regions with bounding boxes.
    """
    regions: list[CroppedRegion] = []
    for step_idx, step in enumerate(parse_result.steps):
        for bbox in step.reactant_boxes:
            regions.append(CroppedRegion(bbox=bbox, role="reactant", step_index=step_idx))
        for bbox in step.product_boxes:
            regions.append(CroppedRegion(bbox=bbox, role="product", step_index=step_idx))
        if step.condition_bbox:
            regions.append(
                CroppedRegion(bbox=step.condition_bbox, role="condition", step_index=step_idx)
            )
    return regions
