"""Phase 17d: Reaction Diagram Parsing tests.

BDD-first RED tests. RxnIM + RxnScribe adapters via subprocess,
cross-validation, and region cropping.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chemscheme.models import AgreementLevel
from chemscheme.recognition.diagram import (
    BBox,
    CroppedRegion,
    DiagramParseResult,
    ParsedStep,
    RxnIMParser,
    RxnScribeParser,
    crop_regions,
    parse_diagram,
)
from chemscheme.recognition.protocols import DiagramParser

# ===========================================================================
# AC-17d.1: DiagramParseResult data type
# ===========================================================================


class TestDiagramParseResult:
    """DiagramParseResult and supporting data types."""

    def test_bbox_fields(self) -> None:
        b = BBox(x=10, y=20, width=100, height=50)
        assert b.x == 10
        assert b.width == 100

    def test_parsed_step_fields(self) -> None:
        step = ParsedStep(
            reactant_boxes=(BBox(0, 0, 50, 50),),
            product_boxes=(BBox(200, 0, 50, 50),),
            arrow_bbox=BBox(60, 20, 130, 10),
            condition_bbox=BBox(60, 5, 130, 15),
        )
        assert len(step.reactant_boxes) == 1
        assert len(step.product_boxes) == 1
        assert step.condition_bbox is not None

    def test_diagram_parse_result(self) -> None:
        step = ParsedStep(
            reactant_boxes=(BBox(0, 0, 50, 50),),
            product_boxes=(BBox(200, 0, 50, 50),),
            arrow_bbox=BBox(60, 20, 130, 10),
        )
        result = DiagramParseResult(
            steps=(step,),
            confidence=0.88,
            source="rxnim",
            agreement=AgreementLevel.SINGLE,
        )
        assert len(result.steps) == 1
        assert result.confidence == 0.88

    def test_diagram_parse_result_frozen(self) -> None:
        step = ParsedStep(
            reactant_boxes=(),
            product_boxes=(),
            arrow_bbox=BBox(0, 0, 10, 10),
        )
        result = DiagramParseResult(
            steps=(step,),
            confidence=0.8,
            source="test",
            agreement=AgreementLevel.SINGLE,
        )
        with pytest.raises(AttributeError):
            result.confidence = 0.5  # type: ignore[misc]


# ===========================================================================
# AC-17d.2 + 17d.3: Parser adapters
# ===========================================================================


class TestRxnIMParser:
    """RxnIM subprocess adapter."""

    def test_implements_protocol(self) -> None:
        p = RxnIMParser()
        assert isinstance(p, DiagramParser)

    def test_parse_with_mock_subprocess(self) -> None:
        p = RxnIMParser()
        mock_output = (
            '{"steps": [{"reactants": [[0,0,50,50]], '
            '"products": [[200,0,50,50]], '
            '"arrow": [60,20,130,10]}]}'
        )
        with patch("chemscheme.recognition.diagram.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output.encode(),
                stderr="",
                returncode=0,
                elapsed=2.0,
            )
            result = p.recognize(b"fake_image")
            assert result.confidence > 0
            assert result.source == "rxnim"

    def test_parse_tool_not_installed(self) -> None:
        p = RxnIMParser()
        with patch("chemscheme.recognition.diagram.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"",
                stderr="Command not found",
                returncode=-1,
                elapsed=0.0,
            )
            result = p.recognize(b"fake_image")
            assert result.confidence == 0.0


class TestRxnScribeParser:
    """RxnScribe subprocess adapter."""

    def test_implements_protocol(self) -> None:
        p = RxnScribeParser()
        assert isinstance(p, DiagramParser)

    def test_parse_with_mock_subprocess(self) -> None:
        p = RxnScribeParser()
        mock_output = (
            '{"steps": [{"reactants": [[0,0,50,50]], '
            '"products": [[200,0,50,50]], '
            '"arrow": [60,20,130,10]}]}'
        )
        with patch("chemscheme.recognition.diagram.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output.encode(),
                stderr="",
                returncode=0,
                elapsed=3.0,
            )
            result = p.recognize(b"fake_image")
            assert result.confidence > 0
            assert result.source == "rxnscribe"


# ===========================================================================
# AC-17d.4: Cross-validation
# ===========================================================================


class TestParseDiagram:
    """parse_diagram cross-validates RxnIM ‖ RxnScribe."""

    def _make_mock_parser(self, steps: int, source: str, confidence: float = 0.85) -> MagicMock:
        step = ParsedStep(
            reactant_boxes=tuple(BBox(i * 60, 0, 50, 50) for i in range(2)),
            product_boxes=(BBox(200, 0, 50, 50),),
            arrow_bbox=BBox(120, 20, 70, 10),
        )
        result = DiagramParseResult(
            steps=tuple(step for _ in range(steps)),
            confidence=confidence,
            source=source,
            agreement=AgreementLevel.SINGLE,
        )
        mock = MagicMock()
        mock.recognize.return_value = result
        return mock

    def test_both_agree_on_step_count(self) -> None:
        p1 = self._make_mock_parser(2, "rxnim")
        p2 = self._make_mock_parser(2, "rxnscribe")
        result = parse_diagram(b"img", parsers=[p1, p2])
        assert result.agreement == AgreementLevel.HIGH
        assert len(result.steps) == 2

    def test_parsers_disagree_on_step_count(self) -> None:
        p1 = self._make_mock_parser(2, "rxnim")
        p2 = self._make_mock_parser(3, "rxnscribe")
        result = parse_diagram(b"img", parsers=[p1, p2])
        assert result.agreement == AgreementLevel.LOW
        assert result.needs_human_review is True

    def test_single_parser(self) -> None:
        p1 = self._make_mock_parser(1, "rxnim")
        result = parse_diagram(b"img", parsers=[p1])
        assert result.agreement == AgreementLevel.SINGLE

    def test_no_parsers_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_diagram(b"img", parsers=[])

    def test_parser_returns_no_steps(self) -> None:
        p1 = self._make_mock_parser(0, "rxnim")
        result = parse_diagram(b"img", parsers=[p1])
        assert len(result.steps) == 0
        assert result.confidence < 0.5


# ===========================================================================
# AC-17d.5: Crop regions
# ===========================================================================


class TestCropRegions:
    """crop_regions extracts sub-images from parsed diagram."""

    def test_crop_single_step(self) -> None:
        step = ParsedStep(
            reactant_boxes=(BBox(0, 0, 50, 50), BBox(60, 0, 50, 50)),
            product_boxes=(BBox(200, 0, 50, 50),),
            arrow_bbox=BBox(120, 20, 70, 10),
            condition_bbox=BBox(120, 5, 70, 15),
        )
        result = DiagramParseResult(
            steps=(step,),
            confidence=0.9,
            source="test",
            agreement=AgreementLevel.HIGH,
        )
        # 400x100 fake image (just bytes, crop is logical)
        regions = crop_regions(b"x" * 40000, result)
        # 2 reactants + 1 product + 1 condition = 4 regions
        assert len(regions) == 4
        roles = [r.role for r in regions]
        assert roles.count("reactant") == 2
        assert roles.count("product") == 1
        assert roles.count("condition") == 1

    def test_crop_multi_step(self) -> None:
        step1 = ParsedStep(
            reactant_boxes=(BBox(0, 0, 50, 50),),
            product_boxes=(BBox(200, 0, 50, 50),),
            arrow_bbox=BBox(60, 20, 130, 10),
        )
        step2 = ParsedStep(
            reactant_boxes=(BBox(0, 100, 50, 50),),
            product_boxes=(BBox(200, 100, 50, 50),),
            arrow_bbox=BBox(60, 120, 130, 10),
        )
        result = DiagramParseResult(
            steps=(step1, step2),
            confidence=0.85,
            source="test",
            agreement=AgreementLevel.HIGH,
        )
        regions = crop_regions(b"x" * 40000, result)
        assert len(regions) == 4  # 2 reactants + 2 products, no conditions
        assert all(isinstance(r, CroppedRegion) for r in regions)
        assert regions[0].step_index == 0
        assert regions[2].step_index == 1

    def test_crop_empty_steps(self) -> None:
        result = DiagramParseResult(
            steps=(),
            confidence=0.0,
            source="test",
            agreement=AgreementLevel.LOW,
        )
        regions = crop_regions(b"x", result)
        assert len(regions) == 0
