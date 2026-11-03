"""MR16 regression tests for Phase 17 review fixes.

# RED-EXEMPT: review finding regression tests
"""

from __future__ import annotations

from unittest.mock import MagicMock

from chemscheme.models import AgreementLevel, RecognitionResult
from chemscheme.recognition.conditions import ConditionResult, extract_conditions
from chemscheme.recognition.cross_validator import cross_validate, smiles_agree
from chemscheme.recognition.review import ConfidenceGate, GateDecision


class TestMR16Regressions:
    """Regression tests for MR16 review findings."""

    def test_c1_recognize_molecule_accepts_image(self) -> None:
        """C1/G1: recognize_molecule has image parameter."""
        from chemscheme.recognition.mocks import MockMoleculeRecognizer
        from chemscheme.recognition.ocsr import recognize_molecule

        mock = MockMoleculeRecognizer(value="CCO", confidence=0.9)
        result = recognize_molecule(
            image=b"\x89PNG_fake_image",
            recognizers=[mock],
        )
        assert result.value == "CCO"

    def test_c3_gate_respects_needs_human_review(self) -> None:
        """C3: ConfidenceGate checks needs_human_review flag."""
        gate = ConfidenceGate(auto_accept=0.8)
        result = RecognitionResult(
            value="CCO",
            confidence=0.95,  # high confidence but flagged
            source="test",
            agreement=AgreementLevel.LOW,
            sources=("a", "b"),
            needs_human_review=True,
        )
        assert gate.decide(result) == GateDecision.REQUIRE_REVIEW

    def test_c4_lookup_returns_smiles_not_name(self) -> None:
        """C4: StructureLookup returns canonical SMILES, not name."""
        import tempfile
        from pathlib import Path

        from chemscheme.recognition.lookup import StructureLookup, build_lookup_db

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.json"
            build_lookup_db(["CCO"], names=["ethanol"], output_path=db_path)
            lookup = StructureLookup(db_path)
            result = lookup.lookup("CCO")
            assert result.value == "CCO"
            assert "ethanol" not in result.value

    def test_c7_conditions_low_branch_reachable(self) -> None:
        """C7: complete disagreement → AgreementLevel.LOW."""
        mock_claude = MagicMock()
        mock_claude.extract.return_value = ConditionResult(
            reagents=("Pd(PPh3)4",),
            solvents=("THF",),
            raw_text="Pd(PPh3)4, THF",
            confidence=0.85,
            source="claude_vision",
        )
        mock_ocr = MagicMock()
        mock_ocr.extract.return_value = ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="K2CO3 DMF 80C",
            confidence=0.7,
            source="easyocr",
        )
        result = extract_conditions(
            b"img",
            claude_extractor=mock_claude,
            ocr_extractor=mock_ocr,
        )
        assert result.agreement == AgreementLevel.LOW
        assert result.needs_human_review is True

    def test_g2_claude_failure_uses_ocr(self) -> None:
        """G2: when Claude fails, OCR result is used as fallback."""
        mock_claude = MagicMock()
        mock_claude.extract.return_value = ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="",
            confidence=0.0,
            source="claude_vision",
        )
        mock_ocr = MagicMock()
        mock_ocr.extract.return_value = ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="K2CO3 DMF",
            confidence=0.7,
            source="easyocr",
        )
        result = extract_conditions(
            b"img",
            claude_extractor=mock_claude,
            ocr_extractor=mock_ocr,
        )
        assert result.confidence > 0
        assert "K2CO3" in result.raw_text

    def test_g3_tiebreak_by_confidence(self) -> None:
        """G3: cross_validate tiebreaks equal groups by confidence."""
        r_low = RecognitionResult(
            value="CCO",
            confidence=0.5,
            source="a",
            agreement=AgreementLevel.SINGLE,
            sources=("a",),
        )
        r_high = RecognitionResult(
            value="CCCO",
            confidence=0.9,
            source="b",
            agreement=AgreementLevel.SINGLE,
            sources=("b",),
        )
        # Both disagree, groups of equal size=1, should pick higher confidence
        merged = cross_validate([r_low, r_high], smiles_agree)
        assert merged.value == "CCCO"  # higher confidence wins
