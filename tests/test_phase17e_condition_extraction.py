"""Phase 17e: Condition Text Extraction tests.

BDD-first RED tests. Claude Vision + EasyOCR cross-validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chemscheme.models import AgreementLevel
from chemscheme.recognition.conditions import (
    ClaudeConditionExtractor,
    ConditionResult,
    EasyOCRExtractor,
    extract_conditions,
)
from chemscheme.recognition.protocols import TextExtractor

# ===========================================================================
# AC-17e.1: ConditionResult data type
# ===========================================================================


class TestConditionResult:
    """ConditionResult frozen dataclass."""

    def test_create_full(self) -> None:
        r = ConditionResult(
            reagents=("K2CO3",),
            solvents=("DMF",),
            temperature="80°C",
            time="12h",
            atmosphere="N2",
            catalyst="Pd(PPh3)4",
            raw_text="K2CO3, DMF, 80°C, 12h, N2",
            confidence=0.9,
            source="claude_vision",
        )
        assert r.temperature == "80°C"
        assert len(r.reagents) == 1

    def test_create_minimal(self) -> None:
        r = ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="",
            confidence=0.0,
            source="test",
        )
        assert r.temperature is None
        assert r.catalyst is None

    def test_frozen(self) -> None:
        r = ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="",
            confidence=0.0,
            source="t",
        )
        with pytest.raises(AttributeError):
            r.temperature = "100°C"  # type: ignore[misc]

    def test_list_to_tuple_coercion(self) -> None:
        r = ConditionResult(
            reagents=["K2CO3", "NaOH"],  # type: ignore[arg-type]
            solvents=["DMF"],  # type: ignore[arg-type]
            raw_text="test",
            confidence=0.8,
            source="test",
        )
        assert isinstance(r.reagents, tuple)
        assert isinstance(r.solvents, tuple)


# ===========================================================================
# AC-17e.2 + 17e.3: Extractor adapters
# ===========================================================================


class TestClaudeConditionExtractor:
    """Claude Vision condition extractor."""

    def test_implements_protocol(self) -> None:
        e = ClaudeConditionExtractor()
        assert isinstance(e, TextExtractor)

    def test_extract_with_mock_api(self) -> None:
        e = ClaudeConditionExtractor()
        with patch.object(e, "_call_claude") as mock_call:
            mock_call.return_value = {
                "reagents": ["K2CO3"],
                "solvents": ["DMF"],
                "temperature": "80°C",
                "time": "12h",
            }
            result = e.extract(b"fake_image")
            assert isinstance(result, ConditionResult)
            assert result.reagents == ("K2CO3",)
            assert result.temperature == "80°C"

    def test_extract_api_failure(self) -> None:
        e = ClaudeConditionExtractor()
        with patch.object(e, "_call_claude", side_effect=Exception("API error")):
            result = e.extract(b"fake_image")
            assert result.confidence == 0.0


class TestEasyOCRExtractor:
    """EasyOCR condition extractor."""

    def test_implements_protocol(self) -> None:
        e = EasyOCRExtractor()
        assert isinstance(e, TextExtractor)

    def test_extract_with_mock(self) -> None:
        e = EasyOCRExtractor()
        with patch.object(e, "_run_ocr") as mock_ocr:
            mock_ocr.return_value = "K2CO3 DMF 80°C 12h"
            result = e.extract(b"fake_image")
            assert isinstance(result, ConditionResult)
            assert "K2CO3" in result.raw_text

    def test_extract_failure(self) -> None:
        e = EasyOCRExtractor()
        with patch.object(e, "_run_ocr", side_effect=Exception("OCR error")):
            result = e.extract(b"fake_image")
            assert result.confidence == 0.0


# ===========================================================================
# AC-17e.4: Cross-validation
# ===========================================================================


class TestExtractConditions:
    """extract_conditions cross-validates Claude Vision ‖ EasyOCR."""

    def test_both_agree(self) -> None:
        mock_claude = MagicMock()
        mock_claude.extract.return_value = ConditionResult(
            reagents=("K2CO3",),
            solvents=("DMF",),
            temperature="80°C",
            time="12h",
            raw_text="K2CO3, DMF, 80°C, 12h",
            confidence=0.9,
            source="claude_vision",
        )
        mock_ocr = MagicMock()
        mock_ocr.extract.return_value = ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="K2CO3 DMF 80°C 12h",
            confidence=0.7,
            source="easyocr",
        )
        result = extract_conditions(
            b"img",
            claude_extractor=mock_claude,
            ocr_extractor=mock_ocr,
        )
        assert result.agreement == AgreementLevel.HIGH
        assert result.reagents == ("K2CO3",)

    def test_claude_hallucinates(self) -> None:
        """Claude outputs token not in OCR → lower confidence."""
        mock_claude = MagicMock()
        mock_claude.extract.return_value = ConditionResult(
            reagents=("K2CO3", "Pd(PPh3)4"),
            solvents=("DMF",),
            raw_text="K2CO3, Pd(PPh3)4, DMF",
            confidence=0.9,
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
        assert result.confidence < 0.9  # penalized

    def test_ocr_has_extra_token(self) -> None:
        """OCR detects token Claude missed → MEDIUM."""
        mock_claude = MagicMock()
        mock_claude.extract.return_value = ConditionResult(
            reagents=("K2CO3",),
            solvents=("DMF",),
            raw_text="K2CO3, DMF",
            confidence=0.9,
            source="claude_vision",
        )
        mock_ocr = MagicMock()
        mock_ocr.extract.return_value = ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="K2CO3 DMF N2",
            confidence=0.7,
            source="easyocr",
        )
        result = extract_conditions(
            b"img",
            claude_extractor=mock_claude,
            ocr_extractor=mock_ocr,
        )
        assert result.agreement == AgreementLevel.MEDIUM

    def test_empty_condition_region(self) -> None:
        mock_claude = MagicMock()
        mock_claude.extract.return_value = ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="",
            confidence=0.1,
            source="claude_vision",
        )
        mock_ocr = MagicMock()
        mock_ocr.extract.return_value = ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="",
            confidence=0.1,
            source="easyocr",
        )
        result = extract_conditions(
            b"img",
            claude_extractor=mock_claude,
            ocr_extractor=mock_ocr,
        )
        assert result.confidence < 0.5

    def test_single_extractor(self) -> None:
        mock_claude = MagicMock()
        mock_claude.extract.return_value = ConditionResult(
            reagents=("K2CO3",),
            solvents=("DMF",),
            raw_text="K2CO3, DMF",
            confidence=0.85,
            source="claude_vision",
        )
        result = extract_conditions(
            b"img",
            claude_extractor=mock_claude,
            ocr_extractor=None,
        )
        assert result.agreement == AgreementLevel.SINGLE
