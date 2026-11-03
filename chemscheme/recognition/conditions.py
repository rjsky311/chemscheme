"""Condition text extraction: Claude Vision + EasyOCR.

Extracts reaction conditions (reagents, solvents, temperature, time)
from text near arrows. Cross-validates semantic AI output against
pixel-faithful OCR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chemscheme.models import AgreementLevel
from chemscheme.recognition.cross_validator import text_token_contain

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionResult:
    """Extracted reaction conditions."""

    reagents: tuple[str, ...] | list[str]
    solvents: tuple[str, ...] | list[str]
    raw_text: str
    confidence: float
    source: str
    temperature: str | None = None
    time: str | None = None
    atmosphere: str | None = None
    catalyst: str | None = None
    agreement: AgreementLevel = AgreementLevel.SINGLE
    needs_human_review: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.reagents, list):
            object.__setattr__(self, "reagents", tuple(self.reagents))
        if isinstance(self.solvents, list):
            object.__setattr__(self, "solvents", tuple(self.solvents))


class ClaudeConditionExtractor:
    """Extract structured conditions using Claude Vision API."""

    def _call_claude(self, image: bytes) -> dict:  # type: ignore[type-arg]
        """Call Claude API with vision. Override in tests."""
        msg = "Claude API not configured"
        raise NotImplementedError(msg)

    def extract(self, image: bytes) -> ConditionResult:
        """Extract conditions from image via Claude Vision."""
        try:
            data = self._call_claude(image)
        except Exception as exc:
            logger.warning("Claude condition extraction failed: %s", exc)
            return ConditionResult(
                reagents=(),
                solvents=(),
                raw_text="",
                confidence=0.0,
                source="claude_vision",
            )

        reagents = tuple(data.get("reagents", []))
        solvents = tuple(data.get("solvents", []))
        temp = data.get("temperature")
        time_val = data.get("time")
        atmosphere = data.get("atmosphere")
        catalyst = data.get("catalyst")

        parts = [*reagents, *solvents]
        if temp:
            parts.append(temp)
        if time_val:
            parts.append(time_val)
        raw_text = ", ".join(parts)

        return ConditionResult(
            reagents=reagents,
            solvents=solvents,
            temperature=temp,
            time=time_val,
            atmosphere=atmosphere,
            catalyst=catalyst,
            raw_text=raw_text,
            confidence=0.85,
            source="claude_vision",
        )

    def recognize(self, image: bytes, image_format: str = "png") -> ConditionResult:
        """TextExtractor Protocol compatibility."""
        return self.extract(image)


class EasyOCRExtractor:
    """Extract raw text using EasyOCR (pixel-faithful, no hallucination)."""

    def _run_ocr(self, image: bytes) -> str:
        """Run EasyOCR. Override in tests."""
        msg = "EasyOCR not configured"
        raise NotImplementedError(msg)

    def extract(self, image: bytes) -> ConditionResult:
        """Extract raw text from image via EasyOCR."""
        try:
            raw_text = self._run_ocr(image)
        except Exception as exc:
            logger.warning("EasyOCR extraction failed: %s", exc)
            return ConditionResult(
                reagents=(),
                solvents=(),
                raw_text="",
                confidence=0.0,
                source="easyocr",
            )

        return ConditionResult(
            reagents=(),
            solvents=(),
            raw_text=raw_text,
            confidence=0.7,
            source="easyocr",
        )

    def recognize(self, image: bytes, image_format: str = "png") -> ConditionResult:
        """TextExtractor Protocol compatibility."""
        return self.extract(image)


def extract_conditions(
    image: bytes,
    claude_extractor: ClaudeConditionExtractor | None = None,
    ocr_extractor: EasyOCRExtractor | None = None,
) -> ConditionResult:
    """Cross-validate condition extraction between Claude Vision and EasyOCR.

    Agreement logic:
    - All OCR tokens in Claude output → HIGH
    - Claude misses OCR tokens → MEDIUM
    - Claude adds tokens not in OCR → lower confidence (hallucination)
    """
    results: list[ConditionResult] = []

    if claude_extractor:
        results.append(claude_extractor.extract(image))
    if ocr_extractor:
        results.append(ocr_extractor.extract(image))

    if not results:
        return ConditionResult(
            reagents=(),
            solvents=(),
            raw_text="",
            confidence=0.0,
            source="none",
        )

    if len(results) == 1:
        r = results[0]
        return ConditionResult(
            reagents=r.reagents,
            solvents=r.solvents,
            temperature=r.temperature,
            time=r.time,
            atmosphere=r.atmosphere,
            catalyst=r.catalyst,
            raw_text=r.raw_text,
            confidence=r.confidence,
            source=r.source,
            agreement=AgreementLevel.SINGLE,
        )

    # Two extractors: Claude (structured) + OCR (raw text)
    claude_result = results[0]
    ocr_result = results[1]

    ocr_text = ocr_result.raw_text
    claude_text = claude_result.raw_text

    # G2 fix: short-circuit if Claude failed entirely
    if claude_result.confidence == 0.0 and ocr_result.confidence > 0:
        return ConditionResult(
            reagents=ocr_result.reagents,
            solvents=ocr_result.solvents,
            raw_text=ocr_result.raw_text,
            confidence=ocr_result.confidence * 0.8,
            source="easyocr_fallback",
            agreement=AgreementLevel.SINGLE,
        )

    # Check if all OCR tokens appear in Claude output
    ocr_in_claude = text_token_contain(ocr_text, claude_text)
    # Check if Claude has tokens not in OCR (hallucination)
    claude_in_ocr = text_token_contain(claude_text, ocr_text) if claude_text else True

    # C7 fix: handle complete disagreement first (both directions fail)
    if not ocr_in_claude and not claude_in_ocr:
        agreement = AgreementLevel.LOW
        confidence = 0.4
        needs_review = True
    elif ocr_in_claude and claude_in_ocr:
        agreement = AgreementLevel.HIGH
        confidence = claude_result.confidence
        needs_review = False
    elif ocr_in_claude and not claude_in_ocr:
        # Claude hallucinated extra tokens
        agreement = AgreementLevel.MEDIUM
        confidence = claude_result.confidence * 0.7
        needs_review = False
    else:
        # Claude missed OCR tokens
        agreement = AgreementLevel.MEDIUM
        confidence = claude_result.confidence * 0.8
        needs_review = False

    # C8 fix: include all extracted fields in raw_text
    parts = [*claude_result.reagents, *claude_result.solvents]
    for field_val in (
        claude_result.temperature,
        claude_result.time,
        claude_result.atmosphere,
        claude_result.catalyst,
    ):
        if field_val:
            parts.append(field_val)
    complete_raw_text = ", ".join(parts) if parts else claude_result.raw_text

    return ConditionResult(
        reagents=claude_result.reagents,
        solvents=claude_result.solvents,
        temperature=claude_result.temperature,
        time=claude_result.time,
        atmosphere=claude_result.atmosphere,
        catalyst=claude_result.catalyst,
        raw_text=complete_raw_text,
        confidence=confidence,
        source="cross_validated",
        agreement=agreement,
        needs_human_review=needs_review,
    )
