"""Mock recognizers for testing without real ML models.

Each mock implements its respective Protocol and returns configurable results.
"""

from __future__ import annotations

import time

from chemscheme.models import AgreementLevel, RecognitionResult


class MockMoleculeRecognizer:
    """Mock molecule recognizer returning configurable SMILES."""

    def __init__(
        self,
        value: str = "CCO",
        confidence: float = 0.9,
        source: str = "mock_molecule",
        delay: float = 0.0,
    ) -> None:
        self._value = value
        self._confidence = confidence
        self._source = source
        self._delay = delay

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        if self._delay > 0:
            time.sleep(self._delay)
        return RecognitionResult(
            value=self._value,
            confidence=self._confidence,
            source=self._source,
            agreement=AgreementLevel.SINGLE,
            sources=(self._source,),
        )


class MockDiagramParser:
    """Mock diagram parser returning configurable step count."""

    def __init__(
        self,
        steps: int = 1,
        confidence: float = 0.85,
        source: str = "mock_diagram",
    ) -> None:
        self._steps = steps
        self._confidence = confidence
        self._source = source

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        value = f"{self._steps} steps detected"
        return RecognitionResult(
            value=value,
            confidence=self._confidence,
            source=self._source,
            agreement=AgreementLevel.SINGLE,
            sources=(self._source,),
        )


class MockTextExtractor:
    """Mock text extractor returning configurable text."""

    def __init__(
        self,
        text: str = "",
        confidence: float = 0.8,
        source: str = "mock_text",
    ) -> None:
        self._text = text
        self._confidence = confidence
        self._source = source

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        return RecognitionResult(
            value=self._text,
            confidence=self._confidence,
            source=self._source,
            agreement=AgreementLevel.SINGLE,
            sources=(self._source,),
        )
