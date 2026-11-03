"""Protocol interfaces for image recognition tools.

All recognizers implement RecognizerProtocol. Specialized sub-protocols
(MoleculeRecognizer, DiagramParser, TextExtractor, FigureClassifier)
add semantic meaning but share the same method signature.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from chemscheme.models import RecognitionResult


@runtime_checkable
class RecognizerProtocol(Protocol):
    """Base protocol for all recognition tools."""

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult: ...


@runtime_checkable
class MoleculeRecognizer(RecognizerProtocol, Protocol):
    """Recognizes molecule images → SMILES."""

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult: ...


@runtime_checkable
class DiagramParser(RecognizerProtocol, Protocol):
    """Parses reaction diagram images → structured steps."""

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult: ...


@runtime_checkable
class TextExtractor(RecognizerProtocol, Protocol):
    """Extracts text from condition/label images."""

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult: ...


@runtime_checkable
class FigureClassifier(RecognizerProtocol, Protocol):
    """Classifies figure type (reaction scheme, spectrum, etc.)."""

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult: ...
