"""Unified type system for ChemScheme.

SmilesResult is the single return type for all SMILES operations.
ValidationResult is the return type for all validation layers.
RecognitionResult is the return type for cross-validated recognition (Phase 17).
"""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class SmilesResult:
    """Unified return type for all SMILES entry points.

    Invalid case: smiles="", confidence=0.0, warnings=["invalid SMILES: <reason>"]
    Warnings are stored as a tuple to enforce immutability.
    """

    smiles: str
    confidence: float
    source: str
    warnings: list[str] | tuple[str, ...] = ()  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Freeze mutable list input into tuple for true immutability
        if isinstance(self.warnings, list):
            object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.9


@dataclass(frozen=True)
class ValidationResult:
    """Return type for validation layers (chemistry, reaction, continuity)."""

    layer: str
    errors: list[str] | tuple[str, ...] = ()  # type: ignore[assignment]
    warnings: list[str] | tuple[str, ...] = ()  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if isinstance(self.errors, list):
            object.__setattr__(self, "errors", tuple(self.errors))
        if isinstance(self.warnings, list):
            object.__setattr__(self, "warnings", tuple(self.warnings))


# ---------------------------------------------------------------------------
# Phase 17: Image Recognition Pipeline types
# ---------------------------------------------------------------------------


class AgreementLevel(Enum):
    """Cross-validation agreement level between multiple recognizers."""

    HIGH = "high"  # All tools agree
    MEDIUM = "medium"  # Majority agrees
    LOW = "low"  # All disagree
    SINGLE = "single"  # Only one tool ran


@dataclass(frozen=True)
class RecognitionResult:
    """Return type for cross-validated image recognition.

    Parallels SmilesResult pattern: frozen, tuple for sequences, confidence+source.
    """

    value: str
    confidence: float
    source: str
    agreement: AgreementLevel
    sources: tuple[str, ...] | list[str]
    disagreements: tuple[str, ...] | list[str] = ()  # type: ignore[assignment]
    needs_human_review: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.sources, list):
            object.__setattr__(self, "sources", tuple(self.sources))
        if isinstance(self.disagreements, list):
            object.__setattr__(self, "disagreements", tuple(self.disagreements))
