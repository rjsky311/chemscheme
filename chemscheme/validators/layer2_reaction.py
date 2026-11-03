"""Layer 2 — Reaction feasibility validation.

AC-6b.1: ReactionPredictor Protocol + MockPredictor.
AC-6b.2: Validate reaction with predictor (mocked).
AC-6b.3: Graceful fallback on predictor failure.

Uses Protocol for IBM RXN / ASKCOS abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from chemscheme.models import ValidationResult


@dataclass(frozen=True)
class PredictionResult:
    """Result from a reaction predictor.

    predicted_products is stored as tuple for immutability (MR3-8).
    """

    feasible: bool
    confidence: float
    predicted_products: tuple[str, ...] = ()
    source: str = "unknown"

    def __post_init__(self) -> None:
        # Normalize list inputs to tuple
        if isinstance(self.predicted_products, list):
            object.__setattr__(self, "predicted_products", tuple(self.predicted_products))


@runtime_checkable
class ReactionPredictor(Protocol):
    """Protocol for reaction prediction services (IBM RXN, ASKCOS, etc.)."""

    def predict_products(self, reactants: list[str], reagents: str) -> PredictionResult: ...


class MockPredictor:
    """Mock predictor for testing. Always returns feasible."""

    def predict_products(self, reactants: list[str], reagents: str) -> PredictionResult:
        return PredictionResult(
            feasible=True,
            confidence=0.8,
            predicted_products=(),
            source="mock",
        )


def validate_reaction(
    reactants: list[str],
    products: list[str],
    reagents: str,
    predictor: ReactionPredictor | None = None,
) -> ValidationResult:
    """Validate reaction feasibility using a predictor.

    If predictor is None or fails, produces warnings (not errors).
    """
    warnings: list[str] = []

    if predictor is None:
        return ValidationResult(
            layer="reaction",
            errors=[],
            warnings=["Reaction prediction skipped: no predictor configured"],
        )

    try:
        result = predictor.predict_products(reactants=reactants, reagents=reagents)
    except Exception as e:
        return ValidationResult(
            layer="reaction",
            errors=[],
            warnings=[f"Reaction predictor unavailable: {e}. Prediction skipped."],
        )

    if not result.feasible:
        warnings.append(
            f"Reaction may not be feasible (confidence: {result.confidence:.2f}, "
            f"source: {result.source})"
        )

    # Compare predicted products to expected products (canonicalize first)
    if result.feasible and result.predicted_products and products:
        from chemscheme.parsers.smiles_utils import canonicalize as _canon

        predicted_set = {_canon(s).smiles or s for s in result.predicted_products}
        expected_set = {_canon(s).smiles or s for s in products}
        if predicted_set != expected_set:
            missing = expected_set - predicted_set
            unexpected = predicted_set - expected_set
            parts: list[str] = []
            if missing:
                parts.append(f"expected but not predicted: {sorted(missing)}")
            if unexpected:
                parts.append(f"predicted but not expected: {sorted(unexpected)}")
            warnings.append(f"Product mismatch ({result.source}): {'; '.join(parts)}")

    return ValidationResult(layer="reaction", errors=[], warnings=warnings)
