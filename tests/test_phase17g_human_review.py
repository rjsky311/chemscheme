"""Phase 17g: Human-in-the-Loop Confirm tests.

BDD-first RED tests. ConfidenceGate + ReviewQueue.
"""

from __future__ import annotations

import pytest

from chemscheme.models import AgreementLevel, RecognitionResult
from chemscheme.recognition.review import (
    ConfidenceGate,
    GateDecision,
    ReviewItem,
    ReviewQueue,
)

# ===========================================================================
# AC-17g.1: ConfidenceGate
# ===========================================================================


class TestConfidenceGate:
    """Confidence-gated review decisions."""

    def test_high_confidence_auto_accept(self) -> None:
        gate = ConfidenceGate()
        result = RecognitionResult(
            value="CCO",
            confidence=0.95,
            source="test",
            agreement=AgreementLevel.HIGH,
            sources=("a",),
        )
        assert gate.decide(result) == GateDecision.AUTO_ACCEPT

    def test_medium_confidence_suggest_review(self) -> None:
        gate = ConfidenceGate()
        result = RecognitionResult(
            value="CCO",
            confidence=0.65,
            source="test",
            agreement=AgreementLevel.MEDIUM,
            sources=("a",),
        )
        assert gate.decide(result) == GateDecision.SUGGEST_REVIEW

    def test_low_confidence_require_review(self) -> None:
        gate = ConfidenceGate()
        result = RecognitionResult(
            value="CCO",
            confidence=0.3,
            source="test",
            agreement=AgreementLevel.LOW,
            sources=("a",),
        )
        assert gate.decide(result) == GateDecision.REQUIRE_REVIEW

    def test_boundary_auto_accept(self) -> None:
        """Confidence exactly at auto_accept threshold → AUTO_ACCEPT."""
        gate = ConfidenceGate(auto_accept=0.8)
        result = RecognitionResult(
            value="CCO",
            confidence=0.8,
            source="test",
            agreement=AgreementLevel.HIGH,
            sources=("a",),
        )
        assert gate.decide(result) == GateDecision.AUTO_ACCEPT

    def test_boundary_human_review(self) -> None:
        """Confidence exactly at human_review threshold → SUGGEST_REVIEW."""
        gate = ConfidenceGate(human_review=0.5)
        result = RecognitionResult(
            value="CCO",
            confidence=0.5,
            source="test",
            agreement=AgreementLevel.MEDIUM,
            sources=("a",),
        )
        assert gate.decide(result) == GateDecision.SUGGEST_REVIEW

    def test_custom_thresholds(self) -> None:
        gate = ConfidenceGate(auto_accept=0.9, human_review=0.6)
        result = RecognitionResult(
            value="CCO",
            confidence=0.75,
            source="test",
            agreement=AgreementLevel.MEDIUM,
            sources=("a",),
        )
        assert gate.decide(result) == GateDecision.SUGGEST_REVIEW

    def test_gate_decision_enum(self) -> None:
        assert GateDecision.AUTO_ACCEPT.value == "auto_accept"
        assert GateDecision.SUGGEST_REVIEW.value == "suggest_review"
        assert GateDecision.REQUIRE_REVIEW.value == "require_review"


# ===========================================================================
# AC-17g.2: ReviewQueue
# ===========================================================================


class TestReviewQueue:
    """ReviewQueue collects and resolves items for human review."""

    def test_add_and_pending(self) -> None:
        queue = ReviewQueue()
        result = RecognitionResult(
            value="CCO",
            confidence=0.3,
            source="test",
            agreement=AgreementLevel.LOW,
            sources=("a",),
            needs_human_review=True,
        )
        item_id = queue.add(result, context={"step": 1})
        assert len(queue.pending()) == 1
        assert queue.pending()[0].item_id == item_id

    def test_resolve_item(self) -> None:
        queue = ReviewQueue()
        result = RecognitionResult(
            value="CCO",
            confidence=0.3,
            source="test",
            agreement=AgreementLevel.LOW,
            sources=("a",),
        )
        item_id = queue.add(result)
        queue.resolve(item_id, corrected_value="CCCO")
        assert len(queue.pending()) == 0

    def test_resolve_returns_correction(self) -> None:
        queue = ReviewQueue()
        result = RecognitionResult(
            value="CCO",
            confidence=0.3,
            source="test",
            agreement=AgreementLevel.LOW,
            sources=("a",),
        )
        item_id = queue.add(result)
        resolved = queue.resolve(item_id, corrected_value="CCCO")
        assert resolved.corrected_value == "CCCO"
        assert resolved.original_value == "CCO"

    def test_resolve_nonexistent_raises(self) -> None:
        queue = ReviewQueue()
        with pytest.raises(KeyError):
            queue.resolve("nonexistent_id", corrected_value="CCO")

    def test_multiple_items(self) -> None:
        queue = ReviewQueue()
        for i in range(3):
            result = RecognitionResult(
                value=f"mol_{i}",
                confidence=0.2,
                source="test",
                agreement=AgreementLevel.LOW,
                sources=("a",),
            )
            queue.add(result)
        assert len(queue.pending()) == 3

    def test_review_item_fields(self) -> None:
        queue = ReviewQueue()
        result = RecognitionResult(
            value="CCO",
            confidence=0.3,
            source="ensemble",
            agreement=AgreementLevel.LOW,
            sources=("a", "b"),
            disagreements=("b",),
            needs_human_review=True,
        )
        item_id = queue.add(result, context={"page": 2})
        item = queue.pending()[0]
        assert isinstance(item, ReviewItem)
        assert item.result.value == "CCO"
        assert item.context == {"page": 2}
        assert item.item_id == item_id
