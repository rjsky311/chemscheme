"""Human-in-the-loop review: ConfidenceGate + ReviewQueue.

Confidence-gated review ensures high-throughput for confident results
and surfaces ambiguous results for human judgment.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

from chemscheme.models import RecognitionResult


class GateDecision(Enum):
    """Decision from ConfidenceGate."""

    AUTO_ACCEPT = "auto_accept"
    SUGGEST_REVIEW = "suggest_review"
    REQUIRE_REVIEW = "require_review"


class ConfidenceGate:
    """Confidence-gated review decisions.

    Routes recognition results based on confidence thresholds:
    - confidence >= auto_accept → AUTO_ACCEPT
    - human_review <= confidence < auto_accept → SUGGEST_REVIEW
    - confidence < human_review → REQUIRE_REVIEW
    """

    def __init__(
        self,
        auto_accept: float = 0.8,
        human_review: float = 0.5,
    ) -> None:
        self._auto_accept = auto_accept
        self._human_review = human_review

    def decide(self, result: RecognitionResult) -> GateDecision:
        """Determine review action for a recognition result.

        C3 fix: respect needs_human_review flag from upstream cross-validation.
        """
        if result.needs_human_review:
            return GateDecision.REQUIRE_REVIEW
        if result.confidence >= self._auto_accept:
            return GateDecision.AUTO_ACCEPT
        if result.confidence >= self._human_review:
            return GateDecision.SUGGEST_REVIEW
        return GateDecision.REQUIRE_REVIEW


@dataclass
class ReviewItem:
    """An item in the review queue awaiting human judgment."""

    item_id: str
    result: RecognitionResult
    context: dict = field(default_factory=dict)  # type: ignore[type-arg]


@dataclass
class ResolvedItem:
    """A review item after human correction."""

    item_id: str
    original_value: str
    corrected_value: str


class ReviewQueue:
    """Collects recognition results that need human review.

    Items can be added, listed as pending, and resolved with corrections.
    """

    def __init__(self) -> None:
        self._pending: dict[str, ReviewItem] = {}

    def add(
        self,
        result: RecognitionResult,
        context: dict | None = None,  # type: ignore[type-arg]
    ) -> str:
        """Add a result to the review queue.

        Returns:
            Unique item ID for later resolution.
        """
        item_id = str(uuid.uuid4())[:8]
        self._pending[item_id] = ReviewItem(
            item_id=item_id,
            result=result,
            context=context or {},
        )
        return item_id

    def pending(self) -> list[ReviewItem]:
        """Return all items awaiting review."""
        return list(self._pending.values())

    def resolve(self, item_id: str, corrected_value: str) -> ResolvedItem:
        """Resolve a review item with a human correction.

        Args:
            item_id: ID of the item to resolve.
            corrected_value: The corrected value from the human reviewer.

        Returns:
            ResolvedItem with original and corrected values.

        Raises:
            KeyError: If item_id not found in pending queue.
        """
        if item_id not in self._pending:
            msg = f"Review item not found: {item_id}"
            raise KeyError(msg)

        item = self._pending.pop(item_id)
        return ResolvedItem(
            item_id=item_id,
            original_value=item.result.value,
            corrected_value=corrected_value,
        )
