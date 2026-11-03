"""Cross-validation framework for recognition results.

Merges multiple RecognitionResults using domain-specific agreement functions.
Implements confidence scoring across recognition backends.
"""

from __future__ import annotations

from collections.abc import Callable

from rdkit import Chem

from chemscheme.models import AgreementLevel, RecognitionResult


def cross_validate(
    results: list[RecognitionResult],
    agree_fn: Callable[[str, str], bool],
) -> RecognitionResult:
    """Merge multiple recognition results with agreement scoring.

    Args:
        results: List of individual recognition results.
        agree_fn: Domain-specific equality function (e.g., smiles_agree).

    Returns:
        Merged RecognitionResult with agreement level and provenance.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        msg = "No results to cross-validate"
        raise ValueError(msg)

    if len(results) == 1:
        r = results[0]
        return RecognitionResult(
            value=r.value,
            confidence=r.confidence,
            source=r.source,
            agreement=AgreementLevel.SINGLE,
            sources=(r.source,),
            disagreements=(),
            needs_human_review=False,
        )

    # Group results by agreement
    n = len(results)
    groups: list[list[int]] = []
    assigned: set[int] = set()

    for i in range(n):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i + 1, n):
            if j in assigned:
                continue
            if agree_fn(results[i].value, results[j].value):
                group.append(j)
                assigned.add(j)
        groups.append(group)

    # Find largest agreement group (G3 fix: tiebreak by max confidence)
    groups.sort(
        key=lambda g: (len(g), max(results[i].confidence for i in g)),
        reverse=True,
    )
    majority_group = groups[0]
    majority_size = len(majority_group)

    # Collect all sources
    all_sources = tuple(r.source for r in results)

    # Determine agreement level
    if majority_size == n:
        agreement = AgreementLevel.HIGH
    elif majority_size > 1:
        agreement = AgreementLevel.MEDIUM
    else:
        agreement = AgreementLevel.LOW

    # Pick value and confidence from majority
    majority_results = [results[i] for i in majority_group]
    best = max(majority_results, key=lambda r: r.confidence)

    # Disagreements: sources not in majority
    disagreement_sources = tuple(results[i].source for g in groups[1:] for i in g)

    needs_review = agreement == AgreementLevel.LOW

    return RecognitionResult(
        value=best.value,
        confidence=best.confidence,
        source=best.source,
        agreement=agreement,
        sources=all_sources,
        disagreements=disagreement_sources,
        needs_human_review=needs_review,
    )


# ---------------------------------------------------------------------------
# Domain-specific agreement functions (AC-17a.4)
# ---------------------------------------------------------------------------


def smiles_agree(a: str, b: str) -> bool:
    """Check if two SMILES represent the same molecule via RDKit canonical form."""
    a = a.strip()
    b = b.strip()
    if not a or not b:
        return False
    try:
        mol_a = Chem.MolFromSmiles(a)
        mol_b = Chem.MolFromSmiles(b)
        if mol_a is None or mol_b is None:
            return False
        return bool(Chem.MolToSmiles(mol_a) == Chem.MolToSmiles(mol_b))
    except Exception:
        return False


def inchikey_agree(a: str, b: str) -> bool:
    """Check if two InChIKeys share the same connectivity layer (first 14 chars)."""
    if not a or not b:
        return False
    if len(a) < 14 or len(b) < 14:
        return False
    return a[:14] == b[:14]


def text_token_contain(ocr_text: str, ai_text: str) -> bool:
    """Check if all OCR tokens appear in the AI-structured text.

    This validates that Claude Vision didn't miss any OCR-detected tokens.
    Empty OCR text is trivially True (no tokens to check).
    Trailing punctuation (commas, semicolons) is stripped before comparison.
    """
    ocr_tokens = ocr_text.strip().split()
    if not ocr_tokens:
        return True
    if not ai_text.strip():
        return False
    ai_lower = ai_text.lower()
    for token in ocr_tokens:
        clean = token.lower().rstrip(",.;:")
        if not clean:
            continue
        if clean not in ai_lower:
            return False
    return True
