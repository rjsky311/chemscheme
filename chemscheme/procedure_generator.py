"""Experimental section generator — scheme.json → procedure text.

Generates publication-quality experimental procedure paragraphs from
structured reaction scheme data. Uses compound_labels for naming and
domain_library for reaction type identification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Common atmosphere expansions
_ATMOSPHERE_NAMES: dict[str, str] = {
    "N2": "nitrogen",
    "Ar": "argon",
    "H2": "hydrogen",
    "O2": "oxygen",
    "air": "air",
    "vacuum": "vacuum",
}


@dataclass(frozen=True)
class StepProcedure:
    """Generated procedure text for a single reaction step."""

    step_number: int
    text: str
    reaction_type: str = ""


@dataclass
class ProcedureResult:
    """Complete procedure generation result."""

    text: str
    steps: list[StepProcedure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def generate_procedure(scheme: dict) -> ProcedureResult:  # type: ignore[type-arg]
    """Generate experimental procedure text from a scheme dict.

    Args:
        scheme: A valid scheme.json dict with steps.

    Returns:
        ProcedureResult with full text, per-step procedures, and warnings.

    Raises:
        ValueError: If scheme is invalid (no steps, missing required fields).
    """
    if not isinstance(scheme, dict):
        msg = "Scheme must be a dict"
        raise ValueError(msg)

    steps = scheme.get("steps")
    if steps is None:
        msg = "'steps' is required in scheme"
        raise ValueError(msg)

    if not isinstance(steps, list) or len(steps) == 0:
        msg = "Scheme must have at least one step"
        raise ValueError(msg)

    warnings: list[str] = []
    step_procedures: list[StepProcedure] = []

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            warnings.append(f"Step {i}: not a dict, skipped")
            continue

        # C2 fix: guard against missing/empty/non-list reactants
        reactants = step.get("reactants", [])
        if not isinstance(reactants, list) or len(reactants) == 0:
            warnings.append(f"Step {i + 1}: no reactants, skipped")
            continue

        products = step.get("products", [])
        if not isinstance(products, list) or len(products) == 0:
            warnings.append(f"Step {i + 1}: no products, skipped")
            continue

        sp = _generate_step(step, i + 1, warnings)
        step_procedures.append(sp)

    # Assemble full text
    paragraphs = []
    for sp in step_procedures:
        if len(step_procedures) > 1:
            paragraphs.append(f"Step {sp.step_number}: {sp.text}")
        else:
            paragraphs.append(sp.text)

    full_text = "\n\n".join(paragraphs)

    return ProcedureResult(
        text=full_text,
        steps=step_procedures,
        warnings=warnings,
    )


def _generate_step(
    step: dict,  # type: ignore[type-arg]
    step_number: int,
    warnings: list[str],
) -> StepProcedure:
    """Generate procedure text for a single step."""
    compound_labels = step.get("compound_labels", {})
    if not isinstance(compound_labels, dict):
        compound_labels = {}

    raw_reactants = step.get("reactants", [])
    products = step.get("products", [])

    has_labels = bool(compound_labels)
    if not has_labels:
        warnings.append(f"Step {step_number}: no compound_labels — using generic compound names")

    # G12 (S68): split reactants into actual structures vs reagent-name strings.
    # Heuristic: if an entry exact-matches a known abbreviation in
    # reagent_aliases.json with category reagent/solvent/catalyst/base/acid,
    # treat it as a reagent (e.g., "NBS" → reagent, not "the starting material").
    # Naive RDKit-parseable check is unreliable: "NBS" parses as a chemically
    # nonsensical 3-atom SMILES (N-B-S), but is a chemist's abbreviation.
    from chemscheme.domain_library import lookup_reagent

    reagent_like_categories = {"reagent", "solvent", "catalyst", "base", "acid"}

    def _is_reagent_name(entry: str) -> bool:
        """True if entry is a known abbreviation with a reagent-like category."""
        try:
            matches = lookup_reagent(entry)
        except Exception:
            return False
        # Exact-abbreviation match: the abbreviation field equals the query
        for r in matches:
            if r.abbreviation == entry and r.category in reagent_like_categories:
                return True
        return False

    reactants: list[str] = []
    extra_reagents: list[str] = []
    for entry in raw_reactants:
        if not isinstance(entry, str):
            continue
        if _is_reagent_name(entry):
            extra_reagents.append(entry)
        else:
            reactants.append(entry)

    # Name compounds (only real-structure reactants; reagent-strings handled below)
    reactant_names = _name_compounds(reactants, compound_labels, "starting material")
    product_names = _name_compounds(products, compound_labels, "product")

    # Get conditions
    arrow_above = step.get("arrow_above", "")
    conditions = _extract_conditions(step)
    # G12 (S68): merge non-SMILES reactant entries into reagent list so they
    # appear in "treated with X, Y" instead of "and the starting material".
    if extra_reagents:
        existing = conditions.get("reagents", [])
        if isinstance(existing, list):
            conditions["reagents"] = list(existing) + extra_reagents
        else:
            conditions["reagents"] = extra_reagents

    # Detect reaction type via domain library
    reaction_type = _identify_reaction_type(arrow_above, conditions)

    # Check if conditions are meaningful
    has_conditions = bool(
        conditions.get("reagents")
        or conditions.get("catalyst")
        or conditions.get("solvents")
        or conditions.get("temperature")
    )
    if not has_conditions:
        warnings.append(f"Step {step_number}: no reaction conditions specified")

    # Build sentence
    text = _format_procedure_sentence(
        reactant_names=reactant_names,
        product_names=product_names,
        conditions=conditions,
        reaction_type=reaction_type,
        yield_str=step.get("yield"),
        atmosphere=step.get("atmosphere"),
        has_conditions=has_conditions,
    )

    return StepProcedure(
        step_number=step_number,
        text=text,
        reaction_type=reaction_type,
    )


def _name_compounds(
    smiles_list: list[str],
    compound_labels: dict[str, str],
    fallback_role: str,
) -> list[str]:
    """Convert SMILES to compound names using labels or fallback."""
    names: list[str] = []
    for smi in smiles_list:
        label = compound_labels.get(smi)
        if label:
            names.append(f"Compound {label}")
        else:
            names.append(f"the {fallback_role}")
    return names


def _extract_conditions(step: dict) -> dict[str, str | list[str]]:  # type: ignore[type-arg]
    """Extract reaction conditions from step fields."""
    conditions: dict[str, str | list[str]] = {}

    # Collect reagent/condition text from both arrow_above and reagents field (C3 fix)
    reagent_parts: list[str] = []
    arrow_above = step.get("arrow_above", "")
    if isinstance(arrow_above, list):
        reagent_parts.extend(item for item in arrow_above if item)
    elif isinstance(arrow_above, str) and arrow_above.strip():
        reagent_parts.append(arrow_above)

    # C3 fix: also include schema-supported "reagents" field
    step_reagents = step.get("reagents", [])
    if isinstance(step_reagents, list):
        for r in step_reagents:
            if isinstance(r, str) and r.strip() and r not in reagent_parts:
                reagent_parts.append(r)

    if reagent_parts:
        conditions["reagents"] = reagent_parts

    solvents = step.get("solvents", [])
    if isinstance(solvents, list) and solvents:
        conditions["solvents"] = solvents

    temperature = step.get("temperature")
    if temperature:
        conditions["temperature"] = str(temperature)

    time_ = step.get("time")
    if time_:
        conditions["time"] = str(time_)

    return conditions


def _identify_reaction_type(
    arrow_above: str | list[str],
    conditions: dict[str, str | list[str]],
) -> str:
    """Try to identify the reaction type using domain_library public API (C6 fix)."""
    try:
        from chemscheme.domain_library import identify_reaction_type as _identify

        # Collect all condition text
        all_text_parts: list[str] = []
        if isinstance(arrow_above, list):
            all_text_parts.extend(arrow_above)
        elif isinstance(arrow_above, str):
            all_text_parts.append(arrow_above)

        # Also include reagents from conditions (C3 synergy)
        cond_reagents = conditions.get("reagents", [])
        if isinstance(cond_reagents, list):
            all_text_parts.extend(str(r) for r in cond_reagents if r not in all_text_parts)

        combined = " ".join(all_text_parts)
        return _identify(combined).name

    except ImportError:
        logger.debug("domain_library not available for reaction type matching")

    return ""


def _format_procedure_sentence(
    *,
    reactant_names: list[str],
    product_names: list[str],
    conditions: dict[str, str | list[str]],
    reaction_type: str,
    yield_str: str | None,
    atmosphere: str | None,
    has_conditions: bool,
) -> str:
    """Format a procedure sentence from extracted information."""
    parts: list[str] = []

    # Subject: reactants (C2 fix: guard empty)
    if len(reactant_names) == 0:
        subject = "the starting material"
        is_plural = False
    elif len(reactant_names) == 1:
        subject = reactant_names[0]
        is_plural = False
    else:
        subject = ", ".join(reactant_names[:-1]) + " and " + reactant_names[-1]
        is_plural = True

    # G12 (S68): subject-verb agreement — "Compound 1 and Compound 2 were treated"
    verb_treated = "were treated" if is_plural else "was treated"
    verb_reacted = "were reacted" if is_plural else "was reacted"
    verb_converted = "were converted" if is_plural else "was converted"

    # Verb phrase
    if has_conditions:
        # Reagents/catalyst
        reagent_list = conditions.get("reagents", [])
        if isinstance(reagent_list, list) and reagent_list:
            reagent_text = ", ".join(reagent_list)
            parts.append(f"{subject} {verb_treated} with {reagent_text}")
        else:
            parts.append(f"{subject} {verb_reacted}")
    else:
        parts.append(f"{subject} {verb_converted} to {', '.join(product_names)}")

    # Via reaction type
    if reaction_type:
        parts.append(f"via {reaction_type}")

    # In solvent
    solvents = conditions.get("solvents")
    if isinstance(solvents, list) and solvents:
        parts.append(f"in {'/'.join(solvents)}")

    # At temperature
    temp = conditions.get("temperature")
    if isinstance(temp, str) and temp:
        parts.append(f"at {temp}")

    # For time
    time_ = conditions.get("time")
    if isinstance(time_, str) and time_:
        parts.append(f"for {time_}")

    # Under atmosphere
    if atmosphere:
        atm_name = _ATMOSPHERE_NAMES.get(atmosphere, atmosphere)
        parts.append(f"under {atm_name} atmosphere")

    sentence = " ".join(parts)

    # Product clause (only if has_conditions, otherwise already mentioned)
    if has_conditions:
        product_text = ", ".join(product_names)
        sentence += f" to afford {product_text}"

    # Yield
    if yield_str:
        sentence += f" in {yield_str} yield"

    sentence += "."

    return sentence
