"""Scheme editor: add/remove steps with validation.

Phase 7 — AC-7.1, AC-7.2, AC-7.3, AC-7.7.
All operations return new dicts (no mutation of input).
"""

from __future__ import annotations

import copy

from chemscheme.schema.scheme_schema import REQUIRED_STEP_FIELDS, validate_scheme_json


def add_step(
    scheme: dict,  # type: ignore[type-arg]
    step: dict,  # type: ignore[type-arg]
    *,
    position: int | None = None,
    validate: bool = False,
) -> dict | tuple[dict, dict]:  # type: ignore[type-arg]
    """Add a step to the scheme.

    Args:
        scheme: Source scheme dict (not mutated).
        step: New step dict with required fields.
        position: Insert position (0-indexed). None = append.
        validate: If True, return (scheme, validation_result) tuple.

    Returns:
        New scheme dict, or (scheme, validation) tuple if validate=True.

    Raises:
        ValueError: If step is missing required fields or has duplicate id.
    """
    # Guard: step must be a dict (MR4.1-P1d)
    if not isinstance(step, dict):
        raise ValueError(f"Step must be a dict, got {type(step).__name__}")

    # Validate required fields
    missing = REQUIRED_STEP_FIELDS - set(step.keys())
    if missing:
        raise ValueError(f"Step missing required fields: {sorted(missing)}")

    # Check duplicate id
    existing_ids = {s["id"] for s in scheme.get("steps", [])}
    if step["id"] in existing_ids:
        raise ValueError(f"Step id '{step['id']}' is a duplicate of an existing step")

    # Deep copy to avoid mutation
    result = copy.deepcopy(scheme)
    result.setdefault("steps", [])  # MR5-G2: support adding first step

    if position is not None:
        num_steps = len(result["steps"])
        if position < 0 or position > num_steps:
            raise ValueError(f"Position {position} out of range (valid: 0..{num_steps})")
        result["steps"].insert(position, copy.deepcopy(step))
    else:
        result["steps"].append(copy.deepcopy(step))

    if validate:
        validation = validate_scheme_json(result)
        return result, validation

    return result


def remove_step(
    scheme: dict,  # type: ignore[type-arg]
    step_id: str,
    *,
    validate: bool = False,
) -> dict | tuple[dict, dict]:  # type: ignore[type-arg]
    """Remove a step from the scheme by id.

    Args:
        scheme: Source scheme dict (not mutated).
        step_id: ID of the step to remove.
        validate: If True, return (scheme, validation_result) tuple.

    Returns:
        New scheme dict, or (scheme, validation) tuple if validate=True.

    Raises:
        ValueError: If step_id not found or would leave scheme empty.
    """
    steps = scheme.get("steps", [])

    # Check step exists
    if not any(s["id"] == step_id for s in steps):
        raise ValueError(f"Step '{step_id}' not found in scheme")

    # Cannot remove last step
    if len(steps) == 1:
        raise ValueError("Cannot remove the last step; scheme must have at least one step")

    # Deep copy to avoid mutation
    result = copy.deepcopy(scheme)
    result["steps"] = [s for s in result["steps"] if s["id"] != step_id]

    # Clean dangling consumes_step_ids references (AC-7.7)
    for step in result["steps"]:
        if "consumes_step_ids" in step:
            step["consumes_step_ids"] = [ref for ref in step["consumes_step_ids"] if ref != step_id]
            # Keep empty list to preserve explicit "no predecessors" semantics (MR5-G1)
            # Deleting the key would cause fallback to implicit graph routing

    if validate:
        validation = validate_scheme_json(result)
        return result, validation

    return result
