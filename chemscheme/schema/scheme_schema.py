"""scheme.json validation and I/O.

Defines the JSON Schema for scheme.json and provides validate/load/save functions.
"""

import json
import tempfile
from pathlib import Path

REQUIRED_STEP_FIELDS = {"id", "reactants", "products", "arrow_above"}


def validate_scheme(scheme: dict) -> list[str]:  # type: ignore[type-arg]
    """Validate a scheme dict against the schema.

    Returns a list of error strings (empty = valid).
    Checks both field presence and field types.
    """
    errors: list[str] = []

    # Guard: top-level must be a dict (MR4.1-C4.1-P2b)
    if not isinstance(scheme, dict):
        errors.append(f"Scheme must be a JSON object, got {type(scheme).__name__}")
        return errors

    # schema_version required and must be a string (MR4.1-P2e)
    if "schema_version" not in scheme:
        errors.append("'schema_version' is required at top level")
    elif not isinstance(scheme["schema_version"], str):
        errors.append("'schema_version' must be a string")

    # Validate known top-level metadata fields (MR5-C7)
    for meta_field in ("title", "style", "layout"):
        if meta_field in scheme and not isinstance(scheme[meta_field], str):
            errors.append(f"'{meta_field}' must be a string")

    # steps required and non-empty
    if "steps" not in scheme:
        errors.append("'steps' is required at top level")
        return errors

    steps = scheme["steps"]
    if not isinstance(steps, list) or len(steps) == 0:
        errors.append("'steps' must be a non-empty list")
        return errors

    # Collect all step IDs and check for duplicates
    step_ids: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "id" in step:
            sid = step["id"]
            if not isinstance(sid, str):
                # Non-string IDs are caught in the per-step type check below;
                # skip set operations to avoid TypeError on unhashable types.
                continue
            if sid in step_ids:
                errors.append(f"Duplicate step id: '{sid}'")
            step_ids.add(sid)

    # Validate each step — maintain running set of preceding IDs (MR4.1-P2a: O(N))
    preceding_ids: set[str] = set()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"Step {i}: must be a dict/object, got {type(step).__name__}")
            continue

        for field in REQUIRED_STEP_FIELDS:
            if field not in step:
                errors.append(f"Step {i}: '{field}' is required")

        # Type checks for required fields
        if "id" in step and not isinstance(step["id"], str):
            errors.append(f"Step {i}: 'id' must be a string")
        if "reactants" in step and not isinstance(step["reactants"], list):
            errors.append(f"Step {i}: 'reactants' must be a list")
        elif "reactants" in step:
            if len(step["reactants"]) == 0:
                errors.append(f"Step {i}: 'reactants' must not be empty")  # MR5-C5
            for j, r in enumerate(step["reactants"]):
                if not isinstance(r, str):
                    errors.append(f"Step {i}: reactants[{j}] must be a string")
        if "products" in step and not isinstance(step["products"], list):
            errors.append(f"Step {i}: 'products' must be a list")
        elif "products" in step:
            # MR8-C3: Empty products allowed (CDX imports have no reaction semantics)
            pass  # products=[] is valid for draft/partial schemes
            for j, p in enumerate(step["products"]):
                if not isinstance(p, str):
                    errors.append(f"Step {i}: products[{j}] must be a string")
        # AC-11.3: arrow_above accepts str or list[str] for multiline conditions (MR11-C1)
        if "arrow_above" in step and not isinstance(step["arrow_above"], (str, list)):
            errors.append(f"Step {i}: 'arrow_above' must be a string or list of strings")
        if "arrow_above" in step and isinstance(step["arrow_above"], list):
            for j, item in enumerate(step["arrow_above"]):
                if not isinstance(item, str):
                    errors.append(f"Step {i}: arrow_above[{j}] must be a string")
        # G7 (S68): arrow_below mirrors arrow_above — accepts str OR list[str]
        # Real OPV papers commonly have multi-line below text (solvent / temp / time
        # split across lines) symmetric with arrow_above multi-line conditions.
        if "arrow_below" in step and not isinstance(step["arrow_below"], (str, list)):
            errors.append(f"Step {i}: 'arrow_below' must be a string or list of strings")
        if "arrow_below" in step and isinstance(step["arrow_below"], list):
            for j, item in enumerate(step["arrow_below"]):
                if not isinstance(item, str):
                    errors.append(f"Step {i}: arrow_below[{j}] must be a string")

        # Validate consumes_step_ids references and type (MR3-14)
        if "consumes_step_ids" in step:
            if not isinstance(step["consumes_step_ids"], list):
                errors.append(f"Step {i}: 'consumes_step_ids' must be a list")
            else:
                sid = step.get("id", "")
                for j, ref_id in enumerate(step["consumes_step_ids"]):
                    if not isinstance(ref_id, str):
                        errors.append(f"Step {i}: consumes_step_ids[{j}] must be a string")
                    elif ref_id not in step_ids:
                        errors.append(
                            f"Step {i}: consumes_step_ids references nonexistent step '{ref_id}'"
                        )
                    elif ref_id == sid:
                        errors.append(f"Step {i}: consumes_step_ids references itself (cycle)")
                    elif ref_id not in preceding_ids:
                        errors.append(
                            f"Step {i}: consumes_step_ids references "
                            f"forward step '{ref_id}' (must reference earlier steps only)"
                        )

        # Validate optional fields consumed downstream (C2)
        # G7 (S68): arrow_below moved out of strict str-only check (now str | list)
        optional_str_fields = ("branch_label", "yield", "atmosphere")
        for field in optional_str_fields:
            if field in step and not isinstance(step[field], str):
                errors.append(f"Step {i}: '{field}' must be a string")

        # G14 (S68): arrow_style enum — "forward" (default) | "reversible" | "retro"
        if "arrow_style" in step:
            arrow_style = step["arrow_style"]
            valid_styles = {"forward", "reversible", "retro"}
            if not isinstance(arrow_style, str):
                errors.append(
                    f"Step {i}: 'arrow_style' must be a string (one of {sorted(valid_styles)})"
                )
            elif arrow_style not in valid_styles:
                errors.append(
                    f"Step {i}: 'arrow_style' value {arrow_style!r} not in {sorted(valid_styles)}"
                )

        optional_list_fields = ("reagents", "solvents")
        for field in optional_list_fields:
            if field in step:
                if not isinstance(step[field], list):
                    errors.append(f"Step {i}: '{field}' must be a list")
                else:
                    for j, item in enumerate(step[field]):
                        if not isinstance(item, str):
                            errors.append(f"Step {i}: {field}[{j}] must be a string")

        if "temperature" in step and not isinstance(step["temperature"], (str, int, float)):
            errors.append(f"Step {i}: 'temperature' must be a string or number")
        if "time" in step and not isinstance(step["time"], (str, int, float)):
            errors.append(f"Step {i}: 'time' must be a string or number")

        # AC-13.1: Validate polymer_info as dict[str, dict] (Phase 13)
        if "polymer_info" in step:
            pi = step["polymer_info"]
            if not isinstance(pi, dict):
                errors.append(f"Step {i}: 'polymer_info' must be a dict")
            else:
                for k, v in pi.items():
                    if not isinstance(k, str):
                        errors.append(f"Step {i}: polymer_info key must be a string")
                    if not isinstance(v, dict):
                        errors.append(f"Step {i}: polymer_info value must be a dict")

        # AC-13.2: Validate substituent_map as dict[str, list[str]] (Phase 13)
        if "substituent_map" in step:
            sm = step["substituent_map"]
            if not isinstance(sm, dict):
                errors.append(f"Step {i}: 'substituent_map' must be a dict")
            else:
                for k, v in sm.items():
                    if not isinstance(k, str):
                        errors.append(f"Step {i}: substituent_map key must be a string")
                    if not isinstance(v, list):
                        errors.append(f"Step {i}: substituent_map value must be a list")

        # C4: Validate compound_labels as dict[str, str] (Phase 11)
        if "compound_labels" in step:
            cl = step["compound_labels"]
            if not isinstance(cl, dict):
                errors.append(f"Step {i}: 'compound_labels' must be a dict")
            else:
                for k, v in cl.items():
                    if not isinstance(k, str):
                        errors.append(f"Step {i}: compound_labels key must be a string")
                    if not isinstance(v, str):
                        errors.append(f"Step {i}: compound_labels value must be a string")

        # Update preceding_ids for next iteration (MR4.1-P2a)
        sid = step.get("id")
        if isinstance(sid, str):
            preceding_ids.add(sid)

    return errors


def load_scheme(path: Path) -> dict:  # type: ignore[type-arg]
    """Load a scheme.json file.

    Raises ValueError with a user-friendly message on invalid JSON.
    """
    text = path.read_text(encoding="utf-8")
    try:
        result: dict = json.loads(text)  # type: ignore[type-arg]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    return result


def validate_scheme_json(scheme: dict) -> dict:  # type: ignore[type-arg]
    """Validate a scheme and return a JSON-compatible result dict.

    Runs all validation layers: schema, chemistry, reaction, continuity.
    Returns dict with keys: status, valid, issues, warnings, layers_checked.
    """
    from chemscheme.validators.layer1_chemistry import validate_chemistry
    from chemscheme.validators.layer2_reaction import validate_reaction
    from chemscheme.validators.layer3_scheme import validate_continuity

    errors = validate_scheme(scheme)
    all_warnings: list[str] = []
    layers = ["schema"]

    # Only run deeper layers if schema is valid
    if not errors:
        steps = scheme.get("steps", [])

        # Layer 1: Chemistry validation
        all_smiles: list[str] = []
        for step in steps:
            all_smiles.extend(step.get("reactants", []))
            all_smiles.extend(step.get("products", []))
        chem_result = validate_chemistry(all_smiles)
        errors.extend(chem_result.errors)
        all_warnings.extend(chem_result.warnings)
        layers.append("chemistry")

        # Layer 2: Reaction feasibility validation (MR4-9)
        for step in steps:
            # G1 fix: arrow_above can be a list (AC-11.3), normalize to string
            _arrow_above = step.get("arrow_above", "")
            if isinstance(_arrow_above, list):
                _arrow_above = ", ".join(_arrow_above)
            rxn_result = validate_reaction(
                reactants=step.get("reactants", []),
                products=step.get("products", []),
                reagents=_arrow_above,
                predictor=None,
            )
            errors.extend(rxn_result.errors)
            all_warnings.extend(rxn_result.warnings)
        layers.append("reaction")

        # Layer 3: Continuity validation
        cont_result = validate_continuity(scheme)
        errors.extend(cont_result.errors)
        all_warnings.extend(cont_result.warnings)
        layers.append("continuity")

    return {
        "status": "ok",
        "valid": len(errors) == 0,
        "issues": errors,
        "warnings": all_warnings,
        "layers_checked": layers,
    }


def save_scheme(scheme: dict, path: Path) -> None:
    """Save a scheme dict to a JSON file (atomic write — MR3-6)."""
    import time

    content = json.dumps(scheme, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
        # Retry replace on Windows PermissionError (MR4-15)
        for attempt in range(3):
            try:
                Path(tmp_path).replace(path)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.1)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
