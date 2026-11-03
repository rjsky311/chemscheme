"""Layer 3 — Scheme continuity validation (graph-aware).

AC-6c.1: Linear step continuity (advisory).
AC-6c.2: Branch/join consumes_step_ids integrity.
AC-6c.3: Missing dependency info -> warning, not block.
AC-6c.4: Large NFA molecules get advisory-level Tanimoto.
"""

from rdkit import Chem  # type: ignore[import-untyped]
from rdkit.Chem import (
    DataStructs,  # type: ignore[import-untyped]
    rdFingerprintGenerator,  # type: ignore[import-untyped]
)

from chemscheme.models import ValidationResult

# Heavy atom count threshold for "large" molecules (NFA-scale)
_LARGE_MOL_THRESHOLD = 40
# Tanimoto similarity threshold for continuity check (default for medium molecules)
_TANIMOTO_THRESHOLD = 0.3
# Small molecule threshold (atom count below which higher Tanimoto is needed)
_SMALL_MOL_THRESHOLD = 10
# Morgan fingerprint generator (replaces deprecated GetMorganFingerprintAsBitVect)
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def validate_continuity(scheme: dict) -> ValidationResult:  # type: ignore[type-arg]
    """Validate scheme continuity across steps.

    Checks linear product->reactant flow and consumes_step_ids integrity.
    All continuity issues are advisory (warnings), not errors,
    except invalid consumes_step_ids references.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Thread-safe local cache — no module-level state (MR4-10)
    fp_cache: dict[str, object] = {}

    steps = scheme.get("steps", [])
    if len(steps) <= 1:
        return ValidationResult(layer="continuity", errors=[], warnings=[])

    step_ids = {step["id"] for step in steps if "id" in step}

    # Check consumes_step_ids references (AC-6c.2)
    _check_dependency_refs(steps, step_ids, errors)

    # Check for missing dependency info on merge steps (AC-6c.3)
    _check_missing_dependencies(steps, warnings)

    # Check linear continuity (AC-6c.1, AC-6c.4)
    _check_linear_continuity(steps, warnings, fp_cache)

    return ValidationResult(layer="continuity", errors=errors, warnings=warnings)


def _check_dependency_refs(
    steps: list[dict],  # type: ignore[type-arg]
    step_ids: set[str],
    errors: list[str],
) -> None:
    """AC-6c.2: Validate consumes_step_ids references exist."""
    for step in steps:
        for ref_id in step.get("consumes_step_ids", []):
            if ref_id not in step_ids:
                errors.append(
                    f"Step '{step.get('id', '?')}': consumes_step_ids "
                    f"references nonexistent step '{ref_id}'"
                )


def _check_missing_dependencies(
    steps: list[dict],  # type: ignore[type-arg]
    warnings: list[str],
) -> None:
    """AC-6c.3: Warn if merge steps lack consumes_step_ids.

    Only warns on the FIRST unlabeled step after branches have been seen
    (the likely merge point), not on subsequent post-merge linear steps.
    Once a merge point is identified (either via consumes_step_ids or by
    warning), all later steps are treated as post-merge linear.
    """
    branches_seen: set[str] = set()

    for step in steps:
        label = step.get("branch_label")
        if label:
            branches_seen.add(label)

    if len(branches_seen) < 2:
        return

    # Second pass: find the first potential merge point after branches
    all_branches_seen: set[str] = set()
    merge_found = False

    for step in steps:
        label = step.get("branch_label")
        if label:
            all_branches_seen.add(label)
            continue

        # Skip steps before any branches
        if not all_branches_seen:
            continue

        # If this step has consumes_step_ids, it's a proper merge — no warning
        if "consumes_step_ids" in step:
            merge_found = True
            continue

        # Only warn on the first potential merge point (unlabeled, no
        # consumes_step_ids, after branches). Once a merge is found or warned
        # about, subsequent steps are post-merge linear — no warning.
        if not merge_found:
            warnings.append(
                f"Step '{step.get('id', '?')}': merge step after branches "
                f"has no consumes_step_ids — dependency tracking incomplete"
            )
            merge_found = True


def _check_linear_continuity(
    steps: list[dict],  # type: ignore[type-arg]
    warnings: list[str],
    fp_cache: dict[str, object],
) -> None:
    """AC-6c.1 + AC-6c.4: Check product->reactant overlap between consecutive steps.

    Groups steps by branch_label so non-contiguous same-branch steps are
    still checked against each other (MR5-G4).
    """
    # Group steps by branch_label, preserving order within each group
    branch_groups: dict[str | None, list[dict]] = {}  # type: ignore[type-arg]
    for step in steps:
        label = step.get("branch_label")
        branch_groups.setdefault(label, []).append(step)

    # Check continuity within each branch group
    for _label, group in branch_groups.items():
        for i in range(len(group) - 1):
            current = group[i]
            next_step = group[i + 1]

            # Skip if next step has explicit consumes_step_ids
            if "consumes_step_ids" in next_step:
                continue

            products = current.get("products", [])
            reactants = next_step.get("reactants", [])

            if not products or not reactants:
                continue

            has_overlap = _check_molecule_overlap(products, reactants, fp_cache)
            if not has_overlap:
                is_large = _has_large_molecule(products + reactants)
                level = "advisory" if is_large else "warning"
                warnings.append(
                    f"Step '{current.get('id', '?')}' -> '{next_step.get('id', '?')}': "
                    f"no product-reactant continuity detected ({level})"
                )


def _adaptive_threshold(heavy_atom_count: int) -> float:
    """AC-13.3: Compute adaptive Tanimoto threshold based on molecule size.

    Small molecules (<10 atoms) need higher threshold because Morgan fingerprints
    produce sparse vectors, making unrelated small molecules appear similar.
    Large molecules (>40 atoms) keep the default threshold (advisory level).
    """
    if heavy_atom_count <= 0:
        return 0.5  # Very small/empty → strict threshold
    if heavy_atom_count < _SMALL_MOL_THRESHOLD:
        return 0.5  # Small molecules → higher threshold
    if heavy_atom_count >= _LARGE_MOL_THRESHOLD:
        return _TANIMOTO_THRESHOLD  # Large molecules → default
    return _TANIMOTO_THRESHOLD  # Medium molecules → default


def _get_fingerprint(smi: str, cache: dict[str, object]) -> object | None:
    """Get or compute Morgan fingerprint with caching (MR3-7)."""
    if smi in cache:
        return cache[smi]  # type: ignore[return-value]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        cache[smi] = None  # type: ignore[assignment]
        return None
    fp: object = _MORGAN_GEN.GetFingerprint(mol)
    cache[smi] = fp
    return fp


def _check_molecule_overlap(
    products: list[str], reactants: list[str], fp_cache: dict[str, object]
) -> bool:
    """Check if any product is similar to any reactant.

    AC-13.3: Uses adaptive threshold based on molecule size.
    """
    for prod_smi in products:
        prod_fp = _get_fingerprint(prod_smi, fp_cache)
        if prod_fp is None:
            continue
        prod_mol = Chem.MolFromSmiles(prod_smi)

        for react_smi in reactants:
            react_fp = _get_fingerprint(react_smi, fp_cache)
            if react_fp is None:
                continue
            react_mol = Chem.MolFromSmiles(react_smi)

            # Use the smaller molecule's size for threshold (stricter)
            min_atoms = min(
                prod_mol.GetNumHeavyAtoms() if prod_mol else 0,
                react_mol.GetNumHeavyAtoms() if react_mol else 0,
            )
            threshold = _adaptive_threshold(min_atoms)

            similarity = DataStructs.TanimotoSimilarity(prod_fp, react_fp)  # type: ignore[call-overload]
            if similarity >= threshold:
                return True

    return False


def _has_large_molecule(smiles_list: list[str]) -> bool:
    """Check if any molecule in the list is large (NFA-scale)."""
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.GetNumHeavyAtoms() >= _LARGE_MOL_THRESHOLD:
            return True
    return False
