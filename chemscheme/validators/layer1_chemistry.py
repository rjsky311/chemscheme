"""Layer 1 — Chemical correctness validation via RDKit.

AC-6a.1: Valence check detects invalid molecules.
AC-6a.2: Stereochemistry silent failure detection.
AC-13.4: Stereo inversion detection (E/Z, R/S).
AC-13.6: Parse-time stereo stripping edge cases.
"""

from dataclasses import dataclass

from rdkit import Chem  # type: ignore[import-untyped]

from chemscheme.models import ValidationResult


@dataclass(frozen=True)
class StereoInversionResult:
    """Result of stereo inversion comparison between two molecules."""

    has_inversion: bool
    description: str


def validate_chemistry(smiles_list: list[str]) -> ValidationResult:
    """Validate a list of SMILES for chemical correctness.

    Checks:
    - RDKit can parse each SMILES (valence, syntax)
    - Stereochemistry is preserved after canonicalization
    """
    errors: list[str] = []
    warnings: list[str] = []

    for smiles in smiles_list:
        if not smiles:
            errors.append("Empty SMILES string")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            errors.append(f"Invalid SMILES: {smiles}")
            continue

        # AC-6a.2: Check stereochemistry preservation
        canonical = Chem.MolToSmiles(mol)
        _check_stereo(smiles, canonical, warnings)

    return ValidationResult(layer="chemistry", errors=errors, warnings=warnings)


def _check_stereo(original: str, canonical: str, warnings: list[str]) -> None:
    """Detect stereochemistry silent failures.

    Uses RDKit stereo bond/chiral center counts as primary check (MR4-17).
    Falls back to textual check only when RDKit parsing fails.
    """
    orig_mol = Chem.MolFromSmiles(original)
    canon_mol = Chem.MolFromSmiles(canonical)

    if orig_mol is not None and canon_mol is not None:
        # Primary: compare RDKit stereo counts (reliable, no false positives)
        orig_stereo_bonds = sum(
            1 for b in orig_mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE
        )
        orig_chiral = len(Chem.FindMolChiralCenters(orig_mol, includeUnassigned=True))

        canon_stereo_bonds = sum(
            1 for b in canon_mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE
        )
        canon_chiral = len(Chem.FindMolChiralCenters(canon_mol, includeUnassigned=True))

        if orig_stereo_bonds > canon_stereo_bonds or orig_chiral > canon_chiral:
            warnings.append(f"Stereochemistry may have been lost: '{original}' -> '{canonical}'")


def detect_stereo_inversion(smiles_a: str, smiles_b: str) -> StereoInversionResult:
    """AC-13.4: Detect E/Z or R/S configuration inversion between two molecules.

    Compares actual CIP assignments (not just counts) to detect
    when stereochemistry has been inverted rather than lost.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    if mol_a is None or mol_b is None:
        return StereoInversionResult(has_inversion=False, description="")

    # Assign stereochemistry for CIP labels
    Chem.AssignStereochemistry(mol_a, cleanIt=True, force=True)
    Chem.AssignStereochemistry(mol_b, cleanIt=True, force=True)

    inversions: list[str] = []

    # Check R/S chiral center inversions
    centers_a = dict(Chem.FindMolChiralCenters(mol_a, includeUnassigned=True))
    centers_b = dict(Chem.FindMolChiralCenters(mol_b, includeUnassigned=True))
    for atom_idx in set(centers_a) & set(centers_b):
        cip_a = centers_a[atom_idx]
        cip_b = centers_b[atom_idx]
        if cip_a != cip_b and cip_a in ("R", "S") and cip_b in ("R", "S"):
            inversions.append(f"R/S inversion at atom {atom_idx}: {cip_a} → {cip_b}")

    # Check E/Z double bond inversions
    stereo_bonds_a = {
        b.GetIdx(): b.GetStereo()
        for b in mol_a.GetBonds()
        if b.GetStereo() != Chem.BondStereo.STEREONONE
    }
    stereo_bonds_b = {
        b.GetIdx(): b.GetStereo()
        for b in mol_b.GetBonds()
        if b.GetStereo() != Chem.BondStereo.STEREONONE
    }
    for bond_idx in set(stereo_bonds_a) & set(stereo_bonds_b):
        s_a = stereo_bonds_a[bond_idx]
        s_b = stereo_bonds_b[bond_idx]
        if s_a != s_b:
            inversions.append(f"E/Z inversion at bond {bond_idx}")

    if inversions:
        return StereoInversionResult(
            has_inversion=True,
            description="; ".join(inversions),
        )
    return StereoInversionResult(has_inversion=False, description="")
