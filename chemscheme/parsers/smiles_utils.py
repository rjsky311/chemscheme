"""SMILES utility functions.

All functions return SmilesResult — no bare tuples allowed.
"""

import re

from rdkit import Chem  # type: ignore[import-untyped]

from chemscheme.models import SmilesResult


def canonicalize(smiles: str) -> SmilesResult:
    """Canonicalize a SMILES string using RDKit.

    Returns SmilesResult with confidence=0.0 for invalid input.
    """
    if not smiles:
        return SmilesResult(
            smiles="",
            confidence=0.0,
            source="rdkit",
            warnings=["invalid SMILES: empty input"],
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return SmilesResult(
            smiles="",
            confidence=0.0,
            source="rdkit",
            warnings=[f"invalid SMILES: {smiles}"],
        )

    canonical = Chem.MolToSmiles(mol)

    # Detect stereo-loss during canonicalization
    warnings: list[str] = []
    confidence = 1.0

    # Check 1: Compare textual stereo markers in original vs canonical.
    # This catches cases where RDKit strips stereo during the initial parse
    # (e.g. "C/C=C(/C)C" → "CC=C(C)C" — stereo is already lost at parse time).
    stereo_pattern = re.compile(r"[/\\@]")
    input_has_stereo = bool(stereo_pattern.search(smiles))
    output_has_stereo = bool(stereo_pattern.search(canonical))

    if input_has_stereo and not output_has_stereo:
        warnings.append(f"stereochemistry lost during canonicalization: {smiles} -> {canonical}")
        confidence = 0.9
    else:
        # Check 2: Compare RDKit stereo bond/chiral counts for subtler losses
        input_stereo_bonds = sum(
            1 for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE
        )
        input_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

        canon_mol = Chem.MolFromSmiles(canonical)
        if canon_mol is not None:
            output_stereo_bonds = sum(
                1 for b in canon_mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE
            )
            output_chiral_centers = len(
                Chem.FindMolChiralCenters(canon_mol, includeUnassigned=True)
            )

            if (
                input_stereo_bonds > output_stereo_bonds
                or input_chiral_centers > output_chiral_centers
            ):
                warnings.append(
                    f"stereochemistry lost during canonicalization: {smiles} -> {canonical}"
                )
                confidence = 0.9

    return SmilesResult(
        smiles=canonical,
        confidence=confidence,
        source="rdkit",
        warnings=warnings,
    )
