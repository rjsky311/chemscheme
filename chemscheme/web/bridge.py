"""SMILES ↔ MOL block bridge for the Ketcher authoring canvas.

Ketcher accepts MOL/SMILES/CDXML/KET on input and emits the same on output.
For S68, we use MOL V2000 as the lingua franca:
- SMILES → MOL: RDKit parse + 2D coords → MOL block (for paste / load → canvas)
- MOL → SMILES: RDKit parse → canonical SMILES (for canvas → save)

CDXML is the natural format for Phase 7 round-trip; left as S69+ scope.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_mol_block(smiles: str) -> tuple[str, str]:
    """Parse SMILES → 2D-coordinate MOL block + canonical SMILES.

    Raises ValueError if SMILES cannot be parsed.
    """
    if not smiles:
        msg = "SMILES is empty"
        raise ValueError(msg)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        msg = f"could not parse SMILES: {smiles!r}"
        raise ValueError(msg)
    AllChem.Compute2DCoords(mol)  # type: ignore[attr-defined]
    mol_block = Chem.MolToMolBlock(mol)
    canonical = Chem.MolToSmiles(mol)
    return mol_block, canonical


def mol_block_to_smiles(mol_block: str) -> str:
    """Parse MOL block (e.g., from Ketcher canvas) → canonical SMILES.

    Raises ValueError if mol block cannot be parsed.
    """
    if not mol_block:
        msg = "mol_block is empty"
        raise ValueError(msg)
    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        msg = "could not parse MOL block"
        raise ValueError(msg)
    return str(Chem.MolToSmiles(mol))
