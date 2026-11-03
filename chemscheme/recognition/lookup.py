"""Structure lookup database using RDKit Morgan fingerprints.

Provides fast local lookup for known structures (OPV/NFA materials).
Uses native RDKit Tanimoto similarity — no FPSim2 dependency required.
FPSim2 HDF5 backend can be added as optional high-performance path.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from chemscheme.models import AgreementLevel, RecognitionResult

logger = logging.getLogger(__name__)

# Morgan fingerprint parameters (matches FPSim2 defaults)
_RADIUS = 2
_NBITS = 2048


def _canonical(smiles: str) -> str | None:
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return str(Chem.MolToSmiles(mol))


_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=_RADIUS, fpSize=_NBITS)


def _morgan_fp(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    """Compute Morgan fingerprint for a molecule."""
    return _FP_GEN.GetFingerprint(mol)  # type: ignore[no-any-return]


def build_lookup_db(
    smiles_list: list[str],
    output_path: Path,
    names: list[str] | None = None,
) -> int:
    """Build a lookup database from a list of SMILES.

    Args:
        smiles_list: List of SMILES strings.
        output_path: Path to write the JSON database.
        names: Optional parallel list of names for each SMILES.

    Returns:
        Number of valid entries written (invalid SMILES are skipped).
    """
    entries: dict[str, str] = {}  # canonical SMILES → name
    for i, smi in enumerate(smiles_list):
        canon = _canonical(smi)
        if canon is None:
            logger.warning("Skipping invalid SMILES: %s", smi)
            continue
        if canon in entries:
            continue  # deduplicate
        name = names[i] if names and i < len(names) else ""
        entries[canon] = name

    db = {
        "version": 1,
        "fingerprint": {"radius": _RADIUS, "nbits": _NBITS},
        "entries": [{"smiles": s, "name": n} for s, n in entries.items()],
    }
    output_path.write_text(json.dumps(db, indent=2), encoding="utf-8")
    return len(entries)


def load_user_library(path: Path) -> list[tuple[str, str]]:
    """Load a user-supplied structure library.

    Supports JSON (list of {smiles, name}) and CSV (smiles,name columns).

    Returns:
        List of (canonical_SMILES, name) tuples.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        msg = f"Library file not found: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()
    results: list[tuple[str, str]] = []

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data:
            smi = entry.get("smiles", "")
            name = entry.get("name", "")
            canon = _canonical(smi)
            if canon:
                results.append((canon, name))
    elif suffix == ".csv":
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = row.get("smiles", "")
                name = row.get("name", "")
                canon = _canonical(smi)
                if canon:
                    results.append((canon, name))
    else:
        msg = f"Unsupported library format: {suffix}"
        raise ValueError(msg)

    return results


class StructureLookup:
    """Local structure lookup using RDKit Morgan fingerprint similarity.

    Implements MoleculeRecognizer Protocol for cross-validation integration.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._entries: list[dict[str, str]] = []
        self._fingerprints: list[DataStructs.ExplicitBitVect] = []
        self._load()

    def _load(self) -> None:
        """Load database and precompute fingerprints."""
        data = json.loads(self._db_path.read_text(encoding="utf-8"))
        self._entries = data.get("entries", [])
        for entry in self._entries:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is not None:
                self._fingerprints.append(_morgan_fp(mol))
            else:
                # Should not happen if build_lookup_db was used
                self._fingerprints.append(None)  # type: ignore[arg-type]

    def lookup(self, smiles: str, threshold: float = 0.85) -> RecognitionResult:
        """Search for a structure by Tanimoto similarity.

        Args:
            smiles: Query SMILES string.
            threshold: Minimum Tanimoto similarity to consider a match.

        Returns:
            RecognitionResult with best match or zero confidence if no match.
        """
        mol = Chem.MolFromSmiles(smiles.strip() if smiles else "")
        if mol is None:
            return RecognitionResult(
                value="",
                confidence=0.0,
                source="structure_lookup",
                agreement=AgreementLevel.SINGLE,
                sources=("structure_lookup",),
            )

        query_fp = _morgan_fp(mol)
        best_sim = 0.0
        best_idx = -1

        for i, fp in enumerate(self._fingerprints):
            if fp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim < threshold or best_idx < 0:
            return RecognitionResult(
                value="",
                confidence=0.0,
                source="structure_lookup",
                agreement=AgreementLevel.SINGLE,
                sources=("structure_lookup",),
            )

        entry = self._entries[best_idx]
        # C4 fix: always return canonical SMILES as value (machine-readable)
        value = entry["smiles"]
        return RecognitionResult(
            value=value,
            confidence=round(best_sim, 4),
            source="structure_lookup",
            agreement=AgreementLevel.SINGLE,
            sources=("structure_lookup",),
        )

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        """MoleculeRecognizer Protocol: treat image bytes as SMILES string.

        In the full pipeline, image bytes would be passed through OCSR first.
        For direct lookup, pass SMILES encoded as bytes.
        """
        smiles = image.decode("utf-8", errors="replace").strip()
        return self.lookup(smiles)

    def info(self) -> dict[str, object]:
        """Return database metadata."""
        return {
            "entry_count": len(self._entries),
            "db_path": str(self._db_path),
        }
