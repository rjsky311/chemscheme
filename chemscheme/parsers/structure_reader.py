"""Read chemical structures from MOL, SDF, and CDX files.

Returns SmilesResult for unified type system integration.
"""

from __future__ import annotations

import struct
from pathlib import Path

from rdkit import Chem

from chemscheme.models import SmilesResult

# Supported extensions
_SUPPORTED_EXTENSIONS = {".mol", ".sdf", ".cdx"}


def read_structure(path: Path | str) -> list[SmilesResult]:
    """Dispatch to the correct reader based on file extension.

    Args:
        path: Path to a MOL, SDF, or CDX file.

    Returns:
        List of SmilesResult (one per molecule found).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If extension is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".mol":
        return [read_mol(path)]
    elif suffix == ".sdf":
        return read_sdf(path)
    elif suffix == ".cdx":
        return read_cdx(path)
    else:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: {supported}")


def read_mol(path: Path | str) -> SmilesResult:
    """Read a MOL file (V2000/V3000) and return canonical SMILES.

    Args:
        path: Path to MOL file.

    Returns:
        SmilesResult with canonical SMILES or empty on failure.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=True)
    if mol is None:
        return SmilesResult(
            smiles="",
            confidence=0.0,
            source="mol_file",
            warnings=[f"Failed to parse MOL file: {path.name}"],
        )

    smiles = Chem.MolToSmiles(mol)
    return SmilesResult(smiles=smiles, confidence=1.0, source="mol_file", warnings=[])


def read_sdf(path: Path | str) -> list[SmilesResult]:
    """Read an SDF file and return SMILES for each molecule.

    Args:
        path: Path to SDF file.

    Returns:
        List of SmilesResult, one per molecule. Empty list if file is empty.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Check for empty file
    if path.stat().st_size == 0:
        return []

    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
    results: list[SmilesResult] = []

    for i, mol in enumerate(supplier):
        if mol is None:
            results.append(
                SmilesResult(
                    smiles="",
                    confidence=0.0,
                    source="sdf_file",
                    warnings=[f"Failed to parse molecule {i} in SDF"],
                )
            )
        else:
            smiles = Chem.MolToSmiles(mol)
            results.append(
                SmilesResult(smiles=smiles, confidence=1.0, source="sdf_file", warnings=[])
            )

    return results


def read_cdx(path: Path | str) -> list[SmilesResult]:
    """Read a ChemDraw CDX binary file and extract molecular structures.

    Parses the CDX binary tagged-object format to extract atom/bond
    connectivity, then builds RDKit molecules for SMILES generation.

    Args:
        path: Path to CDX binary file.

    Returns:
        List of SmilesResult, one per fragment (molecule) found.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file is not a valid CDX binary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # MR15-G1: File size guard (50 MB max, consistent with mathpix_client)
    max_cdx_size = 50 * 1024 * 1024
    file_size = path.stat().st_size
    if file_size > max_cdx_size:
        raise ValueError(
            f"CDX file too large ({file_size / 1024 / 1024:.1f} MB). "
            f"Maximum supported size: {max_cdx_size / 1024 / 1024:.0f} MB."
        )

    data = path.read_bytes()

    # Validate CDX signature
    if len(data) < 22 or data[:8] != b"VjCD0100":
        raise ValueError(f"Invalid CDX file: missing VjCD0100 signature in {path.name}")

    fragments = _parse_cdx_fragments(data)
    results: list[SmilesResult] = []

    for frag in fragments:
        smiles = _fragment_to_smiles(frag)
        if smiles:
            results.append(
                SmilesResult(smiles=smiles, confidence=1.0, source="cdx_file", warnings=[])
            )
        else:
            results.append(
                SmilesResult(
                    smiles="",
                    confidence=0.0,
                    source="cdx_file",
                    warnings=[f"Failed to reconstruct molecule from CDX fragment {frag['id']}"],
                )
            )

    return results


# --- CDX binary parser internals ---

# CDX property tags
_TAG_NODE_ELEMENT = 0x0402
_TAG_BOND_BEGIN = 0x0604
_TAG_BOND_END = 0x0605
_TAG_BOND_ORDER = 0x0600

# CDX object tags
_TAG_FRAGMENT = 0x8003
_TAG_NODE = 0x8004
_TAG_BOND = 0x8005

_CDX_HEADER_SIZE = 22


def _parse_cdx_fragments(data: bytes) -> list[dict]:  # type: ignore[type-arg]
    """Parse CDX binary data and extract fragment structures.

    Each fragment contains nodes (atoms) and bonds.
    """
    pos = _CDX_HEADER_SIZE
    obj_stack: list[tuple[str, int]] = []

    fragments: list[dict] = []  # type: ignore[type-arg]
    current_fragment: dict | None = None  # type: ignore[type-arg]
    current_node: dict | None = None  # type: ignore[type-arg]
    current_bond: dict | None = None  # type: ignore[type-arg]

    while pos < len(data) - 1:
        tag = struct.unpack_from("<H", data, pos)[0]
        pos += 2

        if tag == 0x0000:
            # End of object
            if obj_stack:
                closing_type, _ = obj_stack.pop()
                in_fragment = current_fragment is not None
                if closing_type == "Node" and current_node is not None and in_fragment:
                    assert current_fragment is not None  # for mypy
                    current_fragment["nodes"].append(current_node)
                    current_node = None
                elif closing_type == "Bond" and current_bond is not None and in_fragment:
                    assert current_fragment is not None  # for mypy
                    current_fragment["bonds"].append(current_bond)
                    current_bond = None
                elif closing_type == "Fragment" and current_fragment is not None:
                    fragments.append(current_fragment)
                    current_fragment = None
            continue

        is_object = bool(tag & 0x8000)

        if is_object:
            # Object: read 4-byte ID
            if pos + 4 > len(data):
                break
            obj_id = struct.unpack_from("<I", data, pos)[0]
            pos += 4

            if tag == _TAG_FRAGMENT:
                current_fragment = {"id": obj_id, "nodes": [], "bonds": []}
                obj_stack.append(("Fragment", obj_id))
            elif tag == _TAG_NODE:
                current_node = {"id": obj_id, "element": 6}  # default: Carbon
                obj_stack.append(("Node", obj_id))
            elif tag == _TAG_BOND:
                current_bond = {"id": obj_id, "begin": 0, "end": 0, "order": 1}
                obj_stack.append(("Bond", obj_id))
            else:
                obj_stack.append(("Other", obj_id))
        else:
            # Property: read 2-byte length (or 4-byte if 0xFFFF)
            if pos + 2 > len(data):
                break
            length = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            if length == 0xFFFF:
                if pos + 4 > len(data):
                    break
                length = struct.unpack_from("<I", data, pos)[0]
                pos += 4

            if pos + length > len(data):
                break
            prop_data = data[pos : pos + length]
            pos += length

            # Decode node properties
            if current_node is not None and tag == _TAG_NODE_ELEMENT and length == 2:
                current_node["element"] = struct.unpack_from("<H", prop_data)[0]

            # Decode bond properties
            if current_bond is not None:
                if tag == _TAG_BOND_BEGIN and length == 4:
                    current_bond["begin"] = struct.unpack_from("<I", prop_data)[0]
                elif tag == _TAG_BOND_END and length == 4:
                    current_bond["end"] = struct.unpack_from("<I", prop_data)[0]
                elif tag == _TAG_BOND_ORDER and length == 2:
                    current_bond["order"] = struct.unpack_from("<H", prop_data)[0]

    return fragments


def _fragment_to_smiles(frag: dict) -> str | None:  # type: ignore[type-arg]
    """Convert a parsed CDX fragment (nodes + bonds) to canonical SMILES."""
    nodes = frag.get("nodes", [])
    bonds = frag.get("bonds", [])

    if not nodes:
        return None

    rw_mol = Chem.RWMol()
    node_id_to_idx: dict[int, int] = {}

    for node in nodes:
        element = node.get("element", 6)
        # Guard against invalid atomic numbers
        if element < 1 or element > 118:
            element = 6
        idx = rw_mol.AddAtom(Chem.Atom(element))
        node_id_to_idx[node["id"]] = idx

    for bond in bonds:
        begin_idx = node_id_to_idx.get(bond["begin"])
        end_idx = node_id_to_idx.get(bond["end"])
        if begin_idx is None or end_idx is None or begin_idx == end_idx:
            continue

        order = bond.get("order", 1)
        bond_type = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }.get(order, Chem.BondType.SINGLE)

        rw_mol.AddBond(begin_idx, end_idx, bond_type)

    try:
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles: str = Chem.MolToSmiles(mol)
        return smiles
    except Exception:
        return None
