"""CDXML export and import for ChemScheme.

Phase 7 — AC-7.4, AC-7.5, AC-7.6.
Uses xml.etree.ElementTree + RDKit for 2D coordinates.
No external CDXML library required.

CDXML structure:
  <CDXML>
    <page>
      <fragment id="F1">   ← one per molecule
        <n id="N1" p="x y" Element="6"/>  ← atoms
        <b id="B1" B="N1" E="N2"/>        ← bonds
      </fragment>
      <arrow id="A1" .../>                ← reaction arrows
      <t ...>conditions text</t>          ← arrow_above/below text
      <scheme>
        <step ReactionStepReactants="F1 F2"
              ReactionStepProducts="F3"
              ReactionStepArrows="A1"/>
      </scheme>
    </page>
  </CDXML>
"""

from __future__ import annotations

import contextlib
import logging
from xml.etree.ElementTree import Element, SubElement, indent, tostring

from defusedxml.ElementTree import fromstring  # hardened XML parser
from rdkit import Chem
from rdkit.Chem import AllChem

_logger = logging.getLogger(__name__)

# CDXML coordinate scaling: RDKit Angstroms → CDXML points (1 pt ≈ 1/72 inch)
_SCALE = 30.0  # scale factor for readable layout
_STEP_OFFSET_X = 250.0  # horizontal offset between reaction steps
_FRAGMENT_GAP = 80.0  # gap between reactant/product fragments within a step
_ARROW_LENGTH = 60.0  # arrow length in CDXML points
_BASE_X = 50.0  # starting X position
_BASE_Y = 200.0  # vertical center

# Element number → symbol (for import)
_ELEMENT_SYMBOLS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    26: "Fe",
    29: "Cu",
    30: "Zn",
    34: "Se",
    35: "Br",
    46: "Pd",
    50: "Sn",
    53: "I",
}

# Bond order mapping
_BOND_ORDER_MAP = {
    Chem.BondType.SINGLE: "1",
    Chem.BondType.DOUBLE: "2",
    Chem.BondType.TRIPLE: "3",
    Chem.BondType.AROMATIC: "1.5",
}

# AC-9.1: Bond direction → CDXML Display mapping (MR7-01, MR8-C1 fix)
_BOND_DIR_TO_DISPLAY = {
    Chem.BondDir.BEGINWEDGE: "WedgeBegin",
    Chem.BondDir.BEGINDASH: "WedgedHashBegin",
}


class _IdAllocator:
    """Simple monotonic ID allocator for CDXML elements."""

    def __init__(self, start: int = 1) -> None:
        self._next = start

    def next(self) -> int:
        val = self._next
        self._next += 1
        return val


def _mol_to_fragment(
    mol: Chem.Mol,  # type: ignore[name-defined]
    ids: _IdAllocator,
    offset_x: float,
    offset_y: float,
) -> tuple[Element, int, str]:
    """Convert an RDKit Mol to a CDXML <fragment> element.

    Returns (fragment_element, fragment_id, canonical_smiles).
    """
    AllChem.Compute2DCoords(mol)  # type: ignore[attr-defined]
    # AC-9.1: Assign wedge/hash bond directions for stereo centers (MR7-01)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    Chem.WedgeMolBonds(mol, mol.GetConformer())
    conf = mol.GetConformer()

    frag_id = ids.next()
    fragment = Element("fragment", id=str(frag_id))

    # Map atom index → CDXML node id
    atom_id_map: dict[int, int] = {}

    for atom in mol.GetAtoms():
        node_id = ids.next()
        atom_id_map[atom.GetIdx()] = node_id

        pos = conf.GetAtomPosition(atom.GetIdx())
        x = offset_x + pos.x * _SCALE
        y = offset_y + pos.y * _SCALE

        attrib: dict[str, str] = {
            "id": str(node_id),
            "p": f"{x:.6f} {y:.6f}",
        }

        atomic_num = atom.GetAtomicNum()
        if atomic_num != 6:  # Carbon is default, omit
            attrib["Element"] = str(atomic_num)

        num_h = atom.GetTotalNumHs()
        if num_h > 0 and atomic_num != 6:
            attrib["NumHydrogens"] = str(num_h)

        # AC-9.1: Formal charge preservation (MR7-01)
        charge = atom.GetFormalCharge()
        if charge != 0:
            attrib["Charge"] = str(charge)

        # AC-9.1: Isotope preservation (MR7-01)
        isotope = atom.GetIsotope()
        if isotope != 0:
            attrib["Isotope"] = str(isotope)

        SubElement(fragment, "n", attrib)

    for bond in mol.GetBonds():
        bond_id = ids.next()
        begin = atom_id_map[bond.GetBeginAtomIdx()]
        end = atom_id_map[bond.GetEndAtomIdx()]
        order = _BOND_ORDER_MAP.get(bond.GetBondType(), "1")

        attrib = {
            "id": str(bond_id),
            "B": str(begin),
            "E": str(end),
        }
        if order != "1":
            attrib["Order"] = order

        # AC-9.1: Stereo bond display (wedge/hash)
        bond_dir = bond.GetBondDir()
        display = _BOND_DIR_TO_DISPLAY.get(bond_dir)
        if display:
            attrib["Display"] = display

        SubElement(fragment, "b", attrib)

    smiles = Chem.MolToSmiles(mol)
    return fragment, frag_id, smiles


def _make_arrow(
    ids: _IdAllocator,
    tail_x: float,
    tail_y: float,
    head_x: float,
    head_y: float,
) -> tuple[Element, int]:
    """Create a CDXML <arrow> element."""
    arrow_id = ids.next()
    arrow = Element(
        "arrow",
        id=str(arrow_id),
        FillType="None",
        ArrowheadType="Solid",
        HeadSize="2250",
        ArrowheadHead="Full",
        Head3D=f"{head_x:.6f} {tail_y:.6f} 0.000000",
        Tail3D=f"{tail_x:.6f} {tail_y:.6f} 0.000000",
    )
    return arrow, arrow_id


def _make_text(
    ids: _IdAllocator,
    x: float,
    y: float,
    text: str,
) -> Element:
    """Create a CDXML <t> text element."""
    text_id = ids.next()
    t = Element("t", id=str(text_id), p=f"{x:.6f} {y:.6f}")
    s = SubElement(t, "s", font="3", size="10", face="96")
    s.text = text
    return t


def scheme_to_cdxml(scheme: dict) -> str:  # type: ignore[type-arg]
    """Export a scheme dict to CDXML XML string.

    Args:
        scheme: Valid scheme dict with steps.

    Returns:
        CDXML XML string.
    """
    ids = _IdAllocator(start=1)

    root = Element("CDXML")
    # Store metadata as CDXML attributes for round-trip
    if "schema_version" in scheme:
        root.set("ChemSchemeVersion", scheme["schema_version"])
    if "title" in scheme:
        root.set("ChemSchemeTitle", scheme.get("title", ""))

    page = SubElement(root, "page")

    cursor_x = _BASE_X
    all_step_data: list[dict[str, str | list[str]]] = []

    for step_idx, step in enumerate(scheme["steps"]):
        step_reactant_ids: list[str] = []
        step_product_ids: list[str] = []

        # --- Reactant fragments ---
        for smi in step.get("reactants", []):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(
                    f"Invalid SMILES in step '{step.get('id', step_idx)}' reactants: {smi}"
                )
            fragment, frag_id, _canon = _mol_to_fragment(mol, ids, cursor_x, _BASE_Y)
            # Store original SMILES for round-trip
            fragment.set("ChemSchemeSMILES", smi)
            page.append(fragment)
            step_reactant_ids.append(str(frag_id))
            cursor_x += _FRAGMENT_GAP

        # --- Arrow ---
        arrow_tail_x = cursor_x + 10
        arrow_head_x = arrow_tail_x + _ARROW_LENGTH
        arrow, arrow_id = _make_arrow(ids, arrow_tail_x, _BASE_Y, arrow_head_x, _BASE_Y)
        page.append(arrow)

        # --- arrow_above text (MR12-C1: normalize list to string) ---
        arrow_above = step.get("arrow_above", "")
        if isinstance(arrow_above, list):
            arrow_above = "\n".join(arrow_above)
        if arrow_above:
            t = _make_text(ids, arrow_tail_x, _BASE_Y - 20, arrow_above)
            page.append(t)

        # --- arrow_below text (G7 S68: normalize list to string, mirrors arrow_above) ---
        arrow_below = step.get("arrow_below", "")
        if isinstance(arrow_below, list):
            arrow_below = "\n".join(arrow_below)
        if arrow_below:
            t = _make_text(ids, arrow_tail_x, _BASE_Y + 20, arrow_below)
            page.append(t)

        cursor_x = arrow_head_x + 20

        # --- Product fragments ---
        for smi in step.get("products", []):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(
                    f"Invalid SMILES in step '{step.get('id', step_idx)}' products: {smi}"
                )
            fragment, frag_id, _canon = _mol_to_fragment(mol, ids, cursor_x, _BASE_Y)
            fragment.set("ChemSchemeSMILES", smi)
            page.append(fragment)
            step_product_ids.append(str(frag_id))
            cursor_x += _FRAGMENT_GAP

        cursor_x += _STEP_OFFSET_X - _FRAGMENT_GAP  # gap to next step

        # Collect step data for <scheme> element
        step_data: dict[str, str | list[str]] = {
            "reactant_ids": step_reactant_ids,
            "product_ids": step_product_ids,
            "arrow_id": str(arrow_id),
            "step_id": step.get("id", str(step_idx + 1)),
            "arrow_above": arrow_above,
            "arrow_below": arrow_below,
        }
        if "yield" in step:
            step_data["yield"] = step["yield"]
        # Convergent semantics (C4)
        if "branch_label" in step:
            step_data["branch_label"] = step["branch_label"]
        if "consumes_step_ids" in step:
            step_data["consumes_step_ids"] = step["consumes_step_ids"]
        # Optional metadata for round-trip fidelity (MR5-C2)
        for opt_key in ("reagents", "solvents", "temperature", "time", "atmosphere"):
            if opt_key in step:
                step_data[opt_key] = step[opt_key]
        # MR10-C5/MR11-C8: compound_labels round-trip
        if step.get("compound_labels"):
            step_data["compound_labels"] = step["compound_labels"]
        all_step_data.append(step_data)

    # --- <scheme> element ---
    scheme_elem = SubElement(page, "scheme")
    for sd in all_step_data:
        step_attrib: dict[str, str] = {
            "ReactionStepReactants": " ".join(sd["reactant_ids"]),  # type: ignore[arg-type]
            "ReactionStepProducts": " ".join(sd["product_ids"]),  # type: ignore[arg-type]
            "ReactionStepArrows": str(sd["arrow_id"]),
            "ChemSchemeStepId": str(sd["step_id"]),
        }
        if sd.get("arrow_above"):
            step_attrib["ChemSchemeArrowAbove"] = str(sd["arrow_above"])
        if sd.get("arrow_below"):
            step_attrib["ChemSchemeArrowBelow"] = str(sd["arrow_below"])
        if sd.get("yield"):
            step_attrib["ChemSchemeYield"] = str(sd["yield"])
        # Optional metadata for round-trip (MR5-C2)
        import json as _json

        for opt_key in ("reagents", "solvents", "compound_labels"):
            if sd.get(opt_key):
                step_attrib[f"ChemScheme_{opt_key}"] = _json.dumps(sd[opt_key])
        for opt_key in ("temperature", "time", "atmosphere"):
            # Use `is not None` to preserve falsey values like 0 (MR5R2-C4)
            if sd.get(opt_key) is not None:
                step_attrib[f"ChemScheme_{opt_key}"] = str(sd[opt_key])
        # Convergent semantics (C4)
        if sd.get("branch_label"):
            step_attrib["ChemSchemeBranchLabel"] = str(sd["branch_label"])
        # Use `in` to preserve empty list [] for explicit no-predecessors (MR5R2-C4)
        if "consumes_step_ids" in sd:
            cids = sd["consumes_step_ids"]
            if isinstance(cids, list):
                step_attrib["ChemSchemeConsumesStepIds"] = ",".join(cids)  # MR5-G5: comma separator
        SubElement(scheme_elem, "step", step_attrib)

    # Serialize
    indent(root, space="  ")
    return tostring(root, encoding="unicode", xml_declaration=True)


def _collect_arrow_positions(
    page: Element,
) -> dict[str, tuple[float, float, float, float]]:
    """Map arrow_id → (tail_x, tail_y, head_x, head_y).

    Processes both <arrow> elements (Head3D/Tail3D) and <graphic GraphicType="Line">
    elements (BoundingBox), since ChemDraw uses both representations.
    """
    arrows: dict[str, tuple[float, float, float, float]] = {}
    # Standard <arrow> elements
    for arrow in page.iter("arrow"):
        aid = arrow.get("id", "")
        head3d = arrow.get("Head3D", "")
        tail3d = arrow.get("Tail3D", "")
        if not (aid and head3d and tail3d):
            continue
        try:
            hp = head3d.split()
            tp = tail3d.split()
            arrows[aid] = (float(tp[0]), float(tp[1]), float(hp[0]), float(hp[1]))
        except (ValueError, IndexError):
            continue
    # ChemDraw <graphic> line arrows (referenced by step ReactionStepArrows)
    for graphic in page.iter("graphic"):
        gid = graphic.get("id", "")
        if not gid or graphic.get("GraphicType") != "Line":
            continue
        bbox = graphic.get("BoundingBox", "")
        if not bbox:
            continue
        try:
            parts = bbox.split()
            # BoundingBox: "head_x head_y tail_x tail_y"
            hx, hy, tx, ty = (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
            arrows[gid] = (tx, ty, hx, hy)
        except (ValueError, IndexError):
            continue
    return arrows


def _collect_page_texts(page: Element) -> list[tuple[float, float, str]]:
    """Collect standalone <t> text elements (direct children of page) with positions.

    Concatenates <s> inline runs (ChemDraw uses separate <s> for subscript/
    superscript styling). Preserves explicit newlines within text content.
    """
    texts: list[tuple[float, float, str]] = []
    for t in page.findall("t"):
        p = t.get("p", "")
        if not p:
            continue
        try:
            parts = p.split()
            x, y = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        # Concatenate <s> runs inline (they represent font-style changes, not lines)
        content = "".join(s.text or "" for s in t.findall("s")).strip()
        if content:
            texts.append((x, y, content))
    return texts


def _find_arrow_text(
    arrow_pos: tuple[float, float, float, float],
    texts: list[tuple[float, float, str]],
    x_margin: float = 30.0,
    y_threshold: float = 60.0,
) -> tuple[str, str]:
    """Find text elements above and below an arrow by proximity.

    Returns (arrow_above, arrow_below).
    """
    tx, ty, hx, hy = arrow_pos
    arrow_y = (ty + hy) / 2
    arrow_x_min = min(tx, hx) - x_margin
    arrow_x_max = max(tx, hx) + x_margin

    above: list[tuple[float, str]] = []
    below: list[tuple[float, str]] = []

    for text_x, text_y, content in texts:
        if text_x < arrow_x_min or text_x > arrow_x_max:
            continue
        dy = text_y - arrow_y
        if abs(dy) > y_threshold:
            continue
        if dy < 0:
            above.append((text_y, content))
        else:
            below.append((text_y, content))

    # Sort: closest to arrow first
    above.sort(key=lambda t: -t[0])
    below.sort(key=lambda t: t[0])

    return (
        "\n".join(t[1] for t in above),
        "\n".join(t[1] for t in below),
    )


def cdxml_to_scheme(cdxml_str: str) -> dict:  # type: ignore[type-arg]
    """Import a CDXML string to a scheme dict.

    Args:
        cdxml_str: CDXML XML string.

    Returns:
        Scheme dict with schema_version, steps, etc.
    """
    root = fromstring(cdxml_str)
    page = root.find("page")
    if page is None:
        raise ValueError("CDXML has no <page> element")

    # Recover metadata
    schema_version = root.get("ChemSchemeVersion", "1")
    title = root.get("ChemSchemeTitle")

    # Build fragment id → SMILES lookup
    import_warnings: list[str] = []
    frag_smiles: dict[str, str] = {}
    for fragment in page.findall("fragment"):
        frag_id = fragment.get("id", "")
        # Prefer stored SMILES for round-trip fidelity
        stored = fragment.get("ChemSchemeSMILES")
        if stored:
            # Validate stored SMILES via RDKit before trusting it (MR4-3)
            from rdkit import Chem as _Chem

            mol = _Chem.MolFromSmiles(stored)
            if mol is not None:
                frag_smiles[frag_id] = _Chem.MolToSmiles(mol)  # canonicalize
            else:
                # Stored SMILES invalid — fall back to reconstruction
                import_warnings.append(
                    f"Fragment {frag_id}: stored SMILES '{stored}' is invalid, "
                    f"attempting reconstruction"
                )
                smiles = _fragment_to_smiles(fragment)
                if smiles:
                    frag_smiles[frag_id] = smiles
                else:
                    import_warnings.append(f"Fragment {frag_id}: reconstruction also failed")
        else:
            # Reconstruct from nodes/bonds (fallback — lossy: no stereo/charge/isotope)
            smiles = _fragment_to_smiles(fragment)
            if smiles:
                frag_smiles[frag_id] = smiles
                import_warnings.append(
                    f"Fragment {frag_id}: reconstructed from CDXML nodes "
                    f"(complex stereo or implicit H counts may differ)"
                )
            else:
                import_warnings.append(
                    f"Fragment {frag_id}: could not reconstruct SMILES from CDXML"
                )

    # Parse ALL scheme/step elements (support multi-scheme CDXML)
    scheme_elems = page.findall("scheme")
    if not scheme_elems:
        raise ValueError("CDXML has no <scheme> element")

    # Build arrow position + page text lookups for condition extraction
    arrow_positions = _collect_arrow_positions(page)
    page_texts = _collect_page_texts(page)

    steps: list[dict[str, object]] = []
    for scheme_elem in scheme_elems:
        for step_elem in scheme_elem.findall("step"):
            reactant_ids = step_elem.get("ReactionStepReactants", "").split()
            product_ids = step_elem.get("ReactionStepProducts", "").split()

            # Fail fast on missing fragment references (MR3-1)
            missing_refs = [
                fid for fid in reactant_ids + product_ids if fid and fid not in frag_smiles
            ]
            if missing_refs:
                raise ValueError(
                    f"CDXML import failed: fragment IDs {missing_refs} referenced "
                    f"in step but not found in document"
                )

            reactants = [frag_smiles[rid] for rid in reactant_ids if rid]
            products = [frag_smiles[pid] for pid in product_ids if pid]

            step_id = step_elem.get("ChemSchemeStepId", str(len(steps) + 1))
            arrow_above = step_elem.get("ChemSchemeArrowAbove", "")
            arrow_below = step_elem.get("ChemSchemeArrowBelow")
            yield_val = step_elem.get("ChemSchemeYield")

            # Arrow text proximity extraction (for non-ChemScheme CDXML files)
            if not arrow_above and not arrow_below:
                arrow_ids = step_elem.get("ReactionStepArrows", "").split()
                for aid in arrow_ids:
                    if aid in arrow_positions:
                        prox_above, prox_below = _find_arrow_text(arrow_positions[aid], page_texts)
                        if prox_above:
                            arrow_above = prox_above
                        if prox_below:
                            arrow_below = prox_below
                        break

            step_dict: dict[str, object] = {
                "id": step_id,
                "reactants": reactants,
                "products": products,
                "arrow_above": arrow_above,
            }
            if arrow_below:
                step_dict["arrow_below"] = arrow_below
            if yield_val:
                step_dict["yield"] = yield_val
            # Optional metadata round-trip (MR5-C2)
            import json as _json

            for list_key in ("reagents", "solvents", "compound_labels"):
                raw = step_elem.get(f"ChemScheme_{list_key}")
                if raw:
                    with contextlib.suppress(_json.JSONDecodeError, ValueError):
                        step_dict[list_key] = _json.loads(raw)
            for str_key in ("temperature", "time", "atmosphere"):
                raw = step_elem.get(f"ChemScheme_{str_key}")
                if raw:
                    step_dict[str_key] = raw
            # Convergent semantics (C4)
            branch_label = step_elem.get("ChemSchemeBranchLabel")
            if branch_label:
                step_dict["branch_label"] = branch_label
            consumes_raw = step_elem.get("ChemSchemeConsumesStepIds")
            if consumes_raw is not None:  # preserve empty string → [] (C1 fix)
                # MR5-G5: comma separator; also handle legacy space-separated
                if "," in consumes_raw:
                    parts = [s.strip() for s in consumes_raw.split(",")]
                    step_dict["consumes_step_ids"] = [s for s in parts if s]
                else:
                    step_dict["consumes_step_ids"] = consumes_raw.split()

            steps.append(step_dict)

    result: dict[str, object] = {
        "schema_version": schema_version,
        "steps": steps,
    }
    if title:
        result["title"] = title
    if import_warnings:
        result["import_warnings"] = import_warnings
        for w in import_warnings:
            _logger.warning("CDXML import: %s", w)

    return result  # type: ignore[return-value]


def _fragment_to_smiles(fragment: Element) -> str | None:
    """Reconstruct SMILES from a CDXML fragment (fallback for non-ChemScheme files).

    Uses node Element attributes and bond connectivity to build an RDKit Mol.
    """
    nodes = fragment.findall("n")
    bonds = fragment.findall("b")
    if not nodes:
        return None

    rw_mol = Chem.RWMol()
    node_id_to_idx: dict[str, int] = {}

    for node in nodes:
        node_id = node.get("id", "")
        element_raw = node.get("Element", "6")
        try:
            element_num = int(element_raw)  # default Carbon
        except ValueError:
            # Some CDXML exporters use symbols instead of numbers (MR4.1-P2c)
            from rdkit.Chem import GetPeriodicTable

            pt = GetPeriodicTable()
            try:
                element_num = pt.GetAtomicNumber(element_raw)
            except RuntimeError:
                element_num = 6  # fallback to Carbon
        # Guard against out-of-bounds atomic numbers (MR5R2-G2)
        # Allow 0 for dummy atoms (R-groups, polymer attachments)
        if element_num < 0 or element_num > 118:
            element_num = 6  # fallback to Carbon
        atom = Chem.Atom(element_num)

        # AC-9.2: Read formal charge from CDXML (MR7-02)
        charge_raw = node.get("Charge")
        if charge_raw is not None:
            with contextlib.suppress(ValueError, TypeError):
                atom.SetFormalCharge(int(charge_raw))

        # AC-9.2: Read isotope from CDXML (MR7-02)
        isotope_raw = node.get("Isotope")
        if isotope_raw is not None:
            with contextlib.suppress(ValueError, TypeError, OverflowError):
                atom.SetIsotope(int(isotope_raw))

        idx = rw_mol.AddAtom(atom)
        node_id_to_idx[node_id] = idx

        num_h = node.get("NumHydrogens")
        if num_h is not None:
            with contextlib.suppress(ValueError, TypeError):
                rw_mol.GetAtomWithIdx(idx).SetNumExplicitHs(int(num_h))

    for bond in bonds:
        begin_id = bond.get("B", "")
        end_id = bond.get("E", "")
        if begin_id not in node_id_to_idx or end_id not in node_id_to_idx:
            continue
        # Skip self-referential bonds (MR5R2-G3)
        if begin_id == end_id:
            continue

        order_str = bond.get("Order", "1")
        if order_str == "2":
            bond_type = Chem.BondType.DOUBLE
        elif order_str == "3":
            bond_type = Chem.BondType.TRIPLE
        elif order_str == "1.5":
            bond_type = Chem.BondType.AROMATIC
        else:
            bond_type = Chem.BondType.SINGLE

        bond_idx = rw_mol.AddBond(node_id_to_idx[begin_id], node_id_to_idx[end_id], bond_type)

        # AC-9.2: Read stereo Display attribute (MR7-02, MR8-C2 fix)
        # WedgeBegin: wedge from B→E; WedgeEnd: wedge from E→B (reverse bond)
        display = bond.get("Display", "")
        if display == "WedgeBegin":
            rw_mol.GetBondWithIdx(bond_idx - 1).SetBondDir(Chem.BondDir.BEGINWEDGE)
        elif display == "WedgeEnd":
            rw_mol.GetBondWithIdx(bond_idx - 1).SetBondDir(Chem.BondDir.ENDDOWNRIGHT)
        elif display == "WedgedHashBegin":
            rw_mol.GetBondWithIdx(bond_idx - 1).SetBondDir(Chem.BondDir.BEGINDASH)
        elif display == "WedgedHashEnd":
            rw_mol.GetBondWithIdx(bond_idx - 1).SetBondDir(Chem.BondDir.ENDDOWNRIGHT)

    try:
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        # MR8-G3: Assign stereochemistry and use isomeric SMILES for charge/isotope/stereo
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        smiles: str = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smiles
    except Exception:
        return None
