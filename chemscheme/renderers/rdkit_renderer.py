"""RDKit-based single-step reaction renderer.

Replaces Indigo renderer with RDKit's MolDraw2D for better 2D layout quality,
especially for complex polycyclic molecules (spiro compounds, fused acceptor cores, etc.).

Renders: reactant1 + reactant2 → product1 + product2
with arrow_above/below text annotations.
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from rdkit import Chem
from rdkit.Chem import AllChem, rdAbbreviations
from rdkit.Chem.Draw import rdMolDraw2D

from chemscheme.renderers.themes import DEFAULT_THEME, SchemeTheme

# Layout constants (legacy aliases — used by _compose_reaction_svg PNG path)
_MOL_WIDTH = 250  # base width per molecule
_MOL_HEIGHT = DEFAULT_THEME.mol_height
_PLUS_WIDTH = DEFAULT_THEME.plus_width
_ARROW_WIDTH = 100  # arrow region for legacy SVG path (not drawsvg)
_PADDING = 10  # padding for legacy SVG path
_TEXT_HEIGHT = 20  # text height for legacy SVG path

# Chemistry abbreviation definitions (Ph, Bn, Bz, Ts, Ms)
_CUSTOM_ABBREVS_TEXT = """\
Ph *c1ccccc1 Ph Ph
Bn *Cc1ccccc1 Bn Bn
Bz *C(=O)c1ccccc1 Bz Bz
Ts *S(=O)(=O)c1ccc(C)cc1 Ts Ts
Ms *S(=O)(=O)C Ms Ms
"""
_CUSTOM_ABBREVS: list[rdAbbreviations.AbbreviationDefinition] | None = None
_DEFAULT_ABBREVS: list[rdAbbreviations.AbbreviationDefinition] | None = None
_MAX_COVERAGE = 0.95  # allow abbreviations covering up to 95% of atoms (consistency)


_AbbrevList = list[rdAbbreviations.AbbreviationDefinition]


def _get_abbreviations() -> tuple[_AbbrevList, _AbbrevList]:
    """Return (default_abbrevs, custom_abbrevs) tuple (cached).

    Two-pass condensation avoids RDKit conflict when overlapping matches exist.
    """
    global _CUSTOM_ABBREVS, _DEFAULT_ABBREVS  # noqa: PLW0603
    if _DEFAULT_ABBREVS is None:
        _DEFAULT_ABBREVS = list(rdAbbreviations.GetDefaultAbbreviations())
    if _CUSTOM_ABBREVS is None:
        _CUSTOM_ABBREVS = list(
            rdAbbreviations.ParseAbbreviations(_CUSTOM_ABBREVS_TEXT, removeExtraDummies=True)
        )
    return _DEFAULT_ABBREVS, _CUSTOM_ABBREVS


def _condense_abbreviations(mol: Chem.Mol) -> Chem.Mol:  # type: ignore[name-defined]
    """Apply abbreviation condensation in two passes: defaults first, then custom."""
    defaults, custom = _get_abbreviations()
    mol = rdAbbreviations.CondenseMolAbbreviations(mol, defaults, maxCoverage=_MAX_COVERAGE)
    mol = rdAbbreviations.CondenseMolAbbreviations(mol, custom, maxCoverage=_MAX_COVERAGE)
    return mol


def _validate_and_prepare(smiles_list: list[str], role: str) -> list[Chem.Mol]:  # type: ignore[name-defined]
    """Validate SMILES and return RDKit Mol objects with 2D coords and condensed abbreviations."""
    mols = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Invalid SMILES in {role}: {s}")
        AllChem.Compute2DCoords(mol)  # type: ignore[attr-defined]
        mol = _condense_abbreviations(mol)
        mols.append(mol)
    return mols


def _has_substructure(mol: Chem.Mol, smarts: str) -> bool:  # type: ignore[name-defined]
    """Check if molecule contains a substructure (pre-condensation check)."""
    pat = Chem.MolFromSmarts(smarts)
    return pat is not None and mol.HasSubstructMatch(pat)


def _validate_and_prepare_consistent(
    all_smiles: list[str],
    n_reactants: int,
) -> tuple[list[Chem.Mol], list[Chem.Mol]]:  # type: ignore[name-defined]
    """Validate and prepare ALL molecules in a step with consistent abbreviation policy.

    Step-level consistency: if any molecule containing a particular functional group
    (e.g. phenyl) cannot abbreviate it (due to maxCoverage), then NO molecule in the
    step abbreviates that group. This prevents visual inconsistency like "Br—Ph" next
    to a fully-drawn biphenyl.

    Args:
        all_smiles: All SMILES in the step (reactants + products).
        n_reactants: Number of reactant SMILES at the start of the list.

    Returns:
        (reactant_mols, product_mols) with consistent abbreviations.
    """
    # Phase 1: Parse and compute 2D coords (no abbreviation yet)
    raw_mols: list[Chem.Mol] = []  # type: ignore[name-defined]
    for s in all_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            role = "reactants" if len(raw_mols) < n_reactants else "products"
            raise ValueError(f"Invalid SMILES in {role}: {s}")
        AllChem.Compute2DCoords(mol)  # type: ignore[attr-defined]
        raw_mols.append(mol)

    # Phase 2: Try condensing each molecule independently
    condensed_mols = [_condense_abbreviations(Chem.RWMol(m)) for m in raw_mols]

    # Phase 3: Check consistency — for each custom abbreviation group,
    # if any molecule that CONTAINS the substructure did NOT get it condensed,
    # revert ALL molecules to uncondensed form for that group type.
    #
    # C2/G3 fix: detect condensation per target group by running group-specific
    # condensation on copies, not by comparing total atom counts. On fallback,
    # apply only default abbreviations (not raw) to preserve COOH/tBu/etc.
    abbrev_smarts = {
        "Ph": "c1ccccc1",  # phenyl ring
        "Bn": "[CH2]c1ccccc1",  # benzyl
        "Bz": "C(=O)c1ccccc1",  # benzoyl
    }

    use_condensed = True
    defaults, custom = _get_abbreviations()
    for grp_name, smarts in abbrev_smarts.items():
        mols_with_group: list[int] = []
        condensed_flags: list[bool] = []
        # G-R2-1 fix: filter custom abbrevs to only the target group
        grp_abbrevs = [a for a in custom if a.label == grp_name]
        if not grp_abbrevs:
            continue
        for idx, raw in enumerate(raw_mols):
            if _has_substructure(raw, smarts):
                mols_with_group.append(idx)
                # Per-group check: try condensing ONLY this group's abbrev
                test_mol = rdAbbreviations.CondenseMolAbbreviations(
                    Chem.RWMol(raw), grp_abbrevs, maxCoverage=_MAX_COVERAGE
                )
                condensed_flags.append(test_mol.GetNumAtoms() < raw.GetNumAtoms())

        has_inconsistency = (
            mols_with_group
            and condensed_flags
            and any(condensed_flags)
            and not all(condensed_flags)
        )
        if has_inconsistency:
            # Inconsistency detected — fall back to defaults-only (not raw)
            use_condensed = False
            break

    if use_condensed:
        result = condensed_mols
    else:
        # G3 fix: apply only default abbreviations (preserves COOH, tBu, etc.)
        result = [
            rdAbbreviations.CondenseMolAbbreviations(
                Chem.RWMol(m), defaults, maxCoverage=_MAX_COVERAGE
            )
            for m in raw_mols
        ]

    return result[:n_reactants], result[n_reactants:]


def _mol_to_svg(
    mol: Chem.Mol,  # type: ignore[name-defined]
    width: int,
    height: int,
    theme: SchemeTheme | None = None,
) -> str:
    """Render a single molecule to SVG string (inner content only).

    G3-G5 fix (S67 Part 2): when ``theme.molecule_bw_palette`` is True,
    route through RDKit's black-white atom palette so heteroatoms
    (S/N/O/Br/Cl/P) render as black instead of RDKit defaults
    (olive/blue/red/brown). Journal themes (ACS/Nature/Wiley/RSC) opt in;
    default theme keeps colored for backward compatibility.

    G4 fix (S69 E1): when ``theme.molecule_atoms_as_text`` is True,
    dispatches to the custom in-house drawer that emits bonds as
    ``<line>`` and heteroatom labels as ``<text>`` so vector editors
    can treat atom symbols as editable text.
    """
    if theme is not None and theme.molecule_atoms_as_text:
        return _mol_to_svg_text_atoms(mol, width, height, theme)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False  # type: ignore[assignment]
    if theme is not None and theme.molecule_bw_palette:
        opts.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Strip outer <svg> wrapper, keep just the drawing elements
    svg = re.sub(r"<\?xml[^?]*\?>\s*", "", svg)
    match = re.search(r"<svg[^>]*>(.*)</svg>", svg, re.DOTALL)
    return match.group(1) if match else svg


def _mol_to_svg_text_atoms(
    mol: Chem.Mol,  # type: ignore[name-defined]
    width: int,
    height: int,
    theme: SchemeTheme,
) -> str:
    """Custom SVG drawer: bonds as <line>, heteroatoms as <text>, carbons implicit.

    G4 fix (S69 E1). This drawer sidesteps RDKit's ``MolDraw2DSVG`` and
    Indigo's renderer — both of which vectorize atom symbols into
    ``<path>`` glyph geometry — so that downstream vector editors can
    treat atom symbols as editable text.

    Design notes:
    - RDKit 2D coords are consumed as-is (caller is responsible for
      ``AllChem.Compute2DCoords`` or equivalent).
    - Aromatic bonds are Kekulized first so double-bond positions are
      deterministic.
    - Implicit H count is shown inline (``OH`` / ``NH2``), with the
      numeric suffix rendered as a ``<tspan>`` subscript.
    - Labels include abbreviation pseudo-atoms (Ph, Bn, Bz, Ts, Ms)
      since ``rdAbbreviations`` condenses them before this runs.
    - Bonds touching labeled atoms are shortened by a small margin so
      line ends do not overlap glyph text.
    - Carbon atoms remain implicit (no ``<text>`` emitted) per
      chemistry convention.

    Scope out of v1: stereochemistry wedges, charge labels, radical
    markers, and aromatic-circle variants. These fall back to plain
    single-line rendering; callers needing stereo fidelity should keep
    ``molecule_atoms_as_text=False`` until v2.
    """
    import contextlib
    import math

    # Ensure 2D coords exist (defensive; callers usually prepare first)
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)  # type: ignore[attr-defined]

    # Kekulize a working copy so aromatic bonds get concrete SINGLE/DOUBLE.
    # Some structures (e.g. unusual aromatics) may fail Kekulization; in that
    # case fall through and render as-is.
    work = Chem.RWMol(mol)
    with contextlib.suppress(ValueError, RuntimeError):
        Chem.Kekulize(work, clearAromaticFlags=True)

    n = work.GetNumAtoms()
    if n == 0:
        return ""

    conf = work.GetConformer()
    positions: list[tuple[float, float]] = []
    for i in range(n):
        p = conf.GetAtomPosition(i)
        positions.append((p.x, p.y))

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    content_w = max_x - min_x
    content_h = max_y - min_y

    pad = 15
    usable_w = max(width - 2 * pad, 1)
    usable_h = max(height - 2 * pad, 1)

    if content_w < 1e-6 and content_h < 1e-6:
        scale = 30.0
    elif content_w < 1e-6:
        scale = usable_h / content_h
    elif content_h < 1e-6:
        scale = usable_w / content_w
    else:
        scale = min(usable_w / content_w, usable_h / content_h)
    # Clamp scale so tiny molecules don't balloon and large ones stay readable.
    scale = max(min(scale, 40.0), 8.0)

    rendered_w = content_w * scale
    rendered_h = content_h * scale
    off_x = (width - rendered_w) / 2 - min_x * scale
    off_y = (height - rendered_h) / 2 - min_y * scale

    def svg_xy(idx: int) -> tuple[float, float]:
        px, py = positions[idx]
        # Flip Y (RDKit Y-up, SVG Y-down)
        return (px * scale + off_x, height - (py * scale + off_y))

    # Compute atom labels
    font_family = theme.font_family if theme is not None else "Arial"
    font_size = 12  # atom-label font size; small enough to read inline
    stroke_color = theme.color_text if theme is not None else "black"
    stroke_width = 1.5

    def atom_label(atom: Chem.Atom) -> str | None:  # type: ignore[name-defined]
        # Abbreviation pseudo-atoms (Ph, Bn, Bz, Ts, Ms) carry their display
        # symbol in the ``atomLabel`` property set by rdAbbreviations.
        for prop in ("atomLabel", "_displayLabel"):
            if atom.HasProp(prop):
                return str(atom.GetProp(prop))
        sym = str(atom.GetSymbol())
        if sym == "C":
            return None  # carbon implicit
        if sym == "*":
            # Unlabeled dummy atom (attachment point). Skip.
            return None
        return sym

    labels: dict[int, str] = {}
    h_counts: dict[int, int] = {}
    for i in range(n):
        atom = work.GetAtomWithIdx(i)
        lab = atom_label(atom)
        if lab is not None:
            labels[i] = lab
            # Suppress implicit-H display for abbreviation pseudo-atoms
            # (their label already encodes the group's hydrogens); only
            # real heteroatoms (atomic_num > 0 and not "C") get H counts.
            is_abbrev = atom.GetAtomicNum() == 0
            h_counts[i] = 0 if is_abbrev else atom.GetTotalNumHs()

    labeled = set(labels.keys())

    parts: list[str] = []

    # --- Bond rendering helpers ---

    def emit_line(x1: float, y1: float, x2: float, y2: float) -> None:
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke_color}" stroke-width="{stroke_width}" '
            f'stroke-linecap="round"/>'
        )

    def emit_parallel(x1: float, y1: float, x2: float, y2: float, offsets: list[float]) -> None:
        """Emit parallel lines at perpendicular offsets (px).

        offsets[0] is typically 0 (center line); secondary lines are
        shortened by ``margin`` on each end so double/triple bonds
        look like classical chemistry convention.
        """
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1.0
        nx = -dy / length
        ny = dx / length
        for off in offsets:
            ax = x1 + nx * off
            ay = y1 + ny * off
            bx = x2 + nx * off
            by = y2 + ny * off
            if off != 0:
                # shrink secondary lines by 18% each end for chemistry look
                margin = 0.18
                ax = ax + dx * margin
                ay = ay + dy * margin
                bx = bx - dx * margin
                by = by - dy * margin
            emit_line(ax, ay, bx, by)

    def shrink_ends(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        i1: int,
        i2: int,
    ) -> tuple[float, float, float, float]:
        """Shorten bond so its ends don't overlap atom labels."""
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1.0
        fx, fy = dx / length, dy / length

        def margin_for(idx: int) -> float:
            if idx not in labeled:
                return 0.0
            lab = labels[idx]
            # approximate label half-width in pixels
            return font_size * max(1, len(lab)) * 0.35

        m1 = margin_for(i1)
        m2 = margin_for(i2)
        new_x1 = x1 + fx * m1
        new_y1 = y1 + fy * m1
        new_x2 = x2 - fx * m2
        new_y2 = y2 - fy * m2
        return new_x1, new_y1, new_x2, new_y2

    # --- Draw bonds ---
    bond_offset = 3.0  # px perpendicular offset for double/triple bonds
    for bond in work.GetBonds():
        i1 = bond.GetBeginAtomIdx()
        i2 = bond.GetEndAtomIdx()
        x1, y1 = svg_xy(i1)
        x2, y2 = svg_xy(i2)
        x1, y1, x2, y2 = shrink_ends(x1, y1, x2, y2, i1, i2)
        bt = bond.GetBondType()
        if bt == Chem.BondType.DOUBLE:
            emit_parallel(x1, y1, x2, y2, [0.0, bond_offset])
        elif bt == Chem.BondType.TRIPLE:
            emit_parallel(x1, y1, x2, y2, [0.0, bond_offset, -bond_offset])
        else:
            # SINGLE, AROMATIC (post-Kekulize rare), or UNSPECIFIED
            emit_line(x1, y1, x2, y2)

    # --- Draw atom labels ---
    for idx in sorted(labeled):
        label = labels[idx]
        sx, sy = svg_xy(idx)
        nh = h_counts.get(idx, 0)
        # Render label with optional implicit-H display
        if nh == 0:
            content = xml_escape(label)
        elif nh == 1:
            content = f"{xml_escape(label)}H"
        else:
            # "H" + subscript count
            content = (
                f"{xml_escape(label)}H"
                f'<tspan dy="{theme.subscript_dy}" font-size="{theme.subscript_scale}">'
                f"{nh}</tspan>"
            )
        parts.append(
            f'<text x="{sx:.2f}" y="{sy:.2f}" text-anchor="middle" '
            f'dominant-baseline="central" '
            f'font-family="{font_family}" font-size="{font_size}" '
            f'fill="{stroke_color}">{content}</text>'
        )

    return "".join(parts)


def _compose_reaction_svg(
    reactant_mols: list[Chem.Mol],  # type: ignore[name-defined]
    product_mols: list[Chem.Mol],  # type: ignore[name-defined]
    arrow_above: str,
    arrow_below: str,
) -> str:
    """Compose a full reaction SVG from molecule lists."""
    # Calculate dimensions
    n_reactants = len(reactant_mols)
    n_products = len(product_mols)

    # Scale molecule box width based on atom count (uniform sizing for consistent bond lengths)
    def mol_width_single(n: int) -> int:
        if n <= 6:
            return 180
        elif n <= 15:
            return _MOL_WIDTH
        else:
            return int(_MOL_WIDTH * (n / 15) ** 0.5)

    all_mols = reactant_mols + product_mols
    all_atom_counts = [m.GetNumAtoms() for m in all_mols]
    if all_atom_counts:
        max_atoms = max(all_atom_counts)
        ref_width = mol_width_single(max_atoms)
        px_per_atom = ref_width / max_atoms if max_atoms > 0 else 15.0
        _min_w = 80
        all_widths = [max(_min_w, int(px_per_atom * n)) for n in all_atom_counts]
        reactant_widths = all_widths[:n_reactants]
        product_widths = all_widths[n_reactants:]
    else:
        reactant_widths = [mol_width_single(m.GetNumAtoms()) for m in reactant_mols]
        product_widths = [mol_width_single(m.GetNumAtoms()) for m in product_mols]

    reactants_total = sum(reactant_widths) + _PLUS_WIDTH * max(0, n_reactants - 1)
    products_total = sum(product_widths) + _PLUS_WIDTH * max(0, n_products - 1)
    total_w = _PADDING * 2 + reactants_total + _ARROW_WIDTH + products_total

    has_above = bool(arrow_above)
    has_below = bool(arrow_below)
    text_space = _TEXT_HEIGHT * (int(has_above) + int(has_below))
    total_h = _PADDING * 2 + _MOL_HEIGHT + text_space

    parts: list[str] = []

    # White background
    parts.append(f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="white"/>')

    cursor_x = _PADDING
    mol_y = _PADDING + (_TEXT_HEIGHT if has_above else 0)

    # --- Reactants ---
    for i, (mol, w) in enumerate(zip(reactant_mols, reactant_widths, strict=True)):
        if i > 0:
            # "+" sign
            plus_x = cursor_x + _PLUS_WIDTH / 2
            plus_y = mol_y + _MOL_HEIGHT / 2
            parts.append(
                f'<text x="{plus_x}" y="{plus_y + 5}" text-anchor="middle" '
                f'font-family="Arial" font-size="16" fill="black">+</text>'
            )
            cursor_x += _PLUS_WIDTH

        svg_inner = _mol_to_svg(mol, w, _MOL_HEIGHT)
        parts.append(f'<g transform="translate({cursor_x},{mol_y})">{svg_inner}</g>')
        cursor_x += w

    # --- Arrow ---
    arrow_x1 = cursor_x + 15
    arrow_x2 = cursor_x + _ARROW_WIDTH - 15
    arrow_y = mol_y + _MOL_HEIGHT / 2

    # Arrow line
    parts.append(
        f'<line x1="{arrow_x1}" y1="{arrow_y}" x2="{arrow_x2}" y2="{arrow_y}" '
        f'stroke="black" stroke-width="1.5" marker-end="url(#rxn-arrow)"/>'
    )

    # Arrow text above
    if arrow_above:
        text_x = (arrow_x1 + arrow_x2) / 2
        text_y = arrow_y - 10
        parts.append(
            f'<text x="{text_x}" y="{text_y}" text-anchor="middle" '
            f'font-family="Arial" font-size="11" fill="black">'
            f"{xml_escape(arrow_above)}</text>"
        )

    # Arrow text below
    if arrow_below:
        text_x = (arrow_x1 + arrow_x2) / 2
        text_y = arrow_y + 18
        parts.append(
            f'<text x="{text_x}" y="{text_y}" text-anchor="middle" '
            f'font-family="Arial" font-size="10" fill="black">'
            f"{xml_escape(arrow_below)}</text>"
        )

    cursor_x += _ARROW_WIDTH

    # --- Products ---
    for i, (mol, w) in enumerate(zip(product_mols, product_widths, strict=True)):
        if i > 0:
            plus_x = cursor_x + _PLUS_WIDTH / 2
            plus_y = mol_y + _MOL_HEIGHT / 2
            parts.append(
                f'<text x="{plus_x}" y="{plus_y + 5}" text-anchor="middle" '
                f'font-family="Arial" font-size="16" fill="black">+</text>'
            )
            cursor_x += _PLUS_WIDTH

        svg_inner = _mol_to_svg(mol, w, _MOL_HEIGHT)
        parts.append(f'<g transform="translate({cursor_x},{mol_y})">{svg_inner}</g>')
        cursor_x += w

    # Arrowhead definition
    defs = (
        "<defs>"
        '<marker id="rxn-arrow" markerWidth="10" markerHeight="7" '
        'refX="10" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="black"/>'
        "</marker>"
        "</defs>"
    )

    svg = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">'
        f"{defs}"
        f"{''.join(parts)}"
        f"</svg>"
    )
    return svg


def estimate_step_size(  # type: ignore[type-arg]
    step: dict,  # type: ignore[type-arg]
    theme: SchemeTheme | None = None,
) -> tuple[float, float]:
    """Estimate rendered dimensions for a step based on molecule complexity.

    Uses the same adaptive mol_width logic as _compose_reaction_svg to predict
    the final rendered width and height without actually rendering.

    Args:
        step: Step dict with reactants, products, arrow_above, optional arrow_below.
        theme: Optional theme for layout dimensions. Uses default if None.

    Returns:
        (width, height) tuple in pixels.
    """
    if theme is None:
        theme = DEFAULT_THEME

    t_mol_width = 250  # base mol width (not in theme, internal to sizing)
    t_mol_height = theme.mol_height
    t_plus_width = theme.plus_width
    t_arrow_width = theme.arrow_width
    t_padding = theme.padding
    t_text_height = theme.text_line_height

    reactant_smiles = step.get("reactants", [])
    product_smiles = step.get("products", [])

    def _atom_count(smiles: str) -> int:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumAtoms() if mol is not None else 10  # fallback for invalid

    def _mol_w(n_atoms: int) -> int:
        if n_atoms <= 6:
            return 180
        elif n_atoms <= 15:
            return t_mol_width
        else:
            return int(t_mol_width * (n_atoms / 15) ** 0.5)

    n_reactants = len(reactant_smiles)
    n_products = len(product_smiles)

    if n_reactants == 0 and n_products == 0:
        return (
            float(t_mol_width + t_arrow_width + t_padding * 2),
            float(t_mol_height + t_padding * 2),
        )

    all_atom_counts = [_atom_count(s) for s in reactant_smiles + product_smiles]
    if all_atom_counts:
        max_atoms = max(all_atom_counts)
        ref_width = _mol_w(max_atoms)
        px_per_atom = ref_width / max_atoms if max_atoms > 0 else 15.0
        _min_w = 80
        all_widths = [max(_min_w, int(px_per_atom * n)) for n in all_atom_counts]
        reactant_widths = all_widths[:n_reactants]
        product_widths = all_widths[n_reactants:]
    else:
        reactant_widths = [_mol_w(_atom_count(s)) for s in reactant_smiles]
        product_widths = [_mol_w(_atom_count(s)) for s in product_smiles]

    reactants_total = sum(reactant_widths) + t_plus_width * max(0, n_reactants - 1)
    products_total = sum(product_widths) + t_plus_width * max(0, n_products - 1)
    total_w = t_padding * 2 + reactants_total + t_arrow_width + products_total

    # MR13-C3: Match compose_step_svg's height calculation more closely
    # Count arrow-above lines (multi-line, structured conditions)
    from chemscheme.renderers.drawsvg_composer import _build_arrow_lines

    above_lines = _build_arrow_lines(step)
    has_above = bool(above_lines)
    has_below = bool(step.get("arrow_below", "")) or bool(step.get("yield", ""))
    above_text_space = t_text_height * max(1, len(above_lines)) if has_above else 0
    below_text_space = t_text_height if has_below else 0

    # Account for compound labels and substituent legend
    compound_labels = step.get("compound_labels", {}) or {}
    substituent_map = step.get("substituent_map", {}) or {}
    label_space = theme.label_height if compound_labels else 0
    legend_space = (t_text_height * (len(substituent_map) + 1)) if substituent_map else 0

    total_h = (
        t_padding * 2
        + t_mol_height
        + above_text_space
        + below_text_space
        + label_space
        + legend_space
    )

    return (float(total_w), float(total_h))


def render_step(  # noqa: D401
    step: dict,  # type: ignore[type-arg]
    fmt: str = "svg",
    dpi: int = 300,
    theme: SchemeTheme | None = None,
) -> bytes:
    """Render a single reaction step to SVG or PNG bytes.

    Drop-in replacement for indigo_renderer.render_step.

    Args:
        step: Step dict with reactants, products, arrow_above, optional arrow_below.
        fmt: Output format, "svg" or "png".
        dpi: Resolution for PNG output (default 300).
        theme: Optional journal theme. Uses default if None.

    Returns:
        Rendered image as bytes.

    Raises:
        ValueError: If format is unsupported or SMILES are invalid.
    """
    if fmt not in ("svg", "png"):
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'svg' or 'png'.")

    if fmt == "svg":
        # Delegate to drawsvg_composer for publication-quality SVG with chemistry subscripts
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        return compose_step_svg(step, theme=theme)

    # MR13-C2: PNG path uses themed SVG → PNG conversion for theme parity
    if theme is not None:
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        themed_svg = compose_step_svg(step, theme=theme)
        try:
            return _svg_to_png(themed_svg, dpi)
        except ImportError:
            pass  # Fall through to Pillow-based rendering below

    # Legacy/fallback PNG path: use consistent abbreviation policy
    all_smiles = step["reactants"] + step["products"]
    reactant_mols, product_mols = _validate_and_prepare_consistent(
        all_smiles, len(step["reactants"])
    )
    # C2: Normalize list-form arrow_above to string for PNG path
    arrow_above = step.get("arrow_above", "")
    if isinstance(arrow_above, list):
        arrow_above = ", ".join(arrow_above)
    arrow_below = step.get("arrow_below", "")
    return _render_png(reactant_mols, product_mols, arrow_above, arrow_below, dpi)


def _ensure_cairo_dll() -> None:
    """Pre-load Cairo DLL from RDKit's bundled libraries on Windows.

    cairocffi/CairoSVG can't find the DLL because RDKit mangles the filename.
    We pre-load it so cffi picks up the already-loaded library.
    """
    import platform

    if platform.system() != "Windows":
        return

    import ctypes
    import glob

    rdkit_libs = Path(__file__).parent.parent.parent / ".." / "rdkit.libs"
    if not rdkit_libs.exists():
        # Try site-packages
        try:
            import rdkit

            rdkit_libs = Path(rdkit.__file__).parent.parent / "rdkit.libs"
        except ImportError:
            return
    if not rdkit_libs.exists():
        return

    matches = glob.glob(str(rdkit_libs / "cairo-2*.dll"))
    if matches:
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(rdkit_libs.resolve()))
            ctypes.CDLL(matches[0])
        except OSError:
            pass


def _svg_to_png(svg_bytes: bytes, dpi: int) -> bytes:
    """Convert SVG bytes to PNG using CairoSVG.

    MR13-C2: Used by themed PNG rendering to ensure theme parity.

    Raises:
        ImportError: If CairoSVG is not installed (caller should fall back).
    """
    _ensure_cairo_dll()
    import cairosvg  # type: ignore[import-untyped,import-not-found]

    return bytes(cairosvg.svg2png(bytestring=svg_bytes, dpi=dpi))


def _render_png(
    reactant_mols: list[Chem.Mol],  # type: ignore[name-defined]
    product_mols: list[Chem.Mol],  # type: ignore[name-defined]
    arrow_above: str,
    arrow_below: str,
    dpi: int,
) -> bytes:
    """Render reaction to PNG using Cairo or SVG→PNG fallback."""
    # Strategy: render SVG first, then convert
    svg_str = _compose_reaction_svg(reactant_mols, product_mols, arrow_above, arrow_below)
    svg_bytes = svg_str.encode("utf-8")

    # Try CairoSVG first
    try:
        import cairosvg  # type: ignore[import-untyped,import-not-found]

        return bytes(cairosvg.svg2png(bytestring=svg_bytes, dpi=dpi))
    except ImportError:
        pass

    # Fallback: render each molecule via MolDraw2DCairo and compose with Pillow
    try:
        from PIL import Image, ImageDraw  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "PNG rendering requires CairoSVG or Pillow. "
            "Install with: pip install cairosvg  OR  pip install Pillow"
        ) from None

    scale = dpi / 300.0
    mol_h = int(200 * scale)
    arrow_w = int(80 * scale)
    plus_w = int(30 * scale)
    pad = int(10 * scale)

    all_mols = reactant_mols + product_mols
    n_r = len(reactant_mols)

    # C4/G5 fix: use dynamic per-molecule widths matching the SVG path
    def _png_mol_w(n_atoms: int) -> int:
        if n_atoms <= 6:
            return int(180 * scale)
        elif n_atoms <= 15:
            return int(250 * scale)
        else:
            return int(250 * (n_atoms / 15) ** 0.5 * scale)

    all_atom_counts = [m.GetNumAtoms() for m in all_mols]
    if all_atom_counts:
        max_atoms = max(all_atom_counts)
        ref_w = _png_mol_w(max_atoms)
        px_per_atom = ref_w / max_atoms if max_atoms > 0 else int(15 * scale)
        _min_w = int(80 * scale)
        mol_widths = [max(_min_w, int(px_per_atom * n)) for n in all_atom_counts]
    else:
        mol_widths = [_png_mol_w(m.GetNumAtoms()) for m in all_mols]

    reactant_ws = mol_widths[:n_r]
    product_ws = mol_widths[n_r:]

    total_w = (
        pad * 2
        + sum(mol_widths)
        + max(0, n_r - 1) * plus_w
        + arrow_w
        + max(0, len(product_mols) - 1) * plus_w
    )
    # Reserve space for text annotations (C2 fix)
    text_h = int(20 * scale)
    has_above = bool(arrow_above)
    has_below = bool(arrow_below)
    text_space = text_h * (int(has_above) + int(has_below))
    total_h = pad * 2 + mol_h + text_space
    mol_y = pad + (text_h if has_above else 0)

    canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    cursor_x = pad

    for i, (mol, mw) in enumerate(zip(reactant_mols, reactant_ws, strict=True)):
        if i > 0:
            plus_x = cursor_x + plus_w // 2
            plus_y = mol_y + mol_h // 2
            draw.text(
                (plus_x, plus_y),
                "+",
                fill="black",
                anchor="mm",
            )
            cursor_x += plus_w
        drawer = rdMolDraw2D.MolDraw2DCairo(mw, mol_h)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        canvas.paste(img, (cursor_x, mol_y), img)
        cursor_x += mw

    cursor_x += arrow_w

    for i, (mol, mw) in enumerate(zip(product_mols, product_ws, strict=True)):
        if i > 0:
            plus_x = cursor_x + plus_w // 2
            plus_y = mol_y + mol_h // 2
            draw.text(
                (plus_x, plus_y),
                "+",
                fill="black",
                anchor="mm",
            )
            cursor_x += plus_w
        drawer = rdMolDraw2D.MolDraw2DCairo(mw, mol_h)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        canvas.paste(img, (cursor_x, mol_y), img)
        cursor_x += mw

    # Draw arrow
    ax1 = pad + sum(reactant_ws) + max(0, n_r - 1) * plus_w + int(10 * scale)
    ax2 = ax1 + arrow_w - int(20 * scale)
    ay = mol_y + mol_h // 2
    draw.line([(ax1, ay), (ax2, ay)], fill="black", width=max(2, int(2 * scale)))
    # Arrowhead
    a_size = int(6 * scale)
    draw.polygon(
        [(ax2, ay), (ax2 - a_size, ay - a_size), (ax2 - a_size, ay + a_size)],
        fill="black",
    )

    # Draw arrow text annotations (C2 fix)
    arrow_mid_x = (ax1 + ax2) // 2
    if arrow_above:
        draw.text(
            (arrow_mid_x, ay - int(10 * scale)),
            arrow_above,
            fill="black",
            anchor="mm",
        )
    if arrow_below:
        draw.text(
            (arrow_mid_x, ay + int(16 * scale)),
            arrow_below,
            fill="black",
            anchor="mm",
        )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def render_step_to_file(
    step: dict,  # type: ignore[type-arg]
    output_path: Path,
    fmt: str = "svg",
    dpi: int = 300,
) -> dict:  # type: ignore[type-arg]
    """Render a step and write to file. Returns JSON-compatible result dict."""
    warnings: list[str] = []

    data = render_step(step, fmt=fmt, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)

    return {
        "status": "ok",
        "output_path": str(output_path),
        "warnings": warnings,
    }
