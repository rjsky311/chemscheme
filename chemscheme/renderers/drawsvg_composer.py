"""drawsvg-based SVG composition layer for publication-quality reaction schemes.

Replaces manual string-based SVG assembly with structured drawsvg API.
Features:
  - Professional reaction arrows with arrowhead markers
  - Chemistry-aware text formatting (subscripts for formulas like K₂CO₃)
  - RDKit molecule SVGs embedded as nested elements
  - Compound number labels below molecules (AC-11.1)
  - Yield display below arrow (AC-11.2)
  - Multi-line arrow text with adaptive font size (AC-11.3)
  - Structured condition fields auto-formatting (AC-11.4)
  - Clean SVG output suitable for publication
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape as _xml_escape

import drawsvg as draw  # type: ignore[import-untyped]
from rdkit import Chem  # type: ignore[import-untyped]

from chemscheme.renderers.rdkit_renderer import (
    _mol_to_svg,
    _validate_and_prepare_consistent,
)
from chemscheme.renderers.themes import DEFAULT_THEME, SchemeTheme

# Legacy constants kept as aliases to DEFAULT_THEME for internal reference
_MOL_HEIGHT = DEFAULT_THEME.mol_height
_PLUS_WIDTH = DEFAULT_THEME.plus_width
_ARROW_WIDTH = DEFAULT_THEME.arrow_width
_PADDING = DEFAULT_THEME.padding
_TEXT_HEIGHT = DEFAULT_THEME.text_line_height
_ARROW_Y_OFFSET = DEFAULT_THEME.arrow_text_offset
_LABEL_HEIGHT = DEFAULT_THEME.label_height
_LABEL_FONT_SIZE = DEFAULT_THEME.font_size_labels
_YIELD_FONT_SIZE = DEFAULT_THEME.font_size_yield
_ADAPTIVE_FONT_THRESHOLD = DEFAULT_THEME.adaptive_font_threshold


def format_chem_text(text: str, theme: SchemeTheme | None = None) -> str:
    """Format chemistry text with subscript tspans for formula numbers.

    Converts numbers that follow element symbols (uppercase letter optionally
    followed by lowercase) into <tspan> subscript elements.
    E.g., "K2CO3" → "K<tspan ...>2</tspan>CO<tspan ...>3</tspan>"

    Numbers not preceded by element symbols (e.g., "80 C", "12h") are left as-is.

    Args:
        text: Raw chemistry text.
        theme: Optional theme for subscript styling. Uses default if None.
    """
    if theme is None:
        theme = DEFAULT_THEME

    # AC-9.4: Escape XML special chars BEFORE regex processing (MR7-04)
    text = _xml_escape(text)

    sub_dy = theme.subscript_dy
    sub_scale = theme.subscript_scale

    # Pattern: element symbol or closing paren followed by digit(s)
    # Matches: K2, CO3, PPh3, H2O2, (PPh3)4 but NOT "80" or "12h"
    def _sub_match(m: re.Match[str]) -> str:
        prefix = m.group(1)
        number = m.group(2)
        return (
            f'{prefix}<tspan dy="{sub_dy}" font-size="{sub_scale}">'
            f'{number}</tspan><tspan dy="-{sub_dy}">&#8203;</tspan>'
        )

    result = re.sub(r"([A-Z][a-z]?|\))(\d+)", _sub_match, text)
    return result


def _build_arrow_lines(step: dict) -> list[str]:  # type: ignore[type-arg]
    """Build arrow-above text lines from structured fields or raw arrow_above.

    AC-11.4: If structured fields (reagents, solvents, temperature, time, atmosphere)
    are present, they take precedence over raw arrow_above.
    AC-11.3: arrow_above can be a list or contain \\n for multi-line.
    """
    reagents = step.get("reagents", [])
    solvents = step.get("solvents", [])
    temperature = step.get("temperature")
    time_val = step.get("time")
    atmosphere = step.get("atmosphere")

    # C6: Use `is not None` for numeric fields (temperature=0, time=0 are valid)
    has_structured = (
        reagents
        or solvents
        or temperature is not None
        or time_val is not None
        or atmosphere is not None
    )

    if has_structured:
        lines: list[str] = []
        if reagents:
            lines.append(", ".join(reagents))
        if solvents:
            lines.append(", ".join(solvents))
        # Combine temperature, time, atmosphere on one line
        cond_parts: list[str] = []
        if temperature is not None:
            cond_parts.append(str(temperature))
        if time_val is not None:
            cond_parts.append(str(time_val))
        if atmosphere is not None:
            cond_parts.append(str(atmosphere))
        if cond_parts:
            lines.append(", ".join(cond_parts))
        return lines

    # Fall back to raw arrow_above
    arrow_above = step.get("arrow_above", "")
    if isinstance(arrow_above, list):
        return [line for line in arrow_above if line]
    if isinstance(arrow_above, str) and arrow_above:
        # G1: Filter empty strings from \n split (e.g., "Line1\n\nLine2")
        return [line for line in arrow_above.split("\n") if line]
    return []


def _adaptive_font_size(
    lines: list[str], base_size: int = 11, threshold: int = _ADAPTIVE_FONT_THRESHOLD
) -> int:
    """Compute adaptive font size based on longest line length.

    AC-11.3: Long text (>threshold chars) triggers smaller font to fit arrow width.
    """
    if not lines:
        return base_size
    max_len = max(len(line) for line in lines)
    if max_len > threshold:
        # Scale down proportionally, minimum 7px
        return max(7, int(base_size * threshold / max_len))
    return base_size


def _mol_width(n_atoms: int) -> int:
    """Adaptive molecule box width based on atom count (single-molecule fallback)."""
    if n_atoms <= 6:
        return 180
    elif n_atoms <= 15:
        return 250
    else:
        return int(250 * (n_atoms / 15) ** 0.5)


def _uniform_mol_widths(
    all_mols: list,  # type: ignore[type-arg]
    n_reactants: int,
) -> tuple[list[int], list[int]]:
    """Compute molecule widths using a uniform px-per-atom ratio for consistent bond lengths.

    All molecules in a step share the same scale factor (derived from the largest molecule),
    ensuring that identical substructures (e.g., benzene rings) render at the same size.

    Args:
        all_mols: List of RDKit Mol objects (reactants + products, already condensed).
        n_reactants: Number of reactant mols at the start of all_mols.

    Returns:
        (reactant_widths, product_widths) tuple of int lists.
    """
    if not all_mols:
        return [], []

    atom_counts = [m.GetNumAtoms() for m in all_mols]
    max_atoms = max(atom_counts)

    # Reference width from the largest molecule (same formula as _mol_width)
    ref_width = _mol_width(max_atoms)

    # Uniform px-per-atom ratio based on the largest molecule
    px_per_atom = ref_width / max_atoms if max_atoms > 0 else 15.0

    # Apply consistent ratio with a minimum width so tiny molecules stay visible
    min_width = 80
    widths = [max(min_width, int(px_per_atom * n)) for n in atom_counts]

    return widths[:n_reactants], widths[n_reactants:]


def _is_polymer_unit(smiles: str) -> bool:
    """Detect if a SMILES represents a polymer repeating unit.

    Requires 2+ dummy atoms (*) — single dummy is a generic attachment point,
    not a polymer repeat unit (MR12-C4).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    dummy_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    return dummy_count >= 2


def _render_polymer_brackets(
    d: draw.Drawing,  # type: ignore[name-defined]
    x: float,
    y: float,
    width: float,
    height: float,
    n_label: str = "n",
    *,
    font_family: str = "Arial",
    color: str = "black",
    stroke_width: float = 1.5,
    n_font_size: int = 12,
) -> None:
    """Add polymer bracket notation [ ]n around a molecule region in SVG."""
    bracket_inset = 4
    bracket_w = 6
    # Left bracket [
    d.append(
        draw.Raw(
            f'<g data-role="polymer" class="polymer-bracket">'
            f'<path d="M {x + bracket_inset + bracket_w},{y} '
            f"L {x + bracket_inset},{y} "
            f"L {x + bracket_inset},{y + height} "
            f'L {x + bracket_inset + bracket_w},{y + height}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke_width}"/>'
        )
    )
    # Right bracket ]
    rx = x + width - bracket_inset
    d.append(
        draw.Raw(
            f'<path d="M {rx - bracket_w},{y} '
            f"L {rx},{y} "
            f"L {rx},{y + height} "
            f'L {rx - bracket_w},{y + height}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke_width}"/>'
        )
    )
    # Subscript n
    d.append(
        draw.Raw(
            f'<text x="{rx + 4}" y="{y + height + 12}" '
            f'font-family="{font_family}" font-size="{n_font_size}" font-style="italic" '
            f'fill="{color}">{_xml_escape(n_label)}</text>'
            f"</g>"
        )
    )


def compose_step_svg(
    step: dict,  # type: ignore[type-arg]
    mol_height: int | None = None,
    theme: SchemeTheme | None = None,
) -> bytes:
    """Compose a single reaction step SVG using drawsvg.

    Args:
        step: Step dict with reactants, products, arrow_above, optional arrow_below,
              compound_labels, yield, reagents, solvents, temperature, time, atmosphere.
        mol_height: Height for molecule rendering boxes. If None, uses theme value.
        theme: Journal theme for visual styling. Uses default if None.

    Returns:
        Complete SVG as UTF-8 bytes.

    Raises:
        ValueError: If reactants are empty or contain invalid SMILES.
    """
    if theme is None:
        theme = DEFAULT_THEME
    if mol_height is None:
        mol_height = theme.mol_height

    # Theme-derived constants
    t_plus_width = theme.plus_width
    t_arrow_width = theme.arrow_width
    t_padding = theme.padding
    t_text_height = theme.text_line_height
    t_arrow_y_offset = theme.arrow_text_offset
    t_label_height = theme.label_height
    t_label_font_size = theme.font_size_labels
    t_yield_font_size = theme.font_size_yield
    t_font_family = theme.font_family
    t_color_text = theme.color_text
    t_color_arrow = theme.color_arrow
    t_color_bg = theme.color_background
    t_font_plus = theme.font_size_plus
    t_legend_font = theme.font_size_legend
    t_arrow_stroke = theme.arrow_stroke_width
    t_arrowhead_scale = theme.arrowhead_scale
    t_bracket_stroke = theme.bracket_stroke_width
    t_polymer_n_font = theme.polymer_n_font_size
    t_label_weight = theme.label_font_weight

    reactant_smiles = step.get("reactants", [])
    product_smiles = step.get("products", [])
    arrow_below = step.get("arrow_below", "")
    yield_val = step.get("yield", "")
    compound_labels = step.get("compound_labels", {}) or {}
    polymer_info: dict = step.get("polymer_info", {}) or {}  # type: ignore[assignment]
    substituent_map: dict = step.get("substituent_map", {}) or {}  # type: ignore[assignment]

    # Allow empty reactants for deduped intermediate steps (D1)
    # In publication schemes, a continuation step has no visible reactants

    # AC-11.4 / AC-11.3: Build arrow-above lines from structured fields or raw text
    above_lines = _build_arrow_lines(step)
    font_size = _adaptive_font_size(
        above_lines, base_size=theme.font_size_conditions, threshold=theme.adaptive_font_threshold
    )

    # Validate and prepare molecules with step-level consistent abbreviation policy
    all_smiles = reactant_smiles + product_smiles
    if all_smiles:
        reactant_mols, product_mols = _validate_and_prepare_consistent(
            all_smiles, len(reactant_smiles)
        )
    else:
        reactant_mols, product_mols = [], []

    # Calculate dimensions — uniform scaling for consistent bond lengths
    all_mols = list(reactant_mols) + list(product_mols)
    reactant_widths, product_widths = _uniform_mol_widths(all_mols, len(reactant_mols))

    n_r = len(reactant_mols)
    n_p = len(product_mols)
    reactants_total = sum(reactant_widths) + t_plus_width * max(0, n_r - 1)
    products_total = sum(product_widths) + t_plus_width * max(0, n_p - 1)
    total_w = t_padding * 2 + reactants_total + t_arrow_width + products_total

    has_above = bool(above_lines)
    has_below = bool(arrow_below) or bool(yield_val)
    # Multi-line above: reserve extra height per line
    above_text_space = t_text_height * max(1, len(above_lines)) if has_above else 0
    below_text_space = t_text_height if has_below else 0
    # Compound labels: reserve space below molecules
    has_labels = bool(compound_labels)
    label_space = t_label_height if has_labels else 0
    # AC-13.2: Substituent legend table space
    legend_space = (t_text_height * (len(substituent_map) + 1)) if substituent_map else 0
    total_h = (
        t_padding * 2
        + mol_height
        + above_text_space
        + below_text_space
        + label_space
        + legend_space
    )

    # Create drawing
    d = draw.Drawing(total_w, total_h)

    # Background
    d.append(draw.Rectangle(0, 0, total_w, total_h, fill=t_color_bg))

    # Arrowhead marker definition
    marker = draw.Marker(-0.1, -0.5, 0.9, 0.5, scale=t_arrowhead_scale, orient="auto")
    marker.append(draw.Lines(-0.1, -0.5, -0.1, 0.5, 0.9, 0, fill=t_color_arrow, close=True))
    d.append(marker)

    cursor_x = t_padding
    mol_y = t_padding + above_text_space

    # Track molecule positions for compound labels
    mol_positions: list[tuple[float, float, float, str]] = []  # (x, w, y, smiles)

    # --- Reactants ---
    for i, (mol, w, smi) in enumerate(
        zip(reactant_mols, reactant_widths, reactant_smiles, strict=True)
    ):
        if i > 0:
            plus_x = cursor_x + t_plus_width / 2
            plus_y = mol_y + mol_height / 2
            d.append(
                draw.Text(
                    "+",
                    t_font_plus,
                    plus_x,
                    plus_y,
                    text_anchor="middle",
                    dominant_baseline="central",
                    font_family=t_font_family,
                    fill=t_color_text,
                )
            )
            cursor_x += t_plus_width

        mol_svg_inner = _mol_to_svg(mol, w, mol_height, theme=theme)
        g = draw.Raw(f'<g transform="translate({cursor_x},{mol_y})">{mol_svg_inner}</g>')
        d.append(g)
        mol_positions.append((cursor_x, w, mol_y, smi))
        cursor_x += w

    # --- Reaction Arrow ---
    # G14 (S68): arrow_style dispatches between forward / reversible / retro shapes.
    # Default "forward" preserves the single-line single-arrowhead behavior.
    arrow_x1 = cursor_x + 20
    arrow_x2 = cursor_x + t_arrow_width - 20
    arrow_y = mol_y + mol_height / 2
    arrow_style = step.get("arrow_style", "forward")

    if arrow_style == "reversible":
        # Two anti-parallel arrows offset vertically (⇌). Top arrow points right,
        # bottom arrow points left. Each gets its own arrowhead.
        offset = max(3.0, t_arrow_stroke * 2)
        # Top arrow: left → right
        d.append(
            draw.Line(
                arrow_x1,
                arrow_y - offset,
                arrow_x2,
                arrow_y - offset,
                stroke=t_color_arrow,
                stroke_width=t_arrow_stroke,
                marker_end=marker,
                data_arrow_style="reversible",
            )
        )
        # Bottom arrow: right → left
        d.append(
            draw.Line(
                arrow_x2,
                arrow_y + offset,
                arrow_x1,
                arrow_y + offset,
                stroke=t_color_arrow,
                stroke_width=t_arrow_stroke,
                marker_end=marker,
                data_arrow_style="reversible",
            )
        )
    elif arrow_style == "retro":
        # Retrosynthesis arrow (⇒): double-line with single arrowhead. Render
        # as two parallel lines with the upper one carrying the arrowhead.
        offset = max(2.0, t_arrow_stroke * 1.5)
        d.append(
            draw.Line(
                arrow_x1,
                arrow_y - offset,
                arrow_x2,
                arrow_y - offset,
                stroke=t_color_arrow,
                stroke_width=t_arrow_stroke,
                marker_end=marker,
                data_arrow_style="retro",
            )
        )
        d.append(
            draw.Line(
                arrow_x1,
                arrow_y + offset,
                arrow_x2,
                arrow_y + offset,
                stroke=t_color_arrow,
                stroke_width=t_arrow_stroke,
                data_arrow_style="retro",
            )
        )
    else:
        # "forward" (default): single straight arrow with arrowhead.
        d.append(
            draw.Line(
                arrow_x1,
                arrow_y,
                arrow_x2,
                arrow_y,
                stroke=t_color_arrow,
                stroke_width=t_arrow_stroke,
                marker_end=marker,
            )
        )

    # --- Arrow text above (multi-line, AC-11.3) ---
    if above_lines:
        text_x = (arrow_x1 + arrow_x2) / 2
        for line_idx, line in enumerate(above_lines):
            text_y = (
                mol_y
                - t_arrow_y_offset
                - (t_text_height * (len(above_lines) - 1 - line_idx))
                + mol_height / 2
            )
            formatted = format_chem_text(line, theme=theme)
            d.append(
                draw.Raw(
                    f'<text x="{text_x}" y="{text_y}" text-anchor="middle" '
                    f'font-family="{t_font_family}" font-size="{font_size}" '
                    f'fill="{t_color_text}">'
                    f"{formatted}</text>"
                )
            )

    # --- Arrow text below + yield (AC-11.2) ---
    # G7 (S68): arrow_below mirrors arrow_above multiline behavior — accepts
    # str OR list[str]; list elements render as separate <text> lines stacked
    # vertically.
    below_y_cursor = arrow_y + t_arrow_y_offset + 8
    if arrow_below:
        if isinstance(arrow_below, list):
            below_lines = [line for line in arrow_below if line]
        else:
            below_lines = [line for line in str(arrow_below).split("\n") if line]
        text_x = (arrow_x1 + arrow_x2) / 2
        for line in below_lines:
            formatted_below = format_chem_text(line, theme=theme)
            d.append(
                draw.Raw(
                    f'<text x="{text_x}" y="{below_y_cursor}" text-anchor="middle" '
                    f'font-family="{t_font_family}" font-size="10" fill="{t_color_text}">'
                    f"{formatted_below}</text>"
                )
            )
            below_y_cursor += t_text_height

    if yield_val:
        text_x = (arrow_x1 + arrow_x2) / 2
        escaped_yield = _xml_escape(str(yield_val))
        d.append(
            draw.Raw(
                f'<text x="{text_x}" y="{below_y_cursor}" text-anchor="middle" '
                f'font-family="{t_font_family}" font-size="{t_yield_font_size}" '
                f'fill="{t_color_text}" data-role="yield">'
                f"{escaped_yield}</text>"
            )
        )

    cursor_x += t_arrow_width

    # --- Products ---
    for i, (mol, w, smi) in enumerate(
        zip(product_mols, product_widths, product_smiles, strict=True)
    ):
        if i > 0:
            plus_x = cursor_x + t_plus_width / 2
            plus_y = mol_y + mol_height / 2
            d.append(
                draw.Text(
                    "+",
                    t_font_plus,
                    plus_x,
                    plus_y,
                    text_anchor="middle",
                    dominant_baseline="central",
                    font_family=t_font_family,
                    fill=t_color_text,
                )
            )
            cursor_x += t_plus_width

        mol_svg_inner = _mol_to_svg(mol, w, mol_height, theme=theme)
        g = draw.Raw(f'<g transform="translate({cursor_x},{mol_y})">{mol_svg_inner}</g>')
        d.append(g)
        mol_positions.append((cursor_x, w, mol_y, smi))
        cursor_x += w

    # --- AC-13.1: Polymer brackets around repeating units ---
    for mx, mw, my, smi in mol_positions:
        if _is_polymer_unit(smi):
            n_label = "n"
            if smi in polymer_info and isinstance(polymer_info[smi], dict):
                n_label = str(polymer_info[smi].get("n", "n"))  # MR12-C3: coerce to str
            _render_polymer_brackets(
                d,
                mx,
                my,
                mw,
                mol_height,
                n_label,
                font_family=t_font_family,
                color=t_color_text,
                stroke_width=t_bracket_stroke,
                n_font_size=t_polymer_n_font,
            )

    # --- Compound labels below molecules (AC-11.1) ---
    if compound_labels:
        label_y = mol_y + mol_height + t_label_font_size + 4
        for mx, mw, _my, smi in mol_positions:
            label = compound_labels.get(smi)
            if label:
                label_x = mx + mw / 2
                escaped_label = _xml_escape(str(label))
                d.append(
                    draw.Raw(
                        f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
                        f'font-family="{t_font_family}" font-size="{t_label_font_size}" '
                        f'font-weight="{t_label_weight}" fill="{t_color_text}" '
                        f'data-role="compound-label">'
                        f"{escaped_label}</text>"
                    )
                )

    # --- AC-13.2: Substituent legend table ---
    if substituent_map:
        legend_y = mol_y + mol_height + label_space + t_padding + 4
        d.append(
            draw.Raw(
                f'<g data-role="substituent-legend" transform="translate({t_padding},{legend_y})">'
            )
        )
        row_y = 0
        for var_name, variants in substituent_map.items():
            var_escaped = _xml_escape(str(var_name))
            variants_str = ", ".join(_xml_escape(str(v)) for v in variants)
            d.append(
                draw.Raw(
                    f'<text x="0" y="{row_y}" font-family="{t_font_family}" '
                    f'font-size="{t_legend_font}" fill="{t_color_text}">'
                    f'<tspan font-style="italic">{var_escaped}</tspan>'
                    f" = {variants_str}</text>"
                )
            )
            row_y += t_text_height
        d.append(draw.Raw("</g>"))

    svg_str: str = d.as_svg()
    return svg_str.encode("utf-8")
