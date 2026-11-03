"""Indigo-based single-step reaction renderer.

Renders a reaction step (reactants + products + arrow text) to SVG or PNG.
"""

import re
import threading
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from indigo import Indigo  # type: ignore[import-untyped]
from indigo.renderer import IndigoRenderer  # type: ignore[import-untyped]

# MR4.1-P1a: Indigo C++ core may not be thread-safe for concurrent
# option-setting + rendering.  Serialize all render_step calls.
_render_lock = threading.Lock()


def _build_reaction_smiles(step: dict) -> str:  # type: ignore[type-arg]
    """Build a reaction SMILES string from a step dict.

    Format: reactant1.reactant2>>product1.product2
    """
    reactants = ".".join(step["reactants"])
    products = ".".join(step["products"])
    return f"{reactants}>>{products}"


def _validate_smiles(indigo: Indigo, smiles_list: list[str], role: str) -> None:
    """Validate that all SMILES in a list can be loaded by Indigo."""
    for s in smiles_list:
        try:
            indigo.loadMolecule(s)
        except Exception as err:
            raise ValueError(f"Invalid SMILES in {role}: {s}") from err


def _inject_text_into_svg(svg: bytes, arrow_above: str, arrow_below: str) -> bytes:
    """Inject arrow_above/below as searchable SVG text elements.

    Indigo renders comments as paths (not text), so we add text elements
    manually to ensure they're searchable and accessible.
    """
    svg_str = svg.decode("utf-8")

    # Find the SVG dimensions to position text
    width_match = re.search(r'width="([\d.]+)"', svg_str)
    height_match = re.search(r'height="([\d.]+)"', svg_str)
    if not width_match or not height_match:
        return svg

    width = float(width_match.group(1))
    height = float(height_match.group(1))
    center_x = width / 2

    text_elements = []
    if arrow_above:
        text_elements.append(
            f'<text x="{center_x}" y="15" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="black">'
            f"{xml_escape(arrow_above)}</text>"
        )
    if arrow_below:
        text_elements.append(
            f'<text x="{center_x}" y="{height - 5}" text-anchor="middle" '
            f'font-family="Arial" font-size="11" fill="black">'
            f"{xml_escape(arrow_below)}</text>"
        )

    if text_elements:
        insert_text = "\n".join(text_elements)
        svg_str = svg_str.replace("</svg>", f"{insert_text}\n</svg>")

    return svg_str.encode("utf-8")


def render_step(step: dict, fmt: str = "svg", dpi: int = 300) -> bytes:  # type: ignore[type-arg]
    """Render a single reaction step to SVG or PNG bytes.

    Args:
        step: Step dict with reactants, products, arrow_above, optional arrow_below.
        fmt: Output format, "svg" or "png".
        dpi: Resolution for PNG output (default 300). Controls image size.

    Returns:
        Rendered image as bytes.

    Raises:
        ValueError: If format is unsupported or SMILES are invalid.
    """
    if fmt not in ("svg", "png"):
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'svg' or 'png'.")

    with _render_lock:  # MR4.1-P1a: serialize Indigo access
        indigo = Indigo()

        # Validate SMILES
        _validate_smiles(indigo, step["reactants"], "reactants")
        _validate_smiles(indigo, step["products"], "products")

        # Create renderer FIRST — this registers render options
        renderer = IndigoRenderer(indigo)

        # Set rendering options
        indigo.setOption("render-output-format", fmt)
        indigo.setOption("render-bond-length", 40.0)
        indigo.setOption("render-relative-thickness", 1.5)

        if fmt == "png":
            # Scale image size based on DPI (base size at 300 DPI)
            scale = dpi / 300.0
            base_width = 800
            base_height = 400
            indigo.setOption("render-image-size", int(base_width * scale), int(base_height * scale))

        # Build and load reaction
        rxn_smiles = _build_reaction_smiles(step)
        rxn = indigo.loadReaction(rxn_smiles)

        result = bytes(renderer.renderToBuffer(rxn))

    # For SVG, inject searchable text annotations (outside lock — no Indigo)
    if fmt == "svg":
        arrow_above = step.get("arrow_above", "")
        # MR13-G2: Normalize list-form arrow_above to string
        if isinstance(arrow_above, list):
            arrow_above = "\n".join(arrow_above)
        arrow_below = step.get("arrow_below", "")
        if arrow_above or arrow_below:
            result = _inject_text_into_svg(result, arrow_above, arrow_below)

    return result


def render_step_to_file(
    step: dict,  # type: ignore[type-arg]
    output_path: Path,
    fmt: str = "svg",
    dpi: int = 300,
) -> dict:  # type: ignore[type-arg]
    """Render a step and write to file. Returns JSON-compatible result dict.

    Args:
        step: Step dict.
        output_path: Where to write the output file.
        fmt: Output format, "svg" or "png".
        dpi: Resolution for PNG (default 300).

    Returns:
        Dict with keys: status, output_path, warnings.
    """
    warnings: list[str] = []

    data = render_step(step, fmt=fmt, dpi=dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)

    return {
        "status": "ok",
        "output_path": str(output_path),
        "warnings": warnings,
    }
