"""Multi-step scheme renderer — composes individual step renders into a single image.

Uses layout_engine for coordinate math and indigo_renderer for per-step rendering.
"""

from __future__ import annotations

import io
import re
from xml.sax.saxutils import escape as xml_escape

from chemscheme.renderers.layout_engine import LayoutConfig, _route_connections, compute_layout
from chemscheme.renderers.rdkit_renderer import estimate_step_size, render_step
from chemscheme.renderers.themes import DEFAULT_THEME, SchemeTheme


def _safe_id(step_id: str) -> str:
    """Sanitize step_id for use in SVG id/attribute values (C3).

    Replaces non-alphanumeric chars (except hyphen/underscore) with underscore.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", step_id)


def _parse_svg_dimensions(svg_bytes: bytes) -> tuple[float, float]:
    """Extract width and height from an SVG root element.

    Checks viewBox first (more reliable), falls back to width/height attributes (MR4-16).
    """
    svg_str = svg_bytes.decode("utf-8")
    # Try viewBox first: "minX minY width height"
    # Include optional negative sign for minX/minY coordinates (MR5R2-G6)
    vb_match = re.search(r'viewBox="-?[\d.]+\s+-?[\d.]+\s+([\d.]+)\s+([\d.]+)"', svg_str)
    if vb_match:
        return float(vb_match.group(1)), float(vb_match.group(2))
    # Fallback to explicit width/height attributes
    w_match = re.search(r'width="([\d.]+)', svg_str)
    h_match = re.search(r'height="([\d.]+)', svg_str)
    width = float(w_match.group(1)) if w_match else 300.0
    height = float(h_match.group(1)) if h_match else 200.0
    return width, height


def _strip_svg_wrapper(svg_bytes: bytes, step_id: str) -> str:
    """Remove outer <svg>/<xml> tags, prefix IDs to avoid collisions."""
    svg_str = svg_bytes.decode("utf-8")
    # Remove XML declaration
    svg_str = re.sub(r"<\?xml[^?]*\?>\s*", "", svg_str)
    # Extract content between <svg ...> and </svg>
    match = re.search(r"<svg[^>]*>(.*)</svg>", svg_str, re.DOTALL)
    inner = match.group(1) if match else svg_str
    # Prefix ALL id= and href=#id references with step_id to avoid collisions (G8)
    safe = _safe_id(step_id)
    prefix = f"s{safe}-"
    inner = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', inner)
    # Handle both href="#..." and xlink:href="#..." (MR4-1)
    # Process xlink:href first, then plain href with negative lookbehind
    inner = re.sub(r'xlink:href="#([^"]+)"', lambda m: f'xlink:href="#{prefix}{m.group(1)}"', inner)
    inner = re.sub(r'(?<!xlink:)href="#([^"]+)"', lambda m: f'href="#{prefix}{m.group(1)}"', inner)
    # Also handle url(#...) references (e.g. clip-path, filter)
    inner = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{prefix}{m.group(1)})", inner)
    return inner


def _dedup_intermediates(
    steps: list[dict],  # type: ignore[type-arg]
    connections: list[tuple[str, str]],
) -> dict[str, dict]:  # type: ignore[type-arg]
    """Remove shared intermediate compounds from successor step reactants.

    When step N produces compound X and step N+1 consumes compound X,
    X is removed from step N+1's reactants (publication convention: draw once).
    Compound labels for X are transferred to step N's products.

    Args:
        steps: List of step dicts from scheme.json.
        connections: List of (from_id, to_id) tuples from layout.

    Returns:
        Dict mapping step_id → (possibly modified) step dict.
        Original step dicts are not mutated; copies are made when needed.
    """
    import copy

    from rdkit import Chem

    step_map: dict[str, dict] = {s["id"]: s for s in steps}  # type: ignore[type-arg]
    modified: dict[str, dict] = {}  # type: ignore[type-arg]

    def _get(sid: str) -> dict:  # type: ignore[type-arg]
        return modified.get(sid, step_map[sid])

    def _ensure_copy(sid: str) -> dict:  # type: ignore[type-arg]
        if sid not in modified:
            modified[sid] = copy.deepcopy(step_map[sid])
        return modified[sid]

    def _canonical(smiles: str) -> str | None:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None

    for from_id, to_id in connections:
        from_step = _get(from_id)
        to_step = _get(to_id)

        from_products = {_canonical(s) or s for s in from_step.get("products", [])}
        to_reactants = to_step.get("reactants", [])

        shared = []
        remaining = []
        for smi in to_reactants:
            canon = _canonical(smi) or smi
            if canon in from_products:
                shared.append(smi)
            else:
                remaining.append(smi)

        if not shared:
            continue

        # Remove shared reactants from successor step
        to_copy = _ensure_copy(to_id)
        to_copy["reactants"] = remaining

        # Transfer compound_labels for shared intermediates to predecessor
        to_labels = to_copy.get("compound_labels", {})
        if to_labels:
            from_copy = _ensure_copy(from_id)
            from_labels = from_copy.setdefault("compound_labels", {})
            for smi in shared:
                if smi in to_labels:
                    from_labels[smi] = to_labels.pop(smi)

    # Build final step_map: modified steps override originals
    result = dict(step_map)
    result.update(modified)
    return result


def _auto_step_sizes(
    scheme: dict,
    theme: SchemeTheme | None = None,  # type: ignore[type-arg]
) -> dict[str, tuple[float, float]]:
    """Compute per-step sizes from SMILES atom counts."""
    sizes: dict[str, tuple[float, float]] = {}
    for step in scheme.get("steps", []):
        sizes[step["id"]] = estimate_step_size(step, theme=theme)
    return sizes


def _render_scheme_svg(
    scheme: dict,  # type: ignore[type-arg]
    config: LayoutConfig,
    dpi: int,
    theme: SchemeTheme | None = None,
) -> bytes:
    """Render a multi-step scheme to a single composed SVG."""
    if theme is None:
        theme = DEFAULT_THEME

    # D1: Deduplicate shared intermediates before sizing/rendering
    steps = scheme["steps"]
    # Pre-compute connections to identify shared intermediates
    pre_layout = compute_layout(scheme, config=config)
    step_map = _dedup_intermediates(steps, pre_layout.connections)

    # Re-build scheme with deduped steps for sizing
    deduped_scheme = dict(scheme)
    deduped_scheme["steps"] = [step_map[s["id"]] for s in steps]
    step_sizes = _auto_step_sizes(deduped_scheme, theme=theme)
    layout = compute_layout(scheme, config=config, step_sizes=step_sizes)

    t_connector_stroke = theme.connector_stroke_width
    t_connector_color = theme.connector_color

    # Render each step individually and compose
    parts: list[str] = []

    for box in layout.boxes:
        step = step_map[box.step_id]
        step_svg = render_step(step, fmt="svg", dpi=dpi, theme=theme)
        step_w, step_h = _parse_svg_dimensions(step_svg)
        inner = _strip_svg_wrapper(step_svg, box.step_id)

        # Scale the step SVG to fit the layout box, never enlarge (G4)
        scale_x = box.width / step_w if step_w > 0 else 1.0
        scale_y = box.height / step_h if step_h > 0 else 1.0
        scale = min(scale_x, scale_y, 1.0)  # Cap at 1.0 — don't upscale

        # Center within box
        scaled_w = step_w * scale
        scaled_h = step_h * scale
        offset_x = box.x + (box.width - scaled_w) / 2
        offset_y = box.y + (box.height - scaled_h) / 2

        # Escape step_id for safe SVG attribute (C3)
        safe_step_id = xml_escape(box.step_id, {'"': "&quot;"})
        parts.append(
            f'<g data-step-id="{safe_step_id}" '
            f'transform="translate({offset_x},{offset_y}) scale({scale})">'
            f"{inner}</g>"
        )

    # Draw connection arrows between steps with obstacle avoidance (AC-12.3)
    routed = _route_connections(layout.boxes, layout.connections)
    for conn in routed:
        waypoints = conn["waypoints"]
        if len(waypoints) == 2:
            # Straight line
            x1, y1 = waypoints[0]
            x2, y2 = waypoints[1]
            parts.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{t_connector_color}" stroke-width="{t_connector_stroke}" '
                f'marker-end="url(#arrowhead)"/>'
            )
        else:
            # Multi-segment polyline (L-shape or obstacle detour)
            points_str = " ".join(f"{x},{y}" for x, y in waypoints)
            parts.append(
                f'<polyline points="{points_str}" '
                f'fill="none" stroke="{t_connector_color}" '
                f'stroke-width="{t_connector_stroke}" '
                f'marker-end="url(#arrowhead)"/>'
            )

    # Arrowhead marker definition
    defs = (
        "<defs>"
        '<marker id="arrowhead" markerWidth="10" markerHeight="7" '
        'refX="10" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 10 3.5, 0 7" fill="{t_connector_color}"/>'
        "</marker>"
        "</defs>"
    )

    canvas_w = layout.canvas_width
    canvas_h = layout.canvas_height
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{canvas_w}" height="{canvas_h}" '
        f'viewBox="0 0 {canvas_w} {canvas_h}">'
        f"{defs}"
        f"{''.join(parts)}"
        f"</svg>"
    )
    return svg.encode("utf-8")


def _render_scheme_png(
    scheme: dict,  # type: ignore[type-arg]
    config: LayoutConfig,
    dpi: int,
    theme: SchemeTheme | None = None,
) -> bytes:
    """Render a multi-step scheme to PNG by converting the composed SVG."""
    try:
        import cairosvg  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        pass
    else:
        svg_bytes = _render_scheme_svg(scheme, config, dpi, theme=theme)
        return bytes(cairosvg.svg2png(bytestring=svg_bytes, dpi=dpi))

    # Fallback: use Pillow to compose individual step PNGs (MR4.1-P1c)
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "PNG rendering requires CairoSVG or Pillow. "
            "Install with: pip install cairosvg  OR  pip install Pillow"
        ) from None

    # D1: Deduplicate shared intermediates (same as SVG path)
    steps = scheme["steps"]
    pre_layout = compute_layout(scheme, config=config)
    step_map = _dedup_intermediates(steps, pre_layout.connections)
    deduped_scheme = dict(scheme)
    deduped_scheme["steps"] = [step_map[s["id"]] for s in steps]
    step_sizes = _auto_step_sizes(deduped_scheme, theme=theme)
    layout = compute_layout(scheme, config=config, step_sizes=step_sizes)

    canvas_w = int(layout.canvas_width)
    canvas_h = int(layout.canvas_height)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    for box in layout.boxes:
        step = step_map[box.step_id]
        png_bytes = render_step(step, fmt="png", dpi=dpi, theme=theme)
        step_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        # G2 fix: uniform scale to fit box (preserve aspect ratio, no stretching)
        w, h = step_img.size
        bw, bh = int(box.width), int(box.height)
        scale = min(bw / w, bh / h) if w > 0 and h > 0 else 1.0
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        step_img = step_img.resize(
            (new_w, new_h),
            Image.LANCZOS,  # type: ignore[attr-defined]
        )
        # Center within box
        offset_x = int(box.x) + (bw - new_w) // 2
        offset_y = int(box.y) + (bh - new_h) // 2
        canvas.paste(step_img, (offset_x, offset_y), step_img)

    # Draw inter-step connectors (MR4-6)
    from PIL import ImageDraw  # type: ignore[import-untyped]

    draw = ImageDraw.Draw(canvas)
    import math

    # Use routed connections with obstacle avoidance (MR11-C4)
    routed = _route_connections(layout.boxes, layout.connections)
    for conn in routed:
        waypoints = [(int(x), int(y)) for x, y in conn["waypoints"]]
        # Draw each segment of the polyline
        for i in range(len(waypoints) - 1):
            draw.line([waypoints[i], waypoints[i + 1]], fill="black", width=2)
        # Angle-aware arrowhead at the final segment (MR4.1-P2b)
        if len(waypoints) >= 2:
            x1, y1 = waypoints[-2]
            x2, y2 = waypoints[-1]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length > 0:
                ux, uy = dx / length, dy / length
            else:
                ux, uy = 1.0, 0.0
            px, py = -uy, ux
            arrow_size = 6
            ax1 = int(x2 - arrow_size * ux + arrow_size * px)
            ay1 = int(y2 - arrow_size * uy + arrow_size * py)
            ax2 = int(x2 - arrow_size * ux - arrow_size * px)
            ay2 = int(y2 - arrow_size * uy - arrow_size * py)
            draw.polygon([(x2, y2), (ax1, ay1), (ax2, ay2)], fill="black")

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _render_scheme_via_cairosvg(
    scheme: dict,  # type: ignore[type-arg]
    config: LayoutConfig,
    dpi: int,
    theme: SchemeTheme | None,
    *,
    cairo_fn_name: str,
    fmt_label: str,
) -> bytes:
    """G13 (S68): shared PDF/EPS path via cairosvg.{svg2pdf,svg2eps}.

    Falls through to a clean RuntimeError if cairosvg or libcairo is missing —
    Windows users may need to install the Cairo runtime separately.
    """
    try:
        import cairosvg  # type: ignore[import-untyped,import-not-found]
    except (ImportError, OSError) as exc:
        # OSError covers Windows libcairo DLL load failure at import time
        # (cairocffi raises OSError when 'cairo-2'/'libcairo-2.dll' not found)
        msg = (
            f"{fmt_label.upper()} rendering requires cairosvg + libcairo. "
            "Install with: pip install cairosvg  (Windows users may also need GTK runtime "
            "or the cairo DLL — see https://cairosvg.org/documentation/)"
        )
        raise RuntimeError(msg) from exc
    if cairosvg is None:
        msg = f"cairosvg unavailable; cannot render {fmt_label.upper()}."
        raise RuntimeError(msg)
    cairo_fn = getattr(cairosvg, cairo_fn_name, None)
    if cairo_fn is None:
        msg = (
            f"cairosvg.{cairo_fn_name} unavailable. "
            f"Update cairosvg to a version that supports {fmt_label.upper()} export."
        )
        raise RuntimeError(msg)
    svg_bytes = _render_scheme_svg(scheme, config, dpi, theme=theme)
    return bytes(cairo_fn(bytestring=svg_bytes))


def render_scheme(
    scheme: dict,  # type: ignore[type-arg]
    fmt: str = "svg",
    config: LayoutConfig | None = None,
    dpi: int = 300,
    theme: SchemeTheme | None = None,
) -> bytes:
    """Render a complete multi-step scheme to SVG, PNG, PDF, or EPS.

    Args:
        scheme: A validated scheme dict with "steps" list.
        fmt: Output format, "svg" / "png" / "pdf" / "eps".
        config: Layout configuration. Uses defaults if None.
        dpi: Resolution for PNG output.
        theme: Optional journal theme. Uses default if None.

    Returns:
        Rendered image as bytes.

    Raises:
        ValueError: If format is unsupported.
        RuntimeError: If PDF/EPS requested but cairosvg/libcairo unavailable.
    """
    valid_formats = ("svg", "png", "pdf", "eps")
    if fmt not in valid_formats:
        raise ValueError(f"Unsupported format: {fmt!r}. Use one of {valid_formats}.")

    if config is None:
        config = LayoutConfig()

    if fmt == "svg":
        return _render_scheme_svg(scheme, config, dpi, theme=theme)
    if fmt == "png":
        return _render_scheme_png(scheme, config, dpi, theme=theme)
    if fmt == "pdf":
        return _render_scheme_via_cairosvg(
            scheme, config, dpi, theme, cairo_fn_name="svg2pdf", fmt_label="pdf"
        )
    # fmt == "eps"
    return _render_scheme_via_cairosvg(
        scheme, config, dpi, theme, cairo_fn_name="svg2eps", fmt_label="eps"
    )
