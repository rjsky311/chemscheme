"""Journal theme system for publication-quality reaction schemes.

Phase 14 — AC-14.1 (SchemeTheme dataclass) + AC-14.2 (journal presets).

Extracts all hardcoded visual constants from renderers into a frozen dataclass
with built-in presets for major journal families.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemeTheme:
    """Frozen dataclass capturing every visual constant across the render path.

    Sub-groups:
      Typography, Arrow, Layout, Connector, Colors, Subscript, Polymer.
    """

    # Identity
    name: str
    description: str

    # --- Typography ---
    font_family: str
    font_size_conditions: int  # arrow-above text base size
    font_size_labels: int  # compound label font size
    font_size_yield: int  # yield text font size
    font_size_legend: int  # substituent legend font size
    font_size_plus: int  # plus sign font size
    label_font_weight: str  # e.g. "normal", "bold"
    text_line_height: int  # vertical spacing between text lines

    # --- Arrow ---
    arrow_width: int  # width of arrow region in px
    arrow_stroke_width: float  # arrow line stroke width
    arrow_text_offset: int  # vertical distance from arrow to text
    arrowhead_scale: int  # arrowhead marker scale
    arrowhead_style: str  # "filled" or "open"

    # --- Layout ---
    mol_height: int  # molecule rendering height
    plus_width: int  # width for "+" separator
    padding: int  # canvas padding
    label_height: int  # height reserved for compound labels
    adaptive_font_threshold: int  # chars beyond which font shrinks

    # --- Connector (multi_step_renderer) ---
    connector_stroke_width: float
    connector_color: str

    # --- Colors ---
    color_text: str
    color_arrow: str
    color_background: str

    # --- Subscript ---
    subscript_dy: int  # tspan dy for subscript offset
    subscript_scale: str  # e.g. "75%"

    # --- Polymer ---
    bracket_stroke_width: float
    polymer_n_font_size: int

    # --- Molecule interior (G3-G5 fix, S67 Part 2) ---
    # When True, molecule rendering uses RDKit's black-white atom palette
    # (no colored heteroatom strokes). Publication journals typically
    # require this. Default False preserves pre-S67 colored behavior.
    molecule_bw_palette: bool = False

    # --- Molecule interior (G4 fix, S69 E1) ---
    # When True, molecule rendering uses a custom in-house SVG drawer that
    # emits bonds as <line> and heteroatom labels as <text> (instead of
    # RDKit's MolDraw2D path-based glyph geometry). This allows downstream
    # vector editors (Illustrator, Inkscape) to treat atom symbols as
    # editable text rather than path data. Carbon atoms remain implicit
    # per chemistry convention. Default False preserves the pre-S69
    # RDKit path-based behavior.
    molecule_atoms_as_text: bool = False


# ---------------------------------------------------------------------------
# AC-14.2: Built-in journal presets
# ---------------------------------------------------------------------------

#: Default theme — byte-identical to pre-Phase-14 hardcoded constants.
DEFAULT_THEME = SchemeTheme(
    name="default",
    description="ChemScheme default style (pre-Phase-14 compatible)",
    # Typography
    font_family="Arial",
    font_size_conditions=11,
    font_size_labels=12,
    font_size_yield=10,
    font_size_legend=11,
    font_size_plus=16,
    label_font_weight="normal",
    text_line_height=22,
    # Arrow
    arrow_width=120,
    arrow_stroke_width=1.5,
    arrow_text_offset=12,
    arrowhead_scale=10,
    arrowhead_style="filled",
    # Layout
    mol_height=200,
    plus_width=30,
    padding=15,
    label_height=20,
    adaptive_font_threshold=50,
    # Connector
    connector_stroke_width=1.5,
    connector_color="black",
    # Colors
    color_text="black",
    color_arrow="black",
    color_background="white",
    # Subscript
    subscript_dy=4,
    subscript_scale="75%",
    # Polymer
    bracket_stroke_width=1.5,
    polymer_n_font_size=12,
)

ACS_THEME = SchemeTheme(
    name="acs",
    description="American Chemical Society journal style (JACS, Org. Lett.)",
    # Typography — ACS prefers Arial/Helvetica, slightly smaller text
    font_family="Arial, Helvetica, sans-serif",
    font_size_conditions=10,
    font_size_labels=11,
    font_size_yield=9,
    font_size_legend=10,
    font_size_plus=14,
    label_font_weight="bold",
    text_line_height=20,
    # Arrow — thinner, more refined
    arrow_width=110,
    arrow_stroke_width=1.2,
    arrow_text_offset=10,
    arrowhead_scale=8,
    arrowhead_style="filled",
    # Layout
    mol_height=180,
    plus_width=25,
    padding=12,
    label_height=18,
    adaptive_font_threshold=45,
    # Connector
    connector_stroke_width=1.2,
    connector_color="black",
    # Colors
    color_text="black",
    color_arrow="black",
    color_background="white",
    # Subscript
    subscript_dy=3,
    subscript_scale="75%",
    # Polymer
    bracket_stroke_width=1.2,
    polymer_n_font_size=11,
    # Molecule — ACS structures are all-black
    molecule_bw_palette=True,
)

NATURE_THEME = SchemeTheme(
    name="nature",
    description="Nature/Science high-impact minimal style",
    # Typography — Nature uses Arial with generous spacing
    font_family="Arial, sans-serif",
    font_size_conditions=10,
    font_size_labels=10,
    font_size_yield=9,
    font_size_legend=10,
    font_size_plus=14,
    label_font_weight="bold",
    text_line_height=20,
    # Arrow — clean, moderate weight
    arrow_width=100,
    arrow_stroke_width=1.0,
    arrow_text_offset=10,
    arrowhead_scale=8,
    arrowhead_style="open",
    # Layout — compact for Nature's narrow columns
    mol_height=170,
    plus_width=22,
    padding=10,
    label_height=16,
    adaptive_font_threshold=40,
    # Connector
    connector_stroke_width=1.0,
    connector_color="black",
    # Colors
    color_text="#333333",
    color_arrow="#333333",
    color_background="white",
    # Subscript
    subscript_dy=3,
    subscript_scale="70%",
    # Polymer
    bracket_stroke_width=1.0,
    polymer_n_font_size=10,
    # Molecule — Nature/Science structures are all-black
    molecule_bw_palette=True,
)

RSC_THEME = SchemeTheme(
    name="rsc",
    description="Royal Society of Chemistry journal style",
    # Typography — RSC uses Times/serif for text
    font_family="Times New Roman, Times, serif",
    font_size_conditions=11,
    font_size_labels=11,
    font_size_yield=10,
    font_size_legend=10,
    font_size_plus=15,
    label_font_weight="bold",
    text_line_height=21,
    # Arrow
    arrow_width=115,
    arrow_stroke_width=1.3,
    arrow_text_offset=11,
    arrowhead_scale=9,
    arrowhead_style="filled",
    # Layout
    mol_height=190,
    plus_width=28,
    padding=14,
    label_height=19,
    adaptive_font_threshold=48,
    # Connector
    connector_stroke_width=1.3,
    connector_color="black",
    # Colors
    color_text="black",
    color_arrow="black",
    color_background="white",
    # Subscript
    subscript_dy=4,
    subscript_scale="75%",
    # Polymer
    bracket_stroke_width=1.3,
    polymer_n_font_size=11,
    # Molecule — RSC structures are all-black
    molecule_bw_palette=True,
)

WILEY_THEME = SchemeTheme(
    name="wiley",
    description="Wiley journal style (Angew. Chem., Adv. Mater.)",
    # Typography — Wiley uses Helvetica
    font_family="Helvetica, Arial, sans-serif",
    font_size_conditions=10,
    font_size_labels=11,
    font_size_yield=9,
    font_size_legend=10,
    font_size_plus=14,
    label_font_weight="bold",
    text_line_height=20,
    # Arrow — moderate
    arrow_width=115,
    arrow_stroke_width=1.2,
    arrow_text_offset=11,
    arrowhead_scale=9,
    arrowhead_style="filled",
    # Layout
    mol_height=185,
    plus_width=26,
    padding=12,
    label_height=18,
    adaptive_font_threshold=45,
    # Connector
    connector_stroke_width=1.2,
    connector_color="black",
    # Colors
    color_text="black",
    color_arrow="black",
    color_background="white",
    # Subscript
    subscript_dy=3,
    subscript_scale="75%",
    # Polymer
    bracket_stroke_width=1.2,
    polymer_n_font_size=11,
    # Molecule — Wiley structures are all-black
    molecule_bw_palette=True,
)

# Registry of available themes
_THEME_REGISTRY: dict[str, SchemeTheme] = {
    "default": DEFAULT_THEME,
    "acs": ACS_THEME,
    "nature": NATURE_THEME,
    "rsc": RSC_THEME,
    "wiley": WILEY_THEME,
}


def get_theme(name: str) -> SchemeTheme:
    """Get a theme by name.

    Args:
        name: Theme name (case-insensitive).

    Returns:
        The corresponding SchemeTheme instance.

    Raises:
        ValueError: If theme name is not recognized, listing available themes.
    """
    key = name.lower()
    if key not in _THEME_REGISTRY:
        available = ", ".join(sorted(_THEME_REGISTRY.keys()))
        raise ValueError(f"Unknown theme '{name}'. Available themes: {available}")
    return _THEME_REGISTRY[key]


def list_themes() -> list[str]:
    """Return sorted list of available theme names."""
    return sorted(_THEME_REGISTRY.keys())
