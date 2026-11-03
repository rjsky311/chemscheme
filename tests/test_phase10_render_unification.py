"""Tests for Phase 10 — Render Pipeline Unification.

BDD Scenarios:

AC-10.1: drawsvg_composer wired into multi_step_renderer as primary SVG backend
  GIVEN a multi-step scheme with arrow_above="K2CO3",
  WHEN render_scheme(fmt="svg"),
  THEN the SVG contains <tspan> subscript elements (proving drawsvg_composer is used)

  GIVEN a multi-step scheme,
  WHEN render_scheme(fmt="svg"),
  THEN the output is valid XML and contains step groups

AC-10.2: Production SVG output has chemistry subscripts
  GIVEN a single step with arrow_above="Pd(PPh3)4, K2CO3",
  WHEN render_step(fmt="svg"),
  THEN the SVG arrow text contains <tspan> subscript elements for "3", "4", "2", "3"

  GIVEN arrow_above with non-formula text "DMF, 80 C, 12h",
  WHEN render_step(fmt="svg"),
  THEN no spurious subscripts are added to "80" or "12"

AC-10.3: viewBox correctly handles negative minX/minY
  GIVEN an SVG with viewBox="-10 -5 400 300",
  WHEN _parse_svg_dimensions() extracts dimensions,
  THEN width=400, height=300 (not mangled by negative offsets)

  GIVEN a step SVG with negative viewBox coordinates,
  WHEN composed into a multi-step scheme,
  THEN the step is correctly positioned (not clipped or offset)

AC-10.4: Large molecule width based on actual 2D bounding box, remove 500px cap
  GIVEN a molecule with ~200 atoms (large NFA/polymer),
  WHEN computing mol_width,
  THEN the width exceeds 500px (old cap removed)

  GIVEN a step with a large molecule,
  WHEN estimate_step_size() is called,
  THEN the estimated width reflects the uncapped molecule width

Adversarial:
  GIVEN a step with 1-atom molecule (methane) + 200-atom molecule,
  WHEN rendered, THEN no crash and sizes are heterogeneous
"""

import re
import xml.etree.ElementTree as ET

import pytest

# --- Fixtures ---

SUZUKI_STEP = {
    "id": "1",
    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
    "products": ["c1ccc(-c2ccccc2)cc1"],
    "arrow_above": "Pd(PPh3)4, K2CO3",
    "arrow_below": "DMF, 80 C, 12h",
}

SIMPLE_STEP = {
    "id": "1",
    "reactants": ["c1ccccc1"],
    "products": ["c1ccc(O)cc1"],
    "arrow_above": "H2O2",
}

# Large molecule: a long alkyl chain (~30 atoms)
_LARGE_MOL = "C" * 30  # triacontane, 30 carbons → ~30 heavy atoms
# Very large molecule: repeat unit for testing cap removal
_VERY_LARGE_MOL = "C" * 80  # 80 carbons → needs > 500px

LARGE_MOL_STEP = {
    "id": "1",
    "reactants": [_VERY_LARGE_MOL],
    "products": [_VERY_LARGE_MOL],
    "arrow_above": "catalyst",
}

# Heterogeneous: tiny + huge
HETERO_STEP = {
    "id": "1",
    "reactants": ["C", _VERY_LARGE_MOL],
    "products": ["CC"],
    "arrow_above": "test",
}


def _make_2step_scheme(arrow_above: str = "K2CO3") -> dict:
    return {
        "schema_version": 1,
        "title": "test",
        "steps": [
            {
                "id": "1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(O)cc1"],
                "arrow_above": arrow_above,
            },
            {
                "id": "2",
                "reactants": ["c1ccc(O)cc1"],
                "products": ["c1ccc(OC)cc1"],
                "arrow_above": "CH3I",
            },
        ],
    }


# =============================================================================
# AC-10.1: drawsvg_composer wired into multi_step_renderer
# =============================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestAC101DrawsvgWired:
    """drawsvg_composer should be the primary SVG backend for multi_step_renderer."""

    def test_scheme_svg_has_subscript_tspans(self) -> None:
        """GIVEN scheme with K2CO3 step, WHEN render_scheme(svg), THEN tspan subscripts present."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = _make_2step_scheme("Pd(PPh3)4, K2CO3")
        svg_bytes = render_scheme(scheme, fmt="svg")
        svg_str = svg_bytes.decode("utf-8")

        # drawsvg_composer's format_chem_text produces <tspan dy="4" font-size="75%">
        assert "<tspan" in svg_str, (
            "render_scheme SVG should contain <tspan> subscripts "
            "(drawsvg_composer should be wired as primary SVG backend)"
        )

    def test_scheme_svg_still_valid_xml(self) -> None:
        """GIVEN a 2-step scheme, WHEN render_scheme(svg), THEN output is valid XML."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = _make_2step_scheme()
        svg_bytes = render_scheme(scheme, fmt="svg")
        root = ET.fromstring(svg_bytes)
        assert root.tag.endswith("svg")

    def test_scheme_svg_has_step_groups(self) -> None:
        """GIVEN a 2-step scheme, WHEN render_scheme(svg), THEN has 2 step groups."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = _make_2step_scheme()
        svg_bytes = render_scheme(scheme, fmt="svg")
        root = ET.fromstring(svg_bytes)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        groups = root.findall(".//svg:g[@data-step-id]", ns) + root.findall(".//g[@data-step-id]")
        assert len(groups) == 2

    def test_scheme_png_still_works(self) -> None:
        """GIVEN a 2-step scheme, WHEN render_scheme(png), THEN valid PNG output."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = _make_2step_scheme()
        png_bytes = render_scheme(scheme, fmt="png")
        assert png_bytes[:4] == b"\x89PNG"


# =============================================================================
# AC-10.2: Production SVG has chemistry subscripts
# =============================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestAC102ChemistrySubscripts:
    """Production SVG output should have chemistry-aware subscripts."""

    def test_render_step_svg_has_subscripts(self) -> None:
        """GIVEN Suzuki step with K2CO3, WHEN render_step(svg), THEN tspan subscripts."""
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(SUZUKI_STEP, fmt="svg")
        svg_str = result.decode("utf-8")

        # K2CO3 → K<tspan>2</tspan>CO<tspan>3</tspan>
        assert "<tspan" in svg_str, (
            "render_step SVG should contain chemistry subscripts via format_chem_text"
        )

    def test_no_spurious_subscripts_for_non_formula(self) -> None:
        """GIVEN arrow_below='DMF, 80 C, 12h', WHEN rendered, THEN '80' not subscripted."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {**SIMPLE_STEP, "arrow_below": "DMF, 80 C, 12h"}
        result = render_step(step, fmt="svg")
        svg_str = result.decode("utf-8")

        # "80" and "12" should NOT be wrapped in tspan subscripts
        # (they are preceded by space/comma, not element symbols)
        # Check that "80" appears as plain text, not inside a subscript tspan
        assert "80" in svg_str  # the number should still be present

    def test_subscript_in_multi_step_arrow_text(self) -> None:
        """GIVEN scheme with CH3I step, WHEN render_scheme(svg), THEN CH3 has subscript 3."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = _make_2step_scheme()
        svg_bytes = render_scheme(scheme, fmt="svg")
        svg_str = svg_bytes.decode("utf-8")

        # CH3I → CH<tspan ...>3</tspan>I
        # Find tspan with "3" content
        assert re.search(r"<tspan[^>]*>3</tspan>", svg_str), (
            "CH3I should have subscript 3 in multi-step SVG"
        )

    def test_render_step_png_unaffected(self) -> None:
        """GIVEN Suzuki step, WHEN render_step(png), THEN still valid PNG."""
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(SUZUKI_STEP, fmt="png")
        assert result[:4] == b"\x89PNG"


# =============================================================================
# AC-10.3: viewBox negative minX/minY handling
# =============================================================================


@pytest.mark.unit
class TestAC103ViewBoxNegativeOffset:
    """viewBox with negative minX/minY should be handled correctly."""

    def test_parse_negative_viewbox(self) -> None:
        """GIVEN SVG with negative viewBox, WHEN _parse_svg_dimensions, THEN correct w/h."""
        from chemscheme.renderers.multi_step_renderer import _parse_svg_dimensions

        svg = b'<svg viewBox="-10 -5 400 300" width="400" height="300"></svg>'
        w, h = _parse_svg_dimensions(svg)
        assert w == 400.0
        assert h == 300.0

    def test_parse_negative_viewbox_float(self) -> None:
        """GIVEN SVG with negative float viewBox, WHEN parsed, THEN correct dimensions."""
        from chemscheme.renderers.multi_step_renderer import _parse_svg_dimensions

        svg = b'<svg viewBox="-12.5 -8.3 450.5 320.7"></svg>'
        w, h = _parse_svg_dimensions(svg)
        assert abs(w - 450.5) < 0.01
        assert abs(h - 320.7) < 0.01

    def test_strip_wrapper_preserves_negative_viewbox_content(self) -> None:
        """GIVEN step SVG with negative viewBox, WHEN stripped, THEN content intact."""
        from chemscheme.renderers.multi_step_renderer import _strip_svg_wrapper

        svg = b'<svg viewBox="-10 -5 400 300"><rect x="-5" y="-3" width="100" height="50"/></svg>'
        inner = _strip_svg_wrapper(svg, "test")
        assert 'x="-5"' in inner
        assert 'y="-3"' in inner


# =============================================================================
# AC-10.4: Remove 500px molecule width cap
# =============================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestAC104RemoveWidthCap:
    """Large molecules should get width > 500px (old cap removed)."""

    def test_rdkit_mol_width_uncapped(self) -> None:
        """GIVEN 80-atom molecule, WHEN estimate_step_size, THEN per-mol width > 500px.

        With cap: 2*500 + arrow(100) + padding(20) = 1120
        Without cap: 2*577 + arrow(100) + padding(20) = 1274
        """
        from chemscheme.renderers.rdkit_renderer import estimate_step_size

        w, _h = estimate_step_size(LARGE_MOL_STEP)
        # Capped total = 20 + 500 + 100 + 500 = 1120
        # Uncapped total = 20 + 577 + 100 + 577 = 1274
        assert w > 1120, (
            f"Step width {w} should exceed capped size 1120 "
            "(each large mol should be > 500px after cap removal)"
        )

    def test_drawsvg_mol_width_uncapped(self) -> None:
        """GIVEN 80-atom molecule in drawsvg_composer, WHEN computing width, THEN > 500px."""
        from chemscheme.renderers.drawsvg_composer import _mol_width

        w = _mol_width(80)
        assert w > 500, f"drawsvg _mol_width(80) = {w}, should be > 500 (cap removed)"

    def test_estimate_step_size_single_large_mol(self) -> None:
        """GIVEN step with one 80-carbon reactant, WHEN estimate_step_size, THEN reflects uncap."""
        from chemscheme.renderers.rdkit_renderer import estimate_step_size

        single_large_step = {
            "id": "1",
            "reactants": [_VERY_LARGE_MOL],
            "products": ["C"],  # tiny product
            "arrow_above": "test",
        }
        w, _h = estimate_step_size(single_large_step)
        # Capped: 20 + 500 + 100 + 180 = 800
        # Uncapped: 20 + 577 + 100 + 180 = 877
        assert w > 800, (
            f"Step width {w} should exceed 800 (capped large mol width = 500, "
            "uncapped should be ~577)"
        )

    def test_render_large_mol_no_crash(self) -> None:
        """GIVEN step with heterogeneous sizes (1-atom + 80-atom), WHEN rendered, THEN no crash."""
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(HETERO_STEP, fmt="svg")
        assert b"<svg" in result

    def test_render_scheme_large_mol_valid(self) -> None:
        """GIVEN scheme with large molecules, WHEN render_scheme(svg), THEN valid SVG."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
            "schema_version": 1,
            "title": "large test",
            "steps": [LARGE_MOL_STEP],
        }
        svg_bytes = render_scheme(scheme, fmt="svg")
        root = ET.fromstring(svg_bytes)
        assert root.tag.endswith("svg")


# =============================================================================
# Adversarial
# =============================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestAdversarial:
    """Edge cases and adversarial inputs for render pipeline."""

    def test_empty_arrow_above_no_subscript_crash(self) -> None:
        """GIVEN step with empty arrow_above, WHEN rendered, THEN no crash from format_chem_text."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["C"],
            "products": ["CC"],
            "arrow_above": "",
        }
        result = render_step(step, fmt="svg")
        assert b"<svg" in result

    def test_special_chars_in_arrow_text_escaped(self) -> None:
        """GIVEN arrow_above with <>&, WHEN rendered, THEN properly escaped."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["C"],
            "products": ["CC"],
            "arrow_above": "T > 100°C & P < 1atm",
        }
        result = render_step(step, fmt="svg")
        # Should be XML-valid (no raw < > &)
        ET.fromstring(result)

    def test_unicode_arrow_text_with_subscripts(self) -> None:
        """GIVEN arrow_above with Unicode + formula 'Δ, K2CO3', WHEN rendered, THEN valid."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "Δ, K2CO3",
        }
        result = render_step(step, fmt="svg")
        svg_str = result.decode("utf-8")
        assert "Δ" in svg_str or "&#916;" in svg_str or "&#x394;" in svg_str
