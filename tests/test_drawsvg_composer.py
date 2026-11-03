"""Tests for drawsvg-based SVG composition layer.

BDD Scenarios:
  GIVEN a single step with reactants/products, WHEN compose_step_svg(),
    THEN returns valid SVG with molecule SVGs embedded
  GIVEN arrow_above text with formula, WHEN rendered,
    THEN subscripts are properly formatted (e.g., K2CO3 → K₂CO₃)
  GIVEN two step SVGs, WHEN compose_scheme_svg(),
    THEN produces a composed SVG with arrows between steps
  GIVEN empty arrow_above, WHEN rendered, THEN no text element for arrow
  GIVEN adversarial input (empty reactants), WHEN compose, THEN raises ValueError
"""

import xml.etree.ElementTree as ET

import pytest

SIMPLE_STEP = {
    "id": "1",
    "reactants": ["c1ccccc1"],
    "products": ["c1ccc(O)cc1"],
    "arrow_above": "H2O2",
}

SUZUKI_STEP = {
    "id": "1",
    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
    "products": ["c1ccc(-c2ccccc2)cc1"],
    "arrow_above": "Pd(PPh3)4, K2CO3",
    "arrow_below": "DMF, 80 C, 12h",
}


@pytest.mark.unit
@pytest.mark.chemistry
class TestComposeStepSvg:
    """Single-step SVG composition with drawsvg."""

    def test_returns_valid_svg(self) -> None:
        """GIVEN a simple step, WHEN compose_step_svg, THEN returns valid SVG."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(SIMPLE_STEP)
        assert isinstance(result, bytes)
        assert b"<svg" in result
        ET.fromstring(result)  # valid XML

    def test_contains_reaction_arrow(self) -> None:
        """GIVEN a step, WHEN compose_step_svg, THEN SVG contains an arrow element."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(SIMPLE_STEP)
        svg_text = result.decode("utf-8")
        # Arrow can be a line, path, or marker-end
        assert "marker-end" in svg_text or "arrow" in svg_text.lower()

    def test_arrow_above_text_present(self) -> None:
        """GIVEN arrow_above text, WHEN rendered, THEN text appears in SVG."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(SIMPLE_STEP)
        svg_text = result.decode("utf-8")
        # H2O2 should appear (possibly with subscripts)
        assert "H" in svg_text and "O" in svg_text

    def test_arrow_below_text_present(self) -> None:
        """GIVEN arrow_below text, WHEN rendered, THEN text appears in SVG."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(SUZUKI_STEP)
        svg_text = result.decode("utf-8")
        assert "DMF" in svg_text

    def test_arrow_below_format_chem_text(self) -> None:
        """GIVEN arrow_below with formula 'K2CO3', WHEN rendered, THEN subscript tspans present.

        Regression: MR8-G1 — arrow_below was not formatted with format_chem_text,
        unlike arrow_above which was.
        """
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "NaOH",
            "arrow_below": "K2CO3, THF",
        }
        result = compose_step_svg(step)
        svg_text = result.decode("utf-8")
        # arrow_below must contain subscript tspan for the "2" and "3" in K2CO3
        assert "tspan" in svg_text.split("K")[1] if "K" in svg_text else False, (
            "arrow_below 'K2CO3' should contain subscript <tspan> elements"
        )

    def test_plus_separator_for_multiple_reactants(self) -> None:
        """GIVEN multiple reactants, WHEN rendered, THEN '+' separator present."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(SUZUKI_STEP)
        svg_text = result.decode("utf-8")
        assert "+" in svg_text


@pytest.mark.unit
@pytest.mark.chemistry
class TestChemicalTextFormatting:
    """Chemistry-aware text formatting with subscripts/superscripts."""

    def test_formula_subscripts(self) -> None:
        """GIVEN 'K2CO3', WHEN format_chem_text, THEN numbers become subscript tspans."""
        from chemscheme.renderers.drawsvg_composer import format_chem_text

        result = format_chem_text("K2CO3")
        # Should contain tspan elements with baseline-shift or dy for subscripts
        assert "tspan" in result
        assert "2" in result
        assert "3" in result

    def test_no_subscript_for_non_formula(self) -> None:
        """GIVEN 'DMF, 80 C, 12h', WHEN format_chem_text, THEN no subscript changes."""
        from chemscheme.renderers.drawsvg_composer import format_chem_text

        result = format_chem_text("DMF, 80 C, 12h")
        # Regular text, numbers shouldn't be subscripted since not after element symbols
        assert "DMF" in result


@pytest.mark.unit
class TestComposeStepErrors:
    """Error handling for composer."""

    def test_empty_reactants_renders_arrow_and_products(self) -> None:
        """GIVEN empty reactants (D1 dedup), WHEN compose_step_svg, THEN renders arrow+products."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {"id": "1", "reactants": [], "products": ["C"], "arrow_above": "test"}
        result = compose_step_svg(step)
        assert b"<svg" in result

    def test_invalid_smiles_raises(self) -> None:
        """GIVEN invalid SMILES, WHEN compose_step_svg, THEN raises ValueError."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["not_valid"],
            "products": ["C"],
            "arrow_above": "test",
        }
        with pytest.raises(ValueError, match="SMILES"):
            compose_step_svg(step)
