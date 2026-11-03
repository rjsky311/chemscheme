"""MR9 review regression tests.

Covers fixes for C1 (P1) and C4 (P2) findings from Phase 10 review.
"""

import xml.etree.ElementTree as ET

import pytest

# ===========================================================================
# MR9-C1: compose_step_svg must tolerate products=[] (draft CDX import)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestMR9C1EmptyProducts:
    """compose_step_svg should render reactant-only draft steps without crash."""

    def test_empty_products_renders_svg(self) -> None:
        """GIVEN step with products=[], WHEN compose_step_svg, THEN returns valid SVG."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "draft",
            "reactants": ["c1ccccc1"],
            "products": [],
            "arrow_above": "reagent",
        }
        result = compose_step_svg(step)
        assert isinstance(result, bytes)
        assert b"<svg" in result
        ET.fromstring(result)  # valid XML

    def test_empty_products_via_render_step(self) -> None:
        """GIVEN step with products=[], WHEN render_step(svg), THEN no crash."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "draft",
            "reactants": ["c1ccccc1"],
            "products": [],
            "arrow_above": "test",
        }
        result = render_step(step, fmt="svg")
        assert b"<svg" in result

    def test_empty_products_scheme_renders(self) -> None:
        """GIVEN scheme with draft step (products=[]), WHEN render_scheme, THEN valid SVG."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
            "schema_version": 1,
            "title": "draft",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": [],
                    "arrow_above": "TBD",
                },
            ],
        }
        result = render_scheme(scheme, fmt="svg")
        assert b"<svg" in result

    def test_empty_reactants_renders_for_dedup(self) -> None:
        """GIVEN step with reactants=[] (D1 dedup), WHEN compose_step_svg, THEN renders."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": [],
            "products": ["C"],
            "arrow_above": "test",
        }
        result = compose_step_svg(step)
        assert b"<svg" in result


# ===========================================================================
# MR9-C4: NumHydrogens bare int() without guard
# ===========================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestMR9C4NumHydrogensGuard:
    """NumHydrogens parsing should handle malformed values gracefully."""

    def test_invalid_numhydrogens_no_crash(self) -> None:
        """GIVEN CDXML with NumHydrogens='n/a', WHEN import, THEN no crash."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme

        cdxml = """<?xml version='1.0' encoding='us-ascii'?>
<CDXML>
  <page>
    <fragment id="1">
      <n id="10" p="0 0" Element="6" NumHydrogens="n/a"/>
      <n id="11" p="1 0" Element="6"/>
      <b id="50" B="10" E="11"/>
    </fragment>
    <arrow id="100" FillType="Line"
           Head3D="3 0 0" Tail3D="0 0 0"/>
    <scheme>
      <step ReactionStepReactants="1"
            ReactionStepProducts="1"
            ReactionStepArrows="100"/>
    </scheme>
  </page>
</CDXML>"""
        # Should not crash — malformed NumHydrogens silently ignored
        result = cdxml_to_scheme(cdxml)
        assert "steps" in result
        assert len(result["steps"]) > 0

    def test_valid_numhydrogens_still_works(self) -> None:
        """GIVEN CDXML with NumHydrogens='3', WHEN import, THEN H count preserved."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme

        cdxml = """<?xml version='1.0' encoding='us-ascii'?>
<CDXML>
  <page>
    <fragment id="1">
      <n id="10" p="0 0" Element="7" NumHydrogens="3"/>
    </fragment>
    <arrow id="100" FillType="Line"
           Head3D="3 0 0" Tail3D="0 0 0"/>
    <scheme>
      <step ReactionStepReactants="1"
            ReactionStepProducts="1"
            ReactionStepArrows="100"/>
    </scheme>
  </page>
</CDXML>"""
        result = cdxml_to_scheme(cdxml)
        # NH3 → nitrogen with 3 hydrogens
        smiles = result["steps"][0]["reactants"][0]
        assert "N" in smiles
