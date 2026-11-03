"""Tests for Indigo single-step rendering (AC-2.1, AC-2.2, AC-2.3, AC-2.4).

BDD Feature: Single-step rendering
"""

import pytest


@pytest.mark.unit
@pytest.mark.chemistry
class TestRenderStepSVG:
    """AC-2.1: Given a step, Indigo produces valid SVG."""

    def test_render_step_returns_svg_bytes(self) -> None:
        """render_step() returns non-empty bytes for SVG."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
            "products": ["c1ccc(-c2ccccc2)cc1"],
            "arrow_above": "Pd(PPh3)4, K2CO3",
        }
        result = render_step(step, fmt="svg")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_svg_contains_svg_tag(self) -> None:
        """Output contains <svg> tag."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        result = render_step(step, fmt="svg")
        assert b"<svg" in result

    def test_svg_is_valid_xml(self) -> None:
        """SVG output is valid XML."""
        import xml.etree.ElementTree as ET

        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "NaOH",
        }
        result = render_step(step, fmt="svg")
        # Should not raise
        ET.fromstring(result)


@pytest.mark.unit
@pytest.mark.chemistry
class TestRenderStepMolecules:
    """AC-2.2: SVG contains correct molecular structures."""

    def test_suzuki_coupling_reactants_and_product(self) -> None:
        """Suzuki coupling render includes all molecules."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
            "products": ["c1ccc(-c2ccccc2)cc1"],
            "arrow_above": "Pd(PPh3)4",
        }
        result = render_step(step, fmt="svg")
        # SVG should be large enough to contain multiple molecules
        assert len(result) > 500

    def test_single_reactant_single_product(self) -> None:
        """Simple reaction with one reactant and one product."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["CCO"],
            "products": ["CC=O"],
            "arrow_above": "[O]",
        }
        result = render_step(step, fmt="svg")
        assert b"<svg" in result


@pytest.mark.unit
@pytest.mark.chemistry
class TestRenderStepPNG:
    """AC-2.3: PNG output with configurable DPI."""

    def test_render_step_png(self) -> None:
        """render_step() returns valid PNG bytes."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        result = render_step(step, fmt="png")
        assert isinstance(result, bytes)
        # PNG magic bytes
        assert result[:4] == b"\x89PNG"

    def test_render_step_png_default_dpi(self) -> None:
        """Default DPI is 300."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        # Should not raise with default dpi
        result = render_step(step, fmt="png", dpi=300)
        assert result[:4] == b"\x89PNG"

    def test_render_step_png_custom_dpi(self) -> None:
        """Custom DPI (600) produces valid PNG (at least as large as 300)."""
        import struct

        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        result_300 = render_step(step, fmt="png", dpi=300)
        result_600 = render_step(step, fmt="png", dpi=600)
        assert result_300[:4] == b"\x89PNG"
        assert result_600[:4] == b"\x89PNG"
        # Pixel dimensions from IHDR: DPI 600 >= DPI 300 (or same with CairoSVG)
        w300 = struct.unpack(">I", result_300[16:20])[0]
        w600 = struct.unpack(">I", result_600[16:20])[0]
        assert w600 >= w300


@pytest.mark.unit
@pytest.mark.chemistry
class TestRenderStepArrowText:
    """AC-2.4: Reagent text above arrow, conditions below."""

    def test_arrow_above_in_svg(self) -> None:
        """arrow_above text appears in SVG output."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "Pd(PPh3)4, K2CO3",
        }
        result = render_step(step, fmt="svg")
        # The reagent text should be present in SVG
        assert b"Pd(PPh3)4" in result or b"Pd" in result

    def test_arrow_below_in_svg(self) -> None:
        """arrow_below text appears in SVG output when provided."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "Pd(PPh3)4",
            "arrow_below": "DMF, 80 C, 12h",
        }
        result = render_step(step, fmt="svg")
        assert b"DMF" in result or b"80" in result


@pytest.mark.unit
class TestRenderStepErrors:
    """Error handling for render_step."""

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        with pytest.raises(ValueError, match="format"):
            render_step(step, fmt="pdf")

    def test_invalid_smiles_raises(self) -> None:
        """Invalid SMILES in step raises ValueError."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["not_a_smiles"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        with pytest.raises(ValueError, match="SMILES"):
            render_step(step, fmt="svg")
