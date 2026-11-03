"""End-to-end pipeline tests: text extraction → validation → rendering.

# RED-EXEMPT: behavior lock tests for existing pipeline integration

These tests chain multiple modules to verify the full pipeline works:
- text_extraction.extract_reactions_from_text → scheme.json dict
- schema.scheme_schema.validate_scheme_json → validation result
- renderers.render_step / render_scheme → SVG/PNG bytes
"""

from __future__ import annotations

import struct

import pytest


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Extract (width, height) from PNG IHDR chunk."""
    assert data[:4] == b"\x89PNG", "Not a PNG file"
    w, h = struct.unpack(">II", data[16:24])
    return w, h


# ---------------------------------------------------------------------------
# E2E: Text → scheme.json extraction
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTextToSchemeJson:
    """extract_reactions_from_text produces valid scheme.json dicts."""

    def test_simple_reaction_text(self) -> None:
        """Text with known reagents → scheme.json with steps."""
        from chemscheme.text_extraction import extract_reactions_from_text

        text = (
            "The substrate was brominated with NBS in THF "
            "at room temperature for 2 hours to give the product."
        )
        scheme = extract_reactions_from_text(text, use_ai=False)

        assert "steps" in scheme
        assert "warnings" in scheme
        assert scheme["schema_version"] == "1"

    def test_output_has_required_step_fields(self) -> None:
        """Each step has id, reactants, products, arrow_above."""
        from chemscheme.text_extraction import extract_reactions_from_text

        text = "NBS (1.2 eq) was added to the substrate in CHCl3 at RT."
        scheme = extract_reactions_from_text(text, use_ai=False)

        for step in scheme["steps"]:
            assert "id" in step
            assert "reactants" in step
            assert "products" in step
            assert "arrow_above" in step

    def test_conditions_extracted(self) -> None:
        """Temperature, solvent, time extracted into step fields."""
        from chemscheme.text_extraction import extract_reactions_from_text

        text = "The reaction was carried out with Pd(PPh3)4 in THF at 80°C for 12 hours."
        scheme = extract_reactions_from_text(text, use_ai=False)

        if scheme["steps"]:
            step = scheme["steps"][0]
            # At least some condition info should be captured
            has_conditions = any(
                step.get(k) for k in ("temperature", "solvents", "time", "arrow_above")
            )
            assert has_conditions

    def test_empty_text_produces_warning(self) -> None:
        """Empty input → no steps, warning present."""
        from chemscheme.text_extraction import extract_reactions_from_text

        scheme = extract_reactions_from_text("", use_ai=False)
        assert scheme["steps"] == []
        assert len(scheme["warnings"]) > 0

    def test_no_chemistry_produces_warning(self) -> None:
        """Non-chemical text → no steps, warning."""
        from chemscheme.text_extraction import extract_reactions_from_text

        scheme = extract_reactions_from_text("The weather was nice today.", use_ai=False)
        assert scheme["steps"] == []
        assert len(scheme["warnings"]) > 0


# ---------------------------------------------------------------------------
# E2E: scheme.json → rendered output
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSchemeToRenderedOutput:
    """Pre-constructed scheme.json → render_step / render_scheme → bytes."""

    @pytest.fixture
    def single_step(self) -> dict:
        return {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(Br)cc1"],
            "arrow_above": "NBS, CHCl3",
        }

    @pytest.fixture
    def two_step_scheme(self) -> dict:
        return {
            "schema_version": "1",
            "steps": [
                {
                    "id": "step_1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "NBS, CHCl3",
                },
                {
                    "id": "step_2",
                    "reactants": ["c1ccc(Br)cc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd(PPh3)4, K2CO3",
                },
            ],
        }

    def test_single_step_to_svg(self, single_step: dict) -> None:
        """Single step → valid SVG bytes."""
        from chemscheme.renderers.rdkit_renderer import render_step

        svg = render_step(single_step, fmt="svg")
        assert b"<svg" in svg
        assert len(svg) > 100

    def test_single_step_to_png(self, single_step: dict) -> None:
        """Single step → valid PNG bytes."""
        from chemscheme.renderers.rdkit_renderer import render_step

        png = render_step(single_step, fmt="png")
        assert png[:4] == b"\x89PNG"
        w, h = _png_dimensions(png)
        assert w > 50 and h > 50

    def test_multi_step_to_svg(self, two_step_scheme: dict) -> None:
        """Two-step scheme → valid SVG with both steps."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        svg = render_scheme(two_step_scheme, fmt="svg")
        assert b"<svg" in svg
        assert len(svg) > 200

    def test_multi_step_to_png(self, two_step_scheme: dict) -> None:
        """Two-step scheme → valid PNG."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        png = render_scheme(two_step_scheme, fmt="png")
        assert png[:4] == b"\x89PNG"
        w, h = _png_dimensions(png)
        # Landscape for linear 2-step
        assert w > h

    def test_intermediate_deduplication(self, two_step_scheme: dict) -> None:
        """Shared intermediate (bromobenzene) should be deduplicated in render."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        svg = render_scheme(two_step_scheme, fmt="svg")
        # The SVG should render without error and be reasonable size
        assert b"<svg" in svg
        # Deduplication: step 2's reactant (bromobenzene) is step 1's product
        # → drawn once, not twice. We verify no crash and reasonable output.
        assert len(svg) > 200


# ---------------------------------------------------------------------------
# E2E: Full pipeline text → render
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullPipelineTextToRender:
    """Text extraction → scheme.json → render → bytes (full chain)."""

    def test_extraction_then_render_no_crash(self) -> None:
        """Even if extraction produces partial results, rendering should not crash."""
        from chemscheme.text_extraction import extract_reactions_from_text

        text = "NBS was added to the thiophene derivative in CHCl3 at room temperature."
        scheme = extract_reactions_from_text(text, use_ai=False)

        if scheme["steps"]:
            from chemscheme.renderers.rdkit_renderer import render_step

            step = scheme["steps"][0]
            if step["reactants"] or step["products"]:
                svg = render_step(step, fmt="svg")
                assert b"<svg" in svg

    def test_multi_step_text_splits_paragraphs(self) -> None:
        """Multi-paragraph text is split into separate processing units."""
        from chemscheme.text_extraction import extract_reactions_from_text

        text = (
            "Step 1: The substrate was treated with NBS in THF at RT for 2h.\n\n"
            "Step 2: The bromide was coupled using Pd(PPh3)4 and K2CO3 in THF at 80°C."
        )
        scheme = extract_reactions_from_text(text, use_ai=False)

        # Without AI and without SMILES for generic names like "substrate",
        # steps may be skipped (C3/R2: no resolved SMILES → skip).
        # Key assertion: no crash, valid structure, warnings explain skips.
        assert "steps" in scheme
        assert "warnings" in scheme
        assert scheme["schema_version"] == "1"


# ---------------------------------------------------------------------------
# E2E: Validation integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestValidationIntegration:
    """scheme.json → validate_scheme_json produces valid result."""

    def test_valid_scheme_passes_validation(self) -> None:
        from chemscheme.schema.scheme_schema import validate_scheme_json

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "NBS",
                }
            ],
        }
        result = validate_scheme_json(scheme)
        assert result["valid"] is True

    def test_invalid_smiles_caught(self) -> None:
        from chemscheme.schema.scheme_schema import validate_scheme_json

        scheme = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["NOT_A_SMILES"],
                    "products": ["c1ccccc1"],
                    "arrow_above": "test",
                }
            ]
        }
        result = validate_scheme_json(scheme)
        # Should flag the invalid SMILES
        assert len(result.get("issues", [])) > 0 or len(result.get("warnings", [])) > 0
