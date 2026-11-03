"""Tests for RDKit-based single-step reaction renderer.

BDD Scenarios:
  GIVEN a valid step, WHEN render_step(fmt="svg"), THEN returns valid SVG bytes
  GIVEN a valid step, WHEN render_step(fmt="png"), THEN returns valid PNG bytes
  GIVEN arrow_above text, WHEN rendered to SVG, THEN text appears in output
  GIVEN arrow_below text, WHEN rendered to SVG, THEN text appears in output
  GIVEN invalid SMILES, WHEN render_step, THEN raises ValueError
  GIVEN unsupported fmt, WHEN render_step, THEN raises ValueError
  GIVEN a spiro compound SMILES, WHEN rendered, THEN produces reasonable SVG (not garbled)
"""

import xml.etree.ElementTree as ET

import pytest

SUZUKI_STEP = {
    "id": "1",
    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
    "products": ["c1ccc(-c2ccccc2)cc1"],
    "arrow_above": "Pd(PPh3)4, K2CO3",
}

SIMPLE_STEP = {
    "id": "1",
    "reactants": ["c1ccccc1"],
    "products": ["c1ccc(O)cc1"],
    "arrow_above": "H2O2",
}

SPIRO_STEP = {
    "id": "1",
    "reactants": ["c1ccc2c(c1)C1(c3ccccc3-2)c2ccccc2-c2ccccc21"],  # 9,9'-spirobifluorene
    "products": ["Brc1ccc2c(c1)C1(c3cc(Br)ccc3-2)c2cc(Br)ccc2-c2ccc(Br)cc21"],
    "arrow_above": "Br2, FeCl3",
    "arrow_below": "CHCl3, reflux",
}


@pytest.mark.unit
@pytest.mark.chemistry
class TestRdkitRenderStepSVG:
    def test_returns_svg_bytes(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(SUZUKI_STEP, fmt="svg")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_svg_contains_svg_tag(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(SIMPLE_STEP, fmt="svg")
        assert b"<svg" in result

    def test_svg_is_valid_xml(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(SIMPLE_STEP, fmt="svg")
        ET.fromstring(result)  # should not raise

    def test_svg_large_enough_for_reaction(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(SUZUKI_STEP, fmt="svg")
        assert len(result) > 500

    def test_spiro_compound_renders_without_error(self) -> None:
        """Complex spiro molecule should render successfully."""
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(SPIRO_STEP, fmt="svg")
        assert b"<svg" in result
        assert len(result) > 1000


@pytest.mark.unit
@pytest.mark.chemistry
class TestRdkitRenderStepPNG:
    def test_returns_png_bytes(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(SIMPLE_STEP, fmt="png")
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_custom_dpi_larger(self) -> None:
        import struct

        from chemscheme.renderers.rdkit_renderer import render_step

        r300 = render_step(SIMPLE_STEP, fmt="png", dpi=300)
        r600 = render_step(SIMPLE_STEP, fmt="png", dpi=600)
        # Both must be valid PNGs
        assert r300[:4] == b"\x89PNG"
        assert r600[:4] == b"\x89PNG"
        # Check pixel dimensions from IHDR chunk (offset 16 = width, 20 = height)
        w300 = struct.unpack(">I", r300[16:20])[0]
        w600 = struct.unpack(">I", r600[16:20])[0]
        # DPI 600 should produce at least as large as DPI 300 (or same if CairoSVG)
        assert w600 >= w300


@pytest.mark.unit
@pytest.mark.chemistry
class TestRdkitRenderStepText:
    def test_arrow_above_in_svg(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        result = render_step(SUZUKI_STEP, fmt="svg")
        assert b"Pd(PPh3)4" in result or b"Pd" in result

    def test_arrow_below_in_svg(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {**SIMPLE_STEP, "arrow_below": "DMF, 80 C, 12h"}
        result = render_step(step, fmt="svg")
        assert b"DMF" in result


@pytest.mark.unit
@pytest.mark.chemistry
class TestRdkitAbbreviationGroups:
    """RDKit abbreviation groups condense common functional groups (Ph, Ac, Bz, etc.)."""

    def test_phenyl_condensed(self) -> None:
        """GIVEN a molecule with terminal phenyl, WHEN condensed, THEN atom count decreases."""
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare

        mols = _validate_and_prepare(["O=C(c1ccccc1)c1ccccc1"], "test")  # benzophenone, 14 atoms
        # Without abbreviation, benzophenone has 14 atoms
        # With Ph abbreviation, it should condense to ~4 atoms (O=C + 2xPh dummy)
        assert mols[0].GetNumAtoms() < 14, "Expected phenyl groups to be condensed"

    def test_default_abbreviations_applied(self) -> None:
        """GIVEN ethyl benzoate, WHEN condensed, THEN Et group is abbreviated."""
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare

        mols = _validate_and_prepare(["CCOC(=O)c1ccccc1"], "test")  # ethyl benzoate, 11 atoms
        # Et and OEt are in default abbreviations
        assert mols[0].GetNumAtoms() < 11, "Expected Et/OEt to be condensed"

    def test_abbreviation_does_not_break_small_molecules(self) -> None:
        """GIVEN a small molecule (benzene), WHEN rendered, THEN no crash from abbreviation."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],  # benzene - too small to abbreviate
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "test",
        }
        result = render_step(step, fmt="svg")
        assert b"<svg" in result

    def test_render_with_abbreviations_produces_valid_svg(self) -> None:
        """GIVEN benzophenone, WHEN rendered with abbreviations, THEN valid SVG."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["O=C(c1ccccc1)c1ccccc1"],  # benzophenone
            "products": ["O=C(c1ccccc1)c1ccc(O)cc1"],
            "arrow_above": "AlCl3",
        }
        result = render_step(step, fmt="svg")
        assert b"<svg" in result
        assert len(result) > 500


@pytest.mark.unit
@pytest.mark.chemistry
class TestUniformMoleculeSizing:
    """Molecules in the same step should use a consistent bond-length scale.

    BDD Scenarios:
      GIVEN a step with small (benzene, 6 atoms) and large (biphenyl, 12 atoms) molecules,
      WHEN widths are computed, THEN width ratio ≈ atom-count ratio (not bucketed jumps).

      GIVEN two benzene rings in different roles (reactant vs product),
      WHEN rendered, THEN their allocated widths are identical.
    """

    def test_width_ratio_proportional_to_atom_count(self) -> None:
        """Small and large molecules should have widths proportional to atom counts."""
        from chemscheme.renderers.drawsvg_composer import _uniform_mol_widths
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare

        # Benzene (6 atoms post-condensation) + biphenyl (large, ~12 atoms)
        mols = _validate_and_prepare(["c1ccccc1", "c1ccc(-c2ccccc2)cc1"], "test")
        r_widths, p_widths = _uniform_mol_widths(mols, n_reactants=1)
        all_widths = r_widths + p_widths

        # Width ratio should approximate atom-count ratio (±30% tolerance)
        n0 = mols[0].GetNumAtoms()
        n1 = mols[1].GetNumAtoms()
        expected_ratio = n0 / n1
        actual_ratio = all_widths[0] / all_widths[1]
        assert abs(actual_ratio - expected_ratio) < 0.3, (
            f"Width ratio {actual_ratio:.2f} deviates from atom ratio {expected_ratio:.2f}"
        )

    def test_same_molecule_same_width(self) -> None:
        """Two identical molecules should get identical widths regardless of role."""
        from chemscheme.renderers.drawsvg_composer import _uniform_mol_widths
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare

        # Same benzene as reactant and product
        mols = _validate_and_prepare(["c1ccccc1", "c1ccccc1"], "test")
        r_widths, p_widths = _uniform_mol_widths(mols, n_reactants=1)
        assert r_widths[0] == p_widths[0], "Same molecule should get same width"

    def test_no_bucketed_jump_at_boundary(self) -> None:
        """6-atom and 7-atom molecules should NOT have a huge width jump."""
        from chemscheme.renderers.drawsvg_composer import _uniform_mol_widths
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare

        # 6-atom (benzene) + 7-atom (toluene) — old code had 180 vs 250 jump
        mols = _validate_and_prepare(["c1ccccc1", "Cc1ccccc1"], "test")
        r_widths, p_widths = _uniform_mol_widths(mols, n_reactants=1)
        all_widths = r_widths + p_widths
        ratio = all_widths[1] / all_widths[0]
        # Should be close to 7/6 ≈ 1.17, NOT 250/180 ≈ 1.39
        assert ratio < 1.3, f"Width jump {ratio:.2f} too large for 6→7 atom boundary"


@pytest.mark.unit
@pytest.mark.chemistry
class TestAbbreviationConsistency:
    """Abbreviation decisions should be consistent across molecules in a step.

    BDD Scenarios:
      GIVEN a step with bromobenzene + boronic acid → biphenyl,
      WHEN biphenyl can't abbreviate Ph (100% coverage),
      THEN reactants also don't abbreviate Ph (all draw full rings).

      GIVEN a step where ALL molecules can abbreviate Ph,
      THEN all use Ph abbreviation consistently.

      GIVEN a step with no phenyl groups at all,
      THEN abbreviation consistency is trivially satisfied.
    """

    def test_phenyl_condensed_in_small_molecule(self) -> None:
        """Phenol (7 atoms) should also get Ph abbreviation when standalone."""
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare

        # Phenol: with maxCoverage=0.95, Ph (86% coverage) should be allowed
        mols = _validate_and_prepare(["c1ccc(O)cc1"], "test")  # phenol
        assert mols[0].GetNumAtoms() < 7, "Phenol should have Ph abbreviated for consistency"

    def test_step_level_fallback_when_one_cant_abbreviate(self) -> None:
        """If biphenyl can't abbreviate Ph, reactants shouldn't either."""
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare_consistent

        # Suzuki: BrPh + PhB(OH)2 → biphenyl
        # biphenyl can't condense Ph (100% coverage) → all should stay uncondensed
        r_mols, p_mols = _validate_and_prepare_consistent(
            ["c1ccc(Br)cc1", "OB(O)c1ccccc1", "c1ccc(-c2ccccc2)cc1"],
            n_reactants=2,
        )
        # Biphenyl should NOT be condensed (12 atoms stays 12)
        assert p_mols[0].GetNumAtoms() == 12, "Biphenyl shouldn't be condensed"
        # KEY: Reactants should ALSO not be condensed (full ring structures)
        bromobenzene_atoms = r_mols[0].GetNumAtoms()
        assert bromobenzene_atoms == 7, (
            f"Bromobenzene should keep full rings (7 atoms), got {bromobenzene_atoms}. "
            "Step-level consistency: if product can't abbreviate, reactants shouldn't either."
        )

    def test_all_can_abbreviate_then_all_do(self) -> None:
        """If ALL molecules can abbreviate Ph, they should all use it."""
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare_consistent

        # Buchwald: BrPh + PhNH2 → Ph-NH-Ph (diphenylamine, 13 atoms)
        # All can condense Ph → all should condense
        r_mols, p_mols = _validate_and_prepare_consistent(
            ["c1ccc(Br)cc1", "Nc1ccccc1", "c1ccc(Nc2ccccc2)cc1"],
            n_reactants=2,
        )
        assert r_mols[0].GetNumAtoms() < 7, "Bromobenzene should abbreviate Ph"
        assert r_mols[1].GetNumAtoms() < 7, "Aniline should abbreviate Ph"
        assert p_mols[0].GetNumAtoms() < 13, "Diphenylamine should abbreviate Ph"

    def test_no_phenyl_groups_trivially_consistent(self) -> None:
        """Step with no phenyl groups: consistency is trivially satisfied."""
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare_consistent

        # Alkyl reaction: no phenyl groups
        r_mols, p_mols = _validate_and_prepare_consistent(
            ["CCO", "CC=O"],
            n_reactants=1,
        )
        assert r_mols[0].GetNumAtoms() > 0
        assert p_mols[0].GetNumAtoms() > 0


@pytest.mark.unit
class TestRdkitRenderStepErrors:
    def test_invalid_format_raises(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        with pytest.raises(ValueError, match="format"):
            render_step(SIMPLE_STEP, fmt="pdf")

    def test_invalid_smiles_raises(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["not_a_smiles"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        with pytest.raises(ValueError, match="SMILES"):
            render_step(step, fmt="svg")
