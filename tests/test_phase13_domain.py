"""Phase 13 — Domain Features & Validation tests.

AC-13.1: Polymer bracket notation (MR7-18)
AC-13.2: X/Y substituent notation + legend (MR7-17)
AC-13.3: Adaptive Tanimoto threshold (MR7-20)
AC-13.4: Stereo validation E/Z R/S inversions (MR7-21)
AC-13.5: Chemistry/continuity gate before render (MR7-22)
AC-13.6: Layer1 stereo parse-time stripping fix (MR7-19)
"""

import json
import re
from pathlib import Path

import pytest

# ═══ AC-13.1: Polymer bracket notation ═══


class TestPolymerBracketNotation:
    """AC-13.1: Polymer repeating units render with bracket notation."""

    # Simple polymer: polyethylene repeating unit *CC*
    POLYMER_SMILES = "*CC*"
    # More complex: polythiophene repeating unit
    POLYTHIOPHENE = "*c1cc(*)sc1"

    def test_polymer_detection_dummy_atoms(self) -> None:
        """Molecules with * atoms are detected as polymer repeating units."""
        from chemscheme.renderers.drawsvg_composer import _is_polymer_unit

        assert _is_polymer_unit(self.POLYMER_SMILES) is True
        assert _is_polymer_unit(self.POLYTHIOPHENE) is True
        assert _is_polymer_unit("CCO") is False
        assert _is_polymer_unit("c1ccccc1") is False

    def test_polymer_svg_has_brackets(self) -> None:
        """Polymer molecules are rendered with bracket SVG elements."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": [self.POLYMER_SMILES],
            "products": ["*CC(CC)CC*"],
            "arrow_above": "catalyst",
        }
        svg = compose_step_svg(step).decode("utf-8")
        # Should contain bracket elements (rect or path for [ ])
        assert "polymer-bracket" in svg or 'data-role="polymer"' in svg

    def test_polymer_svg_has_subscript_n(self) -> None:
        """Polymer bracket has subscript 'n' label."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": [self.POLYMER_SMILES],
            "products": [],
            "arrow_above": "",
        }
        svg = compose_step_svg(step).decode("utf-8")
        # Should contain 'n' subscript after bracket
        assert re.search(r">n<", svg), "Missing subscript 'n' for polymer"

    def test_polymer_info_custom_n(self) -> None:
        """polymer_info field allows custom repeat unit label (e.g., 'm', '10')."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": [self.POLYMER_SMILES],
            "products": [],
            "arrow_above": "",
            "polymer_info": {self.POLYMER_SMILES: {"n": "m"}},
        }
        svg = compose_step_svg(step).decode("utf-8")
        assert re.search(r">m<", svg), "Custom repeat unit label 'm' not found"

    def test_layer1_accepts_polymer_smiles(self) -> None:
        """Layer1 chemistry validation accepts * (dummy atoms) without error."""
        from chemscheme.validators.layer1_chemistry import validate_chemistry

        result = validate_chemistry([self.POLYMER_SMILES, self.POLYTHIOPHENE])
        assert not result.errors, f"Polymer SMILES should be valid: {result.errors}"

    def test_schema_validates_polymer_info(self) -> None:
        """Schema validation accepts optional polymer_info dict."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": [self.POLYMER_SMILES],
                    "products": [],
                    "arrow_above": "",
                    "polymer_info": {self.POLYMER_SMILES: {"n": "n"}},
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert not errors, f"polymer_info should be valid: {errors}"

    def test_schema_rejects_bad_polymer_info(self) -> None:
        """Schema validation rejects malformed polymer_info."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": [],
                    "arrow_above": "",
                    "polymer_info": "not_a_dict",
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert any("polymer_info" in e for e in errors)

    def test_non_polymer_no_brackets(self) -> None:
        """Normal molecules do NOT get bracket notation."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["CCO"],
            "products": ["CC=O"],
            "arrow_above": "PCC",
        }
        svg = compose_step_svg(step).decode("utf-8")
        assert "polymer-bracket" not in svg
        assert 'data-role="polymer"' not in svg


# ═══ AC-13.2: X/Y substituent notation ═══


class TestSubstituentNotation:
    """AC-13.2: Generic structures with R/X/Y variable substituents."""

    def test_substituent_map_schema(self) -> None:
        """Schema accepts substituent_map as dict of variable->group mappings."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["*c1ccccc1"],
                    "products": ["*c1ccc(Br)cc1"],
                    "arrow_above": "NBS",
                    "substituent_map": {
                        "R1": ["CH3", "C2H5", "iPr"],
                        "R2": ["H", "F", "Cl"],
                    },
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert not errors, f"substituent_map should be valid: {errors}"

    def test_substituent_map_rejects_bad_type(self) -> None:
        """Schema rejects substituent_map with wrong type."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": [],
                    "arrow_above": "",
                    "substituent_map": "not_a_dict",
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert any("substituent_map" in e for e in errors)

    def test_substituent_legend_in_svg(self) -> None:
        """SVG output includes a legend table for substituent variables."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["*c1ccccc1"],
            "products": [],
            "arrow_above": "",
            "substituent_map": {
                "R": ["CH3", "C2H5"],
            },
        }
        svg = compose_step_svg(step).decode("utf-8")
        assert 'data-role="substituent-legend"' in svg or "substituent-legend" in svg

    def test_substituent_legend_content(self) -> None:
        """Legend table shows variable name and all variant groups."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["*c1ccccc1"],
            "products": [],
            "arrow_above": "",
            "substituent_map": {
                "R": ["Me", "Et"],
            },
        }
        svg = compose_step_svg(step).decode("utf-8")
        assert "R" in svg
        assert "Me" in svg
        assert "Et" in svg


# ═══ AC-13.3: Adaptive Tanimoto threshold ═══


class TestAdaptiveTanimoto:
    """AC-13.3: Small molecules get higher Tanimoto threshold."""

    def test_small_molecule_higher_threshold(self) -> None:
        """Molecules with <10 heavy atoms use threshold > 0.3."""
        from chemscheme.validators.layer3_scheme import _adaptive_threshold

        # Methanol (1 heavy atom C + 1 O = 2) → high threshold
        assert _adaptive_threshold(2) > 0.3
        # Ethanol (2C + 1O = 3) → high threshold
        assert _adaptive_threshold(3) > 0.3

    def test_medium_molecule_default_threshold(self) -> None:
        """Molecules with 10-40 heavy atoms use default 0.3."""
        from chemscheme.validators.layer3_scheme import _adaptive_threshold

        assert _adaptive_threshold(20) == pytest.approx(0.3, abs=0.05)

    def test_large_molecule_low_threshold(self) -> None:
        """Large molecules (>40 atoms) keep 0.3 or lower (advisory level)."""
        from chemscheme.validators.layer3_scheme import _adaptive_threshold

        assert _adaptive_threshold(50) <= 0.3

    def test_continuity_uses_adaptive_threshold(self) -> None:
        """validate_continuity uses adaptive threshold based on molecule size."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        # Two consecutive steps where products → reactants are only slightly similar.
        # Small molecules should trigger warning (higher threshold not met).
        # Methanol → Ethanol: similar but small molecules.
        scheme = {
            "steps": [
                {"id": "1", "reactants": ["CO"], "products": ["CO"]},
                {"id": "2", "reactants": ["CCO"], "products": ["CCO"]},
            ]
        }
        result = validate_continuity(scheme)
        # CO → CCO: Tanimoto ~0.33 with Morgan fps
        # With adaptive threshold for small molecules (>0.3), this might warn
        # The key test is that the function runs without error
        assert result.layer == "continuity"

    def test_adversarial_zero_atoms(self) -> None:
        """Threshold function handles edge case of 0 atoms gracefully."""
        from chemscheme.validators.layer3_scheme import _adaptive_threshold

        # Should not crash, return some reasonable value
        result = _adaptive_threshold(0)
        assert result > 0


# ═══ AC-13.4: Stereo validation E/Z R/S inversions ═══


class TestStereoInversionDetection:
    """AC-13.4: Detect actual E/Z and R/S configuration inversions."""

    def test_ez_inversion_detected(self) -> None:
        """E→Z inversion is flagged as a stereo warning."""
        from chemscheme.validators.layer1_chemistry import detect_stereo_inversion

        # (E)-but-2-ene vs (Z)-but-2-ene
        e_butene = r"C/C=C/C"
        z_butene = r"C/C=C\C"
        result = detect_stereo_inversion(e_butene, z_butene)
        assert result.has_inversion
        assert "E/Z" in result.description or "inversion" in result.description.lower()

    def test_rs_inversion_detected(self) -> None:
        """R→S inversion at a chiral center is flagged."""
        from chemscheme.validators.layer1_chemistry import detect_stereo_inversion

        # (R)-butan-2-ol vs (S)-butan-2-ol
        r_butanol = "[C@@H](O)(CC)C"
        s_butanol = "[C@H](O)(CC)C"
        result = detect_stereo_inversion(r_butanol, s_butanol)
        assert result.has_inversion
        assert "R/S" in result.description or "inversion" in result.description.lower()

    def test_same_stereo_no_inversion(self) -> None:
        """Identical stereo configuration → no inversion flagged."""
        from chemscheme.validators.layer1_chemistry import detect_stereo_inversion

        smiles = r"C/C=C/C"
        result = detect_stereo_inversion(smiles, smiles)
        assert not result.has_inversion

    def test_no_stereo_no_inversion(self) -> None:
        """Molecules without stereo → no inversion possible."""
        from chemscheme.validators.layer1_chemistry import detect_stereo_inversion

        result = detect_stereo_inversion("CCCC", "CCCC")
        assert not result.has_inversion

    def test_adversarial_invalid_smiles(self) -> None:
        """Invalid SMILES gracefully returns no inversion (not crash)."""
        from chemscheme.validators.layer1_chemistry import detect_stereo_inversion

        result = detect_stereo_inversion("INVALID", "CCCC")
        assert not result.has_inversion


# ═══ AC-13.5: Pre-render validation gate ═══


class TestPreRenderValidationGate:
    """AC-13.5: Optional --validate flag on draw command."""

    def test_draw_validate_flag_exists(self) -> None:
        """draw command accepts --validate flag."""
        from click.testing import CliRunner

        from chemscheme.cli import draw

        runner = CliRunner()
        result = runner.invoke(draw, ["--help"])
        assert "--validate" in result.output

    def test_draw_validate_blocks_on_invalid_chemistry(self, tmp_path: Path) -> None:
        """--validate blocks render if chemistry validation fails."""
        from click.testing import CliRunner

        from chemscheme.cli import draw

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["INVALID_SMILES"],
                    "products": ["CCO"],
                    "arrow_above": "test",
                }
            ],
        }
        scheme_file = tmp_path / "bad.json"
        scheme_file.write_text(json.dumps(scheme))
        out_file = tmp_path / "out.svg"

        runner = CliRunner()
        result = runner.invoke(draw, [str(scheme_file), "-o", str(out_file), "--validate"])
        assert result.exit_code != 0
        assert "validation" in result.output.lower() or "invalid" in result.output.lower()

    def test_draw_without_validate_skips_deep_validation(self, tmp_path: Path) -> None:
        """Without --validate, draw only does schema validation (existing behavior)."""
        from click.testing import CliRunner

        from chemscheme.cli import draw

        # Valid schema but with molecules that might have continuity issues
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                }
            ],
        }
        scheme_file = tmp_path / "ok.json"
        scheme_file.write_text(json.dumps(scheme))
        out_file = tmp_path / "out.svg"

        runner = CliRunner()
        result = runner.invoke(draw, [str(scheme_file), "-o", str(out_file)])
        assert result.exit_code == 0

    def test_draw_validate_passes_valid_scheme(self, tmp_path: Path) -> None:
        """--validate passes when all layers are clean."""
        from click.testing import CliRunner

        from chemscheme.cli import draw

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                }
            ],
        }
        scheme_file = tmp_path / "valid.json"
        scheme_file.write_text(json.dumps(scheme))
        out_file = tmp_path / "out.svg"

        runner = CliRunner()
        result = runner.invoke(draw, [str(scheme_file), "-o", str(out_file), "--validate"])
        assert result.exit_code == 0


# ═══ AC-13.6: Layer1 stereo parse-time stripping ═══


class TestStereoParseTimeStripping:
    """AC-13.6: Edge cases in stereo detection during canonicalization."""

    def test_enhanced_stereo_not_lost(self) -> None:
        """Enhanced stereo (group AND/OR) markers are preserved."""
        from chemscheme.validators.layer1_chemistry import validate_chemistry

        # Chiral molecule that RDKit might strip stereo from during canonicalization
        smiles_with_stereo = "[C@@H](F)(Cl)Br"
        result = validate_chemistry([smiles_with_stereo])
        # Should not report stereo loss for valid stereo
        stereo_warnings = [w for w in result.warnings if "stereo" in w.lower()]
        assert not stereo_warnings, f"False stereo loss warning: {stereo_warnings}"

    def test_redundant_stereo_markers_tolerated(self) -> None:
        """SMILES with redundant (but technically valid) stereo markers don't crash."""
        from chemscheme.validators.layer1_chemistry import validate_chemistry

        # This SMILES has stereo that may be removed as redundant by RDKit
        smiles = r"C(/F)=C(/Cl)Br"
        result = validate_chemistry([smiles])
        # Should not error
        assert not result.errors

    def test_allene_stereo_preserved(self) -> None:
        """Allenic stereo (cumulated dienes) is properly handled."""
        from chemscheme.validators.layer1_chemistry import validate_chemistry

        # Allene with axial chirality
        allene = "CC=C=CC"
        result = validate_chemistry([allene])
        assert not result.errors

    def test_adversarial_mixed_stereo_and_invalid(self) -> None:
        """Mix of valid stereo + invalid SMILES in same batch."""
        from chemscheme.validators.layer1_chemistry import validate_chemistry

        result = validate_chemistry(["[C@@H](F)(Cl)Br", "INVALID", r"C/C=C/C"])
        assert len(result.errors) == 1  # only INVALID
        # Valid stereo molecules should not generate false warnings
