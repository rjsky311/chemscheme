"""Tests for integrated validation with all 3 layers (AC-6.5)."""

import json

import pytest
from click.testing import CliRunner

from chemscheme.cli import main


# ---------------------------------------------------------------------------
# AC-6.5  chemscheme validate integrates all three layers
# ---------------------------------------------------------------------------
class TestIntegratedValidation:
    """AC-6.5: validate command outputs layered report."""

    @pytest.mark.integration
    def test_validate_json_includes_all_layers(self, fixtures_dir) -> None:
        """--json output lists all checked layers."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["validate", "--json", str(fixtures_dir / "sample_scheme.json")],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "layers_checked" in data
        # Should check at least schema + chemistry + continuity
        assert "schema" in data["layers_checked"]
        assert "chemistry" in data["layers_checked"]
        assert "continuity" in data["layers_checked"]

    @pytest.mark.integration
    def test_validate_reports_chemistry_errors(self, fixtures_dir, tmp_path) -> None:
        """Scheme with invalid SMILES shows chemistry layer errors."""
        bad_scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["INVALID_SMILES"],
                    "products": ["c1ccccc1"],
                    "arrow_above": "magic",
                }
            ],
        }
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps(bad_scheme))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--json", str(bad_file)])
        data = json.loads(result.output)
        assert data["valid"] is False
        assert any("INVALID_SMILES" in issue for issue in data["issues"])

    @pytest.mark.integration
    def test_validate_text_output_mentions_layers(self, fixtures_dir) -> None:
        """Text output mentions validation layers."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["validate", str(fixtures_dir / "sample_scheme.json")],
        )
        assert result.exit_code == 0

    @pytest.mark.integration
    def test_validate_continuity_warnings_in_report(self, tmp_path) -> None:
        """Continuity warnings appear in JSON report."""
        discontinuous = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "H2O2",
                },
                {
                    "id": "2",
                    "reactants": ["CCCC"],
                    "products": ["CCCCO"],
                    "arrow_above": "O2",
                },
            ],
        }
        scheme_file = tmp_path / "disc.json"
        scheme_file.write_text(json.dumps(discontinuous))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--json", str(scheme_file)])
        data = json.loads(result.output)
        # Should have warnings (advisory) but still be valid
        assert "warnings" in data
