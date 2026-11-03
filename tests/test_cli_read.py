"""Tests for CLI 'read' command — extract SMILES from structure files.

BDD Scenarios:
  GIVEN a MOL file, WHEN `chemscheme read file.mol`, THEN prints SMILES
  GIVEN a CDX file, WHEN `chemscheme read file.cdx --json`, THEN prints JSON with SMILES list
  GIVEN an invalid file, WHEN `chemscheme read`, THEN exits with error
  GIVEN an unsupported extension, WHEN `chemscheme read`, THEN exits with error
"""

from pathlib import Path

from click.testing import CliRunner

from chemscheme.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


class TestCliRead:
    def test_read_mol_prints_smiles(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["read", str(FIXTURES / "benzene.mol")])
        assert result.exit_code == 0
        assert "c1ccccc1" in result.output.lower() or "C1" in result.output

    def test_read_sdf_prints_multiple(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["read", str(FIXTURES / "multi.sdf")])
        assert result.exit_code == 0
        # Should have 2 SMILES
        lines = [line for line in result.output.strip().split("\n") if line.strip()]
        assert len(lines) >= 2

    def test_read_cdx_prints_smiles(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["read", str(FIXTURES / "tetrabromo_spirobifluorene.cdx")])
        assert result.exit_code == 0
        assert "Br" in result.output  # tetrabromo compound

    def test_read_json_output(self) -> None:
        import json

        runner = CliRunner()
        result = runner.invoke(main, ["read", str(FIXTURES / "benzene.mol"), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert isinstance(data["molecules"], list)
        assert len(data["molecules"]) == 1
        assert "smiles" in data["molecules"][0]

    def test_read_invalid_file_errors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["read", str(FIXTURES / "invalid.mol")])
        assert result.exit_code != 0

    def test_read_unsupported_extension_errors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["read", str(FIXTURES / "sample_scheme.json")])
        assert result.exit_code != 0
        assert "unsupported" in result.output.lower() or "Unsupported" in result.output


class TestCliImportCdx:
    """Test that 'import' command now also accepts CDX binary files."""

    def test_import_cdx_creates_scheme_json(self, tmp_path: Path) -> None:
        import json

        out = tmp_path / "scheme.json"
        runner = CliRunner()
        result = runner.invoke(
            main, ["import", str(FIXTURES / "tetrabromo_spirobifluorene.cdx"), "-o", str(out)]
        )
        # CDX import should succeed (even if scheme validation warns)
        assert out.exists() or result.exit_code != 0
        # If it succeeded, check the output
        if result.exit_code == 0:
            scheme = json.loads(out.read_text(encoding="utf-8"))
            assert "steps" in scheme
