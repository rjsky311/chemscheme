"""CLI integration tests for edit, export, and import commands.

Phase 7 — AC-7.1, AC-7.2, AC-7.3, AC-7.4, AC-7.5.
"""

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from chemscheme.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def tmp_scheme(tmp_path: Path) -> Path:
    """Copy sample_scheme.json to tmp_path for editing."""
    src = FIXTURES_DIR / "sample_scheme.json"
    dst = tmp_path / "scheme.json"
    shutil.copy(src, dst)
    return dst


@pytest.fixture
def tmp_scheme_3step(tmp_path: Path) -> Path:
    """Copy 3-step scheme to tmp_path for editing."""
    src = FIXTURES_DIR / "sample_scheme_3step.json"
    dst = tmp_path / "scheme_3step.json"
    shutil.copy(src, dst)
    return dst


@pytest.mark.integration
class TestCliEdit:
    """CLI edit command tests."""

    def test_edit_add_step(self, runner: CliRunner, tmp_scheme: Path) -> None:
        """AC-7.1: chemscheme edit --add-step adds a step."""
        new_step = json.dumps(
            {
                "id": "2",
                "reactants": ["c1ccc(-c2ccccc2)cc1"],
                "products": ["C1CCC(C2CCCCC2)CC1"],
                "arrow_above": "H2, Pd/C",
            }
        )
        result = runner.invoke(main, ["edit", str(tmp_scheme), "--add-step", new_step])
        assert result.exit_code == 0
        assert "Added step" in result.output

        # Verify file was modified
        data = json.loads(tmp_scheme.read_text(encoding="utf-8"))
        assert len(data["steps"]) == 2

    def test_edit_remove_step(self, runner: CliRunner, tmp_scheme_3step: Path) -> None:
        """AC-7.2: chemscheme edit --remove-step removes a step."""
        result = runner.invoke(main, ["edit", str(tmp_scheme_3step), "--remove-step", "2"])
        assert result.exit_code == 0
        assert "Removed step" in result.output

        data = json.loads(tmp_scheme_3step.read_text(encoding="utf-8"))
        assert len(data["steps"]) == 2
        ids = [s["id"] for s in data["steps"]]
        assert "2" not in ids

    def test_edit_add_step_json_output(self, runner: CliRunner, tmp_scheme: Path) -> None:
        """--json flag outputs JSON with validation result."""
        new_step = json.dumps(
            {
                "id": "2",
                "reactants": ["C"],
                "products": ["CC"],
                "arrow_above": "test",
            }
        )
        result = runner.invoke(main, ["edit", str(tmp_scheme), "--add-step", new_step, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert "validation" in data

    def test_edit_remove_nonexistent_step_fails(self, runner: CliRunner, tmp_scheme: Path) -> None:
        """Removing nonexistent step exits with error."""
        result = runner.invoke(main, ["edit", str(tmp_scheme), "--remove-step", "999"])
        assert result.exit_code != 0

    def test_edit_no_operation_fails(self, runner: CliRunner, tmp_scheme: Path) -> None:
        """Must specify --add-step or --remove-step."""
        result = runner.invoke(main, ["edit", str(tmp_scheme)])
        assert result.exit_code != 0

    def test_edit_add_step_with_position(self, runner: CliRunner, tmp_scheme_3step: Path) -> None:
        """AC-7.1: --position inserts at correct index."""
        new_step = json.dumps(
            {
                "id": "1b",
                "reactants": ["C"],
                "products": ["CC"],
                "arrow_above": "test",
            }
        )
        result = runner.invoke(
            main,
            ["edit", str(tmp_scheme_3step), "--add-step", new_step, "--position", "1"],
        )
        assert result.exit_code == 0
        data = json.loads(tmp_scheme_3step.read_text(encoding="utf-8"))
        assert data["steps"][1]["id"] == "1b"


@pytest.mark.integration
class TestCliExport:
    """CLI export command tests."""

    def test_export_cdxml(self, runner: CliRunner, tmp_scheme: Path, tmp_path: Path) -> None:
        """AC-7.4: chemscheme export produces a .cdxml file."""
        out = tmp_path / "output.cdxml"
        result = runner.invoke(main, ["export", str(tmp_scheme), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<CDXML" in content

    def test_export_json_output(self, runner: CliRunner, tmp_scheme: Path, tmp_path: Path) -> None:
        """--json flag returns structured output."""
        out = tmp_path / "output.cdxml"
        result = runner.invoke(main, ["export", str(tmp_scheme), "-o", str(out), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert data["format"] == "cdxml"

    def test_export_overwrite_protection(
        self, runner: CliRunner, tmp_scheme: Path, tmp_path: Path
    ) -> None:
        """Export refuses to overwrite without --force."""
        out = tmp_path / "output.cdxml"
        out.write_text("existing", encoding="utf-8")
        result = runner.invoke(main, ["export", str(tmp_scheme), "-o", str(out)])
        assert result.exit_code != 0

    def test_export_force_overwrite(
        self, runner: CliRunner, tmp_scheme: Path, tmp_path: Path
    ) -> None:
        """--force allows overwriting."""
        out = tmp_path / "output.cdxml"
        out.write_text("existing", encoding="utf-8")
        result = runner.invoke(main, ["export", str(tmp_scheme), "-o", str(out), "--force"])
        assert result.exit_code == 0


@pytest.mark.integration
class TestCliImport:
    """CLI import command tests."""

    def test_import_cdxml(self, runner: CliRunner, tmp_scheme: Path, tmp_path: Path) -> None:
        """AC-7.5: chemscheme import converts .cdxml to scheme.json."""
        # First export
        cdxml_out = tmp_path / "test.cdxml"
        runner.invoke(main, ["export", str(tmp_scheme), "-o", str(cdxml_out)])
        assert cdxml_out.exists()

        # Then import
        scheme_out = tmp_path / "imported.json"
        result = runner.invoke(main, ["import", str(cdxml_out), "-o", str(scheme_out)])
        assert result.exit_code == 0
        assert scheme_out.exists()

        data = json.loads(scheme_out.read_text(encoding="utf-8"))
        assert "steps" in data
        assert len(data["steps"]) > 0

    def test_import_json_output(self, runner: CliRunner, tmp_scheme: Path, tmp_path: Path) -> None:
        """--json flag returns structured output."""
        cdxml_out = tmp_path / "test.cdxml"
        runner.invoke(main, ["export", str(tmp_scheme), "-o", str(cdxml_out)])

        scheme_out = tmp_path / "imported.json"
        result = runner.invoke(main, ["import", str(cdxml_out), "-o", str(scheme_out), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert data["steps"] >= 1
