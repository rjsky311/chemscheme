"""Phase 4 CLI tests — chemscheme draw + validate commands.

Covers AC-4.3 through AC-4.11.
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from chemscheme.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# AC-4.3: chemscheme draw scheme.json -o output.svg
# ---------------------------------------------------------------------------


class TestDrawCommand:
    """chemscheme draw produces SVG/PNG output from scheme.json."""

    @pytest.mark.integration
    def test_draw_svg(self, tmp_path: Path) -> None:
        """draw produces a valid SVG file."""
        scheme_path = FIXTURES_DIR / "sample_scheme.json"
        out_path = tmp_path / "output.svg"

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(scheme_path), "-o", str(out_path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out_path.exists()
        content = out_path.read_bytes()
        assert b"<svg" in content

    @pytest.mark.integration
    def test_draw_png(self, tmp_path: Path) -> None:
        """draw with .png output produces a valid PNG file."""
        scheme_path = FIXTURES_DIR / "sample_scheme.json"
        out_path = tmp_path / "output.png"

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(scheme_path), "-o", str(out_path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out_path.exists()
        content = out_path.read_bytes()
        assert content[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.integration
    def test_draw_multi_step(self, tmp_path: Path) -> None:
        """draw renders multi-step scheme."""
        scheme_path = FIXTURES_DIR / "sample_scheme_3step.json"
        out_path = tmp_path / "output.svg"

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(scheme_path), "-o", str(out_path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out_path.exists()
        assert out_path.stat().st_size > 100

    @pytest.mark.integration
    def test_draw_nonexistent_input(self) -> None:
        """draw with nonexistent input file shows error (not 'No such command')."""
        runner = CliRunner()
        result = runner.invoke(main, ["draw", "nonexistent.json", "-o", "out.svg"])

        assert result.exit_code != 0
        # Must be a file-not-found error, not a "No such command" error
        assert "no such command" not in result.output.lower()
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    @pytest.mark.integration
    def test_draw_invalid_scheme(self, tmp_path: Path) -> None:
        """draw with invalid scheme shows validation error (not 'No such command')."""
        bad_scheme = tmp_path / "bad.json"
        bad_scheme.write_text('{"steps": []}', encoding="utf-8")
        out_path = tmp_path / "output.svg"

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(bad_scheme), "-o", str(out_path)])

        assert result.exit_code != 0
        assert "no such command" not in result.output.lower()


# ---------------------------------------------------------------------------
# AC-4.4: chemscheme validate scheme.json
# ---------------------------------------------------------------------------


class TestValidateCommand:
    """chemscheme validate outputs validation results."""

    @pytest.mark.integration
    def test_validate_valid_scheme(self) -> None:
        """validate on valid scheme exits 0 with success message."""
        scheme_path = FIXTURES_DIR / "sample_scheme.json"

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(scheme_path)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    @pytest.mark.integration
    def test_validate_invalid_scheme(self, tmp_path: Path) -> None:
        """validate on invalid scheme exits non-zero (not 'No such command')."""
        bad_scheme = tmp_path / "bad.json"
        bad_scheme.write_text('{"steps": [{"id": "1"}]}', encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(bad_scheme)])

        assert result.exit_code != 0
        assert "no such command" not in result.output.lower()

    @pytest.mark.integration
    def test_validate_nonexistent(self) -> None:
        """validate with nonexistent file shows error (not 'No such command')."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "nonexistent.json"])

        assert result.exit_code != 0
        assert "no such command" not in result.output.lower()


# ---------------------------------------------------------------------------
# AC-4.5: --json structured output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """All commands support --json for structured output."""

    @pytest.mark.unit
    def test_validate_json_output(self) -> None:
        """validate --json outputs JSON with status/valid/issues."""
        scheme_path = FIXTURES_DIR / "sample_scheme.json"

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(scheme_path), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "valid" in data
        assert "issues" in data

    @pytest.mark.unit
    def test_draw_json_output(self, tmp_path: Path) -> None:
        """draw --json outputs JSON with status and output_path."""
        scheme_path = FIXTURES_DIR / "sample_scheme.json"
        out_path = tmp_path / "output.svg"

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(scheme_path), "-o", str(out_path), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "status" in data
        assert "output_path" in data


# ---------------------------------------------------------------------------
# AC-4.6: Paths with spaces
# ---------------------------------------------------------------------------


class TestPathsWithSpaces:
    """Paths containing spaces are handled correctly."""

    @pytest.mark.unit
    def test_draw_path_with_spaces(self, tmp_path: Path) -> None:
        """draw works with spaces in input and output paths."""
        spaced_dir = tmp_path / "my project" / "output files"
        spaced_dir.mkdir(parents=True)
        scheme_path = FIXTURES_DIR / "sample_scheme.json"
        out_path = spaced_dir / "my output.svg"

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(scheme_path), "-o", str(out_path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out_path.exists()

    @pytest.mark.unit
    def test_validate_path_with_spaces(self, tmp_path: Path) -> None:
        """validate works with spaces in path."""
        spaced_dir = tmp_path / "path with spaces"
        spaced_dir.mkdir()
        scheme_copy = spaced_dir / "my scheme.json"
        scheme_copy.write_text(
            (FIXTURES_DIR / "sample_scheme.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(scheme_copy)])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC-4.7: Unicode reagent names
# ---------------------------------------------------------------------------


class TestUnicodeReagents:
    """Unicode characters in reagent names handled correctly."""

    @pytest.mark.unit
    def test_draw_unicode_scheme(self, tmp_path: Path) -> None:
        """draw handles Unicode reagent names (subscripts, CJK)."""
        scheme_path = FIXTURES_DIR / "sample_scheme_unicode.json"
        out_path = tmp_path / "output.svg"

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(scheme_path), "-o", str(out_path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "Pd(PPh₃)₄" in content or "<svg" in content

    @pytest.mark.unit
    def test_validate_unicode_scheme(self) -> None:
        """validate handles Unicode reagent names."""
        scheme_path = FIXTURES_DIR / "sample_scheme_unicode.json"

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(scheme_path)])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC-4.8: Overwrite protection + --force
# ---------------------------------------------------------------------------


class TestOverwriteProtection:
    """Existing files require confirmation or --force to overwrite."""

    @pytest.mark.unit
    def test_draw_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        """draw refuses to overwrite existing file without --force."""
        scheme_path = FIXTURES_DIR / "sample_scheme.json"
        out_path = tmp_path / "output.svg"
        out_path.write_text("existing content", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(scheme_path), "-o", str(out_path)])

        assert result.exit_code != 0
        assert "exist" in result.output.lower() or "overwrite" in result.output.lower()
        # Original content should be preserved
        assert out_path.read_text(encoding="utf-8") == "existing content"

    @pytest.mark.unit
    def test_draw_force_overwrites(self, tmp_path: Path) -> None:
        """draw --force overwrites existing file."""
        scheme_path = FIXTURES_DIR / "sample_scheme.json"
        out_path = tmp_path / "output.svg"
        out_path.write_text("existing content", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(scheme_path), "-o", str(out_path), "--force"])

        assert result.exit_code == 0
        assert out_path.read_text(encoding="utf-8") != "existing content"


# ---------------------------------------------------------------------------
# AC-4.9: Auto-create output directory
# ---------------------------------------------------------------------------


class TestAutoCreateDir:
    """Output directory is created automatically if it doesn't exist."""

    @pytest.mark.unit
    def test_draw_creates_output_dir(self, tmp_path: Path) -> None:
        """draw auto-creates output directory."""
        scheme_path = FIXTURES_DIR / "sample_scheme.json"
        out_path = tmp_path / "new_dir" / "sub_dir" / "output.svg"

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(scheme_path), "-o", str(out_path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out_path.exists()


# ---------------------------------------------------------------------------
# AC-4.11: Installed wheel smoke test
# ---------------------------------------------------------------------------


class TestInstalledWheel:
    """Installed package works correctly."""

    @pytest.mark.smoke
    def test_help_shows_draw_and_validate(self) -> None:
        """--help lists draw and validate commands."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "draw" in result.output
        assert "validate" in result.output
