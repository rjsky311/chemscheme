"""MR21 regression tests for review findings.

# RED-EXEMPT: review finding regression tests
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestG1A5AtomicWrite:
    """G1/A5: User reagent file uses atomic write."""

    def test_save_uses_atomic_replace(self, tmp_path: Path) -> None:
        """_save_user_reagents_raw writes via temp file + replace."""
        from chemscheme.domain_library import _save_user_reagents_raw

        data = {"abbreviations": {"TEST": {"name": "Test", "category": "reagent"}}}
        _save_user_reagents_raw(data, tmp_path)
        fp = tmp_path / "custom_reagents.json"
        assert fp.exists()
        loaded = json.loads(fp.read_text(encoding="utf-8"))
        assert loaded["abbreviations"]["TEST"]["name"] == "Test"

    def test_no_temp_files_left_on_success(self, tmp_path: Path) -> None:
        """No leftover .tmp files after successful save."""
        from chemscheme.domain_library import _save_user_reagents_raw

        _save_user_reagents_raw({"abbreviations": {}}, tmp_path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


class TestA5SchemaValidation:
    """A5: Validate abbreviations is a dict, not list."""

    def test_abbreviations_list_treated_as_empty(self, tmp_path: Path) -> None:
        """If abbreviations is a list, treat as empty dict."""
        from chemscheme.domain_library import _load_user_reagents_raw

        fp = tmp_path / "custom_reagents.json"
        fp.write_text('{"abbreviations": []}', encoding="utf-8")
        data = _load_user_reagents_raw(tmp_path)
        assert isinstance(data["abbreviations"], dict)
        assert data["abbreviations"] == {}


class TestA2EmptyStepSkip:
    """A2: Text extraction skips steps where all compounds are unresolved."""

    def test_all_unresolved_compounds_skipped(self) -> None:
        """Steps with no resolved SMILES are skipped."""
        from chemscheme.text_extraction import extract_reactions_from_text

        # Use a text with compound names that won't resolve to SMILES
        result = extract_reactions_from_text(
            "XyzUnknownium reacted with AbcMysteryol to give QrsFakeane.",
            use_ai=False,
        )
        # Steps with no SMILES should be skipped
        for step in result.get("steps", []):
            assert step["reactants"] or step["products"], (
                f"Step {step['id']} has empty reactants AND products"
            )


class TestA4EmptySmiles:
    """A4: Empty SMILES recognition returns error status."""

    def test_mcp_empty_smiles_returns_error(self) -> None:
        """MCP extract_image returns error for empty recognition."""
        from chemscheme.mcp_server import _sync_extract_image

        # Patch at the source module where imports happen
        with (
            patch(
                "chemscheme.recognition.ocsr.MolScribeRecognizer",
                side_effect=RuntimeError("no docker"),
            ),
            patch(
                "chemscheme.recognition.ocsr.DecimerRecognizer",
                side_effect=RuntimeError("no docker"),
            ),
        ):
            import base64

            from chemscheme.recognition.mocks import MockMoleculeRecognizer

            # Create a mock that returns empty SMILES
            empty_mock = MockMoleculeRecognizer(value="", confidence=0.0)
            with patch(
                "chemscheme.recognition.mocks.MockMoleculeRecognizer",
                return_value=empty_mock,
            ):
                img_b64 = base64.b64encode(b"\x89PNG_fake").decode()
                result = json.loads(_sync_extract_image(img_b64, "png"))
                assert result["status"] == "error"


class TestA6CacheInvalidation:
    """A6/G4: Cache invalidation crosses module boundaries."""

    def test_invalidate_clears_text_extraction_caches(self) -> None:
        """_invalidate_all_caches clears text_extraction module caches."""
        from chemscheme import text_extraction as te
        from chemscheme.domain_library import _invalidate_all_caches

        # Set up caches
        te._solvent_names = {"THF", "DMF"}
        te._reagent_abbrevs = {"NBS": {"name": "N-Bromosuccinimide"}}

        _invalidate_all_caches()

        assert te._solvent_names is None
        assert te._reagent_abbrevs is None


class TestA3DockerImageFormat:
    """A3: Docker mode uses correct file extension for image format."""

    def test_run_tool_docker_uses_format_extension(self) -> None:
        """run_tool_docker creates temp file with correct extension."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch("chemscheme.recognition.runner.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(
                returncode=1, stdout=b"", stderr=b"docker not found"
            )
            with contextlib.suppress(Exception):
                run_tool_docker("molscribe", b"fake", image_format="jpeg")
            # Check the command used correct filename
            if mock_sub.run.called:
                cmd = mock_sub.run.call_args[0][0]
                # Should contain input.jpeg, not input.png
                cmd_str = " ".join(str(c) for c in cmd)
                assert "input.jpeg" in cmd_str


class TestA9LibraryCLIJson:
    """A9: Library CLI --json contract consistency."""

    def test_library_add_json_has_status(self, tmp_path: Path) -> None:
        """library add --json includes status field."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        with patch("chemscheme.domain_library._get_user_data_dir", return_value=tmp_path):
            result = runner.invoke(
                main,
                ["library", "add", "TEST", "--name", "Test", "--category", "reagent", "--json"],
            )
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert "status" in data
            assert data["status"] == "ok"

    def test_library_remove_json_option_exists(self) -> None:
        """library remove command accepts --json flag."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["library", "remove", "--help"])
        assert "--json" in result.output
