"""User-extensible library tests.

BDD-first tests for ~/.chemscheme/custom_reagents.json and custom_reactions.json.
Users can add/remove/list custom reagents and reactions that merge with built-in data.

Covers:
  - Data layer: add/remove/list, merge precedence, SMILES validation
  - CLI: library add/remove/list commands
  - MCP: add/remove/list tools
  - Integration: custom reagents appear in lookup/find_compounds
  - Adversarial: corrupt JSON, missing dir, unicode, invalid SMILES
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def user_data_dir(tmp_path: Path) -> Path:
    """Provide a temp dir as ~/.chemscheme/ for testing."""
    d = tmp_path / ".chemscheme"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _isolate_user_dir(user_data_dir: Path) -> None:  # type: ignore[type-arg]
    """Patch _get_user_data_dir to use temp dir and clear caches."""
    import chemscheme.domain_library as dl

    with patch.object(dl, "_get_user_data_dir", return_value=user_data_dir):
        # Clear caches so each test starts fresh
        dl._reagent_cache = None  # type: ignore[attr-defined]
        dl._reaction_cache = None  # type: ignore[attr-defined]
        yield
        dl._reagent_cache = None  # type: ignore[attr-defined]
        dl._reaction_cache = None  # type: ignore[attr-defined]


# ===========================================================================
# Data layer: add/remove/list
# ===========================================================================


class TestUserReagentAdd:
    """Adding custom reagents."""

    def test_add_user_reagent_basic(self, user_data_dir: Path) -> None:
        """Add a custom reagent with name and category."""
        from chemscheme.domain_library import add_user_reagent

        result = add_user_reagent("MyNFA", "Custom NFA Acceptor", "nfa_acceptor")
        assert result.abbreviation == "MyNFA"
        assert result.name == "Custom NFA Acceptor"
        assert result.category == "nfa_acceptor"

    def test_add_user_reagent_with_smiles(self, user_data_dir: Path) -> None:
        """Add a custom reagent with SMILES (canonicalized)."""
        from chemscheme.domain_library import add_user_reagent

        result = add_user_reagent("TestR", "Test Reagent", "reagent", smiles="c1ccccc1")
        assert result.smiles is not None
        assert result.smiles != ""  # Should be canonicalized

    def test_add_user_reagent_persisted(self, user_data_dir: Path) -> None:
        """Added reagent is saved to custom_reagents.json."""
        from chemscheme.domain_library import add_user_reagent

        add_user_reagent("PersistR", "Persistent", "catalyst")
        json_file = user_data_dir / "custom_reagents.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert "PersistR" in data["abbreviations"]

    def test_add_invalid_smiles_rejected(self, user_data_dir: Path) -> None:
        """Invalid SMILES should raise ValueError."""
        from chemscheme.domain_library import add_user_reagent

        with pytest.raises(ValueError, match="[Ii]nvalid SMILES"):
            add_user_reagent("BadR", "Bad Reagent", "reagent", smiles="not_a_smiles!!!")


class TestUserReagentRemove:
    """Removing custom reagents."""

    def test_remove_existing_reagent(self, user_data_dir: Path) -> None:
        """Remove a previously added custom reagent."""
        from chemscheme.domain_library import add_user_reagent, remove_user_reagent

        add_user_reagent("TempR", "Temporary", "reagent")
        result = remove_user_reagent("TempR")
        assert result is True

    def test_remove_nonexistent_returns_false(self, user_data_dir: Path) -> None:
        """Removing a reagent that doesn't exist returns False."""
        from chemscheme.domain_library import remove_user_reagent

        result = remove_user_reagent("DoesNotExist")
        assert result is False

    def test_cannot_remove_builtin(self, user_data_dir: Path) -> None:
        """Cannot remove built-in reagents (only custom ones)."""
        from chemscheme.domain_library import remove_user_reagent

        # NBS is a built-in reagent
        result = remove_user_reagent("NBS")
        assert result is False


class TestUserReagentList:
    """Listing custom reagents."""

    def test_list_empty_initially(self, user_data_dir: Path) -> None:
        """No custom reagents initially."""
        from chemscheme.domain_library import list_user_reagents

        result = list_user_reagents()
        assert result == []

    def test_list_after_add(self, user_data_dir: Path) -> None:
        """Custom reagent appears after add."""
        from chemscheme.domain_library import add_user_reagent, list_user_reagents

        add_user_reagent("ListR", "Listed Reagent", "catalyst")
        result = list_user_reagents()
        assert len(result) == 1
        assert result[0].abbreviation == "ListR"


class TestMergePrecedence:
    """User entries override built-in on conflict."""

    def test_user_overrides_builtin(self, user_data_dir: Path) -> None:
        """User reagent with same abbreviation as built-in takes precedence."""
        from chemscheme.domain_library import add_user_reagent, lookup_reagent

        # NBS is built-in; add a user version with different category
        add_user_reagent("NBS", "Custom NBS", "custom_cat")
        results = lookup_reagent("NBS")
        assert len(results) >= 1
        # First result should be user's version
        assert results[0].category == "custom_cat"

    def test_builtin_still_available_after_user_remove(self, user_data_dir: Path) -> None:
        """After removing user override, built-in NBS is restored."""
        from chemscheme.domain_library import (
            add_user_reagent,
            lookup_reagent,
            remove_user_reagent,
        )

        add_user_reagent("NBS", "Custom NBS", "custom_cat")
        remove_user_reagent("NBS")
        results = lookup_reagent("NBS")
        assert len(results) >= 1
        # Should be back to built-in
        assert results[0].category != "custom_cat"


# ===========================================================================
# Adversarial / edge cases
# ===========================================================================


class TestAdversarial:
    """Edge cases and error recovery."""

    def test_missing_dir_auto_created(self, tmp_path: Path) -> None:
        """If ~/.chemscheme/ doesn't exist, it's created automatically."""
        from chemscheme.domain_library import _get_user_data_dir

        with patch("chemscheme.domain_library.Path.home", return_value=tmp_path):
            d = _get_user_data_dir()
            assert d.exists()

    def test_corrupt_json_recovery(self, user_data_dir: Path) -> None:
        """Corrupt custom_reagents.json doesn't crash, returns empty."""
        from chemscheme.domain_library import list_user_reagents

        (user_data_dir / "custom_reagents.json").write_text("{bad json!!!")
        result = list_user_reagents()
        assert result == []

    def test_unicode_reagent_name(self, user_data_dir: Path) -> None:
        """Unicode names (e.g., Chinese) should work."""
        from chemscheme.domain_library import add_user_reagent

        result = add_user_reagent("甲醇", "Methanol (Chinese)", "solvent", smiles="CO")
        assert result.abbreviation == "甲醇"

    def test_empty_abbreviation_rejected(self, user_data_dir: Path) -> None:
        """Empty abbreviation should raise ValueError."""
        from chemscheme.domain_library import add_user_reagent

        with pytest.raises(ValueError, match="[Aa]bbreviation"):
            add_user_reagent("", "Empty", "reagent")


# ===========================================================================
# CLI commands
# ===========================================================================


class TestCLILibrary:
    """CLI library command group."""

    def test_cli_library_list_help(self) -> None:
        """'chemscheme library list --help' works."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["library", "list", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower()

    def test_cli_library_add_and_list(self, user_data_dir: Path) -> None:
        """Add a reagent via CLI and list it."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        # Add
        result = runner.invoke(
            main,
            ["library", "add", "CLITest", "--name", "CLI Test Reagent", "--category", "catalyst"],
        )
        assert result.exit_code == 0

        # List custom
        result = runner.invoke(main, ["library", "list", "--custom"])
        assert result.exit_code == 0
        assert "CLITest" in result.output

    def test_cli_library_remove(self, user_data_dir: Path) -> None:
        """Remove a reagent via CLI."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        runner.invoke(
            main,
            ["library", "add", "RemoveMe", "--name", "To Remove", "--category", "reagent"],
        )
        result = runner.invoke(main, ["library", "remove", "RemoveMe"])
        assert result.exit_code == 0

    def test_cli_library_add_json_output(self, user_data_dir: Path) -> None:
        """Add with --json returns JSON."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["library", "add", "JsonR", "--name", "JSON Reagent", "--category", "base", "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["abbreviation"] == "JsonR"


# ===========================================================================
# MCP tools
# ===========================================================================


class TestMCPUserLibrary:
    """MCP tools for user library management."""

    def test_mcp_add_user_reagent_registered(self) -> None:
        """MCP tool 'add_user_reagent' is registered."""
        from chemscheme.mcp_server import chemscheme_mcp

        tool_names = [t.name for t in chemscheme_mcp._tool_manager.list_tools()]
        assert "add_user_reagent" in tool_names

    def test_mcp_remove_user_reagent_registered(self) -> None:
        """MCP tool 'remove_user_reagent' is registered."""
        from chemscheme.mcp_server import chemscheme_mcp

        tool_names = [t.name for t in chemscheme_mcp._tool_manager.list_tools()]
        assert "remove_user_reagent" in tool_names

    def test_mcp_list_user_reagents_registered(self) -> None:
        """MCP tool 'list_user_reagents' is registered."""
        from chemscheme.mcp_server import chemscheme_mcp

        tool_names = [t.name for t in chemscheme_mcp._tool_manager.list_tools()]
        assert "list_user_reagents" in tool_names


# ===========================================================================
# Integration: custom reagent appears in existing lookup/text_extraction
# ===========================================================================


class TestIntegration:
    """Custom reagents flow through to lookup and text extraction."""

    def test_custom_reagent_in_lookup(self, user_data_dir: Path) -> None:
        """Custom reagent appears in lookup_reagent results."""
        from chemscheme.domain_library import add_user_reagent, lookup_reagent

        add_user_reagent("XYZ123", "Custom Compound", "catalyst", smiles="CC")
        results = lookup_reagent("XYZ123")
        assert len(results) >= 1
        assert results[0].abbreviation == "XYZ123"

    def test_custom_reagent_in_text_extraction(self, user_data_dir: Path) -> None:
        """Custom reagent is found by find_compounds in text."""
        from chemscheme.domain_library import add_user_reagent

        add_user_reagent("MyCustomCat", "My Custom Catalyst", "catalyst", smiles="[Pd]")

        # Clear text_extraction caches
        import chemscheme.text_extraction as te

        te._reagent_abbrevs = None  # type: ignore[attr-defined]

        from chemscheme.text_extraction import find_compounds

        compounds = find_compounds("reacted with MyCustomCat in THF")
        names = {c.name for c in compounds}
        assert "MyCustomCat" in names
