"""Phase 15 — PyPI Release & CI/CD: package integrity tests.

AC-15.6: Verify package data, theme system, MCP server, and CLI entry point
are all importable and functional after installation.
"""

from __future__ import annotations

import importlib
import json

import pytest


class TestPackageDataImport:
    """AC-15.6: importlib.resources can load package data files."""

    @pytest.mark.unit
    def test_reagent_aliases_loadable(self) -> None:
        """reagent_aliases.json is accessible via importlib.resources."""
        data_files = importlib.resources.files("chemscheme.data")
        aliases_path = data_files.joinpath("reagent_aliases.json")
        content = aliases_path.read_text(encoding="utf-8")
        aliases = json.loads(content)
        assert isinstance(aliases, dict)
        assert len(aliases) > 0

    @pytest.mark.unit
    def test_lab_context_loadable(self) -> None:
        """lab_context.json is accessible via importlib.resources."""
        data_files = importlib.resources.files("chemscheme.data")
        context_path = data_files.joinpath("lab_context.json")
        content = context_path.read_text(encoding="utf-8")
        context = json.loads(content)
        assert isinstance(context, dict)


class TestThemeImport:
    """AC-15.6: Theme system accessible after import."""

    @pytest.mark.unit
    def test_get_theme_importable(self) -> None:
        """get_theme and list_themes are importable."""
        from chemscheme.renderers.themes import get_theme, list_themes

        themes = list_themes()
        assert "default" in themes
        assert "acs" in themes

        theme = get_theme("default")
        assert theme is not None

    @pytest.mark.unit
    def test_all_preset_themes_valid(self) -> None:
        """All listed themes are retrievable without error."""
        from chemscheme.renderers.themes import get_theme, list_themes

        for name in list_themes():
            theme = get_theme(name)
            assert theme.font_size_conditions > 0


class TestMCPServerImportable:
    """AC-15.6: MCP server importable and tools registered."""

    @pytest.mark.unit
    def test_mcp_server_importable(self) -> None:
        """chemscheme.mcp_server module is importable."""
        from chemscheme.mcp_server import chemscheme_mcp

        assert chemscheme_mcp is not None

    @pytest.mark.unit
    def test_mcp_tools_registered(self) -> None:
        """FastMCP instance has expected tools registered."""
        from chemscheme.mcp_server import chemscheme_mcp

        tool_names = [t.name for t in chemscheme_mcp._tool_manager.list_tools()]
        assert "draw_scheme" in tool_names
        assert "validate_scheme" in tool_names
        assert "resolve_name" in tool_names


class TestCLIEntryPoint:
    """AC-15.6: CLI entry point importable."""

    @pytest.mark.unit
    def test_cli_main_importable(self) -> None:
        """chemscheme.cli:main is importable."""
        from chemscheme.cli import main

        assert callable(main)

    @pytest.mark.unit
    def test_cli_version_available(self) -> None:
        """__version__ is defined in chemscheme package."""
        from chemscheme import __version__

        assert isinstance(__version__, str)
        assert "." in __version__
