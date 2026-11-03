"""Tests for --json output format and MCP tool schema (AC-2.5, AC-2.6).

BDD Feature: JSON output mode and MCP contract
"""

import json
from pathlib import Path

import pytest


@pytest.mark.unit
class TestJsonOutputFormat:
    """AC-2.5: --json structured output format."""

    def test_render_result_has_required_keys(self) -> None:
        """JSON render result has status, output_path, warnings."""
        import tempfile

        from chemscheme.renderers.rdkit_renderer import render_step_to_file

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.svg"
            result = render_step_to_file(step, out_path, fmt="svg")
            assert "status" in result
            assert "output_path" in result
            assert "warnings" in result

    def test_render_result_status_ok(self) -> None:
        """Successful render returns status='ok'."""
        import tempfile

        from chemscheme.renderers.rdkit_renderer import render_step_to_file

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.svg"
            result = render_step_to_file(step, out_path, fmt="svg")
            assert result["status"] == "ok"
            assert Path(result["output_path"]).exists()

    def test_render_result_is_json_serializable(self) -> None:
        """Result dict can be serialized to JSON."""
        import tempfile

        from chemscheme.renderers.rdkit_renderer import render_step_to_file

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "H2O2",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.svg"
            result = render_step_to_file(step, out_path, fmt="svg")
            serialized = json.dumps(result)
            assert isinstance(serialized, str)

    def test_validate_result_has_required_keys(self) -> None:
        """JSON validate result has status, valid, issues, layers_checked."""
        from chemscheme.schema.scheme_schema import validate_scheme_json

        scheme = {
            "schema_version": "1",
            "title": "Test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "H2O2",
                }
            ],
        }
        result = validate_scheme_json(scheme)
        assert "status" in result
        assert "valid" in result
        assert "issues" in result
        assert "layers_checked" in result

    def test_validate_result_valid_scheme(self) -> None:
        """Valid scheme returns status='ok', valid=True."""
        from chemscheme.schema.scheme_schema import validate_scheme_json

        scheme = {
            "schema_version": "1",
            "title": "Test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "H2O2",
                }
            ],
        }
        result = validate_scheme_json(scheme)
        assert result["status"] == "ok"
        assert result["valid"] is True
        assert result["issues"] == []

    def test_validate_result_invalid_scheme(self) -> None:
        """Invalid scheme returns valid=False with issues."""
        from chemscheme.schema.scheme_schema import validate_scheme_json

        scheme = {
            "schema_version": "1",
            "title": "Bad",
            "steps": [{"id": "1", "reactants": ["c1ccccc1"], "arrow_above": "X"}],
        }
        result = validate_scheme_json(scheme)
        assert result["valid"] is False
        assert len(result["issues"]) > 0


@pytest.mark.unit
class TestMCPToolSchema:
    """AC-2.6: MCP tools registered with correct names (code-based, not file-based)."""

    def test_draw_scheme_tool_registered(self) -> None:
        """draw_scheme tool is registered in FastMCP server."""
        from chemscheme.mcp_server import chemscheme_mcp

        tool_names = [t.name for t in chemscheme_mcp._tool_manager.list_tools()]
        assert "draw_scheme" in tool_names

    def test_validate_scheme_tool_registered(self) -> None:
        """validate_scheme tool is registered in FastMCP server."""
        from chemscheme.mcp_server import chemscheme_mcp

        tool_names = [t.name for t in chemscheme_mcp._tool_manager.list_tools()]
        assert "validate_scheme" in tool_names

    def test_resolve_name_tool_registered(self) -> None:
        """resolve_name tool is registered in FastMCP server."""
        from chemscheme.mcp_server import chemscheme_mcp

        tool_names = [t.name for t in chemscheme_mcp._tool_manager.list_tools()]
        assert "resolve_name" in tool_names
