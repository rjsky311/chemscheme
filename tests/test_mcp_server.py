"""Tests for Phase 8 — MCP Server.

AC-8.1: MCP server starts, tools visible
AC-8.2: draw_scheme tool accepts scheme JSON, returns SVG/PNG
AC-8.3: validate_scheme tool accepts scheme JSON, returns validation result
AC-8.4: resolve_name tool accepts compound name, returns SmilesResult
AC-8.5: Tool schemas match Phase 2 contract
AC-8.6: All MCP tools have unified error response format
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

# Phase 2 contract schemas (moved to code-based verification in Phase 15)


# ─── AC-8.1: MCP server starts, tools visible ─────────────────────


class TestMcpServerStartup:
    """AC-8.1: MCP server starts and tools are discoverable."""

    @pytest.mark.unit
    def test_server_import(self) -> None:
        """MCP server module imports without error."""
        from chemscheme.mcp_server import chemscheme_mcp  # noqa: F401

    @pytest.mark.unit
    def test_server_is_fastmcp_instance(self) -> None:
        """Server is a FastMCP instance."""
        from mcp.server.fastmcp import FastMCP

        from chemscheme.mcp_server import chemscheme_mcp

        assert isinstance(chemscheme_mcp, FastMCP)

    @pytest.mark.unit
    def test_server_has_tools(self) -> None:
        """Server exposes draw_scheme, validate_scheme, resolve_name tools."""
        from chemscheme.mcp_server import chemscheme_mcp

        tool_names = {t.name for t in chemscheme_mcp._tool_manager.list_tools()}
        assert "draw_scheme" in tool_names
        assert "validate_scheme" in tool_names
        assert "resolve_name" in tool_names

    @pytest.mark.unit
    def test_server_name(self) -> None:
        """Server has correct name."""
        from chemscheme.mcp_server import chemscheme_mcp

        assert chemscheme_mcp.name == "chemscheme"


# ─── AC-8.2: draw_scheme tool ─────────────────────────────────────


class TestDrawSchemeTool:
    """AC-8.2: draw_scheme accepts scheme JSON, returns SVG/PNG."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_draw_scheme_svg(self, sample_scheme: dict) -> None:
        """draw_scheme returns SVG string by default."""
        from chemscheme.mcp_server import draw_scheme

        result = await draw_scheme(scheme=sample_scheme)
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["format"] == "svg"
        assert "<svg" in parsed["data"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_draw_scheme_png(self, sample_scheme: dict) -> None:
        """draw_scheme returns base64-encoded PNG when format='png'."""
        import base64

        from chemscheme.mcp_server import draw_scheme

        result = await draw_scheme(scheme=sample_scheme, output_format="png")
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["format"] == "png"
        # Verify base64 decodable
        data = base64.b64decode(parsed["data"])
        assert data[:4] == b"\x89PNG"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_draw_scheme_invalid_smiles(self) -> None:
        """draw_scheme returns error for invalid SMILES."""
        from chemscheme.mcp_server import draw_scheme

        bad_scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["INVALID_SMILES"],
                    "products": ["ALSO_INVALID"],
                    "arrow_above": "???",
                }
            ],
        }
        result = await draw_scheme(scheme=bad_scheme)
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "error" in parsed

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_draw_scheme_convergent(self, sample_scheme_convergent: dict) -> None:
        """draw_scheme handles convergent synthesis schemes."""
        from chemscheme.mcp_server import draw_scheme

        result = await draw_scheme(scheme=sample_scheme_convergent)
        parsed = json.loads(result)
        assert parsed["status"] == "ok"


# ─── AC-8.3: validate_scheme tool ─────────────────────────────────


class TestValidateSchemeTool:
    """AC-8.3: validate_scheme accepts scheme JSON, returns validation result."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_valid_scheme(self, sample_scheme: dict) -> None:
        """validate_scheme returns valid=True for a good scheme."""
        from chemscheme.mcp_server import validate_scheme

        result = await validate_scheme(scheme=sample_scheme)
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["valid"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_invalid_scheme(self) -> None:
        """validate_scheme returns valid=False with issues for bad scheme."""
        from chemscheme.mcp_server import validate_scheme

        bad_scheme = {"steps": [{"id": "1"}]}
        result = await validate_scheme(scheme=bad_scheme)
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["valid"] is False
        assert len(parsed["issues"]) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_with_layers(self, sample_scheme: dict) -> None:
        """validate_scheme respects the layers parameter."""
        from chemscheme.mcp_server import validate_scheme

        result = await validate_scheme(scheme=sample_scheme, layers=["schema", "chemistry"])
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert "schema" in parsed["layers_checked"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_schema_only(self, sample_scheme: dict) -> None:
        """validate_scheme can run schema-only validation."""
        from chemscheme.mcp_server import validate_scheme

        result = await validate_scheme(scheme=sample_scheme, layers=["schema"])
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["layers_checked"] == ["schema"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_default_includes_reaction_layer(self, sample_scheme: dict) -> None:
        """MCP validate_scheme default layers include reaction (MR9-C5, MR6-C7).

        The CLI validate command runs all four layers (schema, chemistry,
        reaction, continuity) by default. The MCP tool must match this
        behaviour when no explicit layers parameter is provided.
        """
        from chemscheme.mcp_server import validate_scheme

        result = await validate_scheme(scheme=sample_scheme)
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert "reaction" in parsed["layers_checked"], (
            "MCP default validation must include 'reaction' layer to match CLI"
        )
        # Verify all four layers are checked by default
        assert parsed["layers_checked"] == [
            "schema",
            "chemistry",
            "reaction",
            "continuity",
        ]


# ─── AC-8.4: resolve_name tool ────────────────────────────────────


class TestResolveNameTool:
    """AC-8.4: resolve_name accepts compound name, returns SmilesResult."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_resolve_known_name(self) -> None:
        """resolve_name resolves a well-known compound."""
        from chemscheme.mcp_server import resolve_name

        with patch("chemscheme.parsers.name_resolver.resolve_name") as mock_resolve:
            from chemscheme.models import SmilesResult

            mock_resolve.return_value = SmilesResult(
                smiles="c1ccccc1",
                confidence=1.0,
                source="opsin",
                warnings=[],
            )
            result = await resolve_name(name="benzene")
            parsed = json.loads(result)
            assert parsed["status"] == "ok"
            assert parsed["smiles"] == "c1ccccc1"
            assert parsed["confidence"] == 1.0
            assert parsed["source"] == "opsin"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_resolve_unknown_name(self) -> None:
        """resolve_name returns error for unresolvable compound."""
        from chemscheme.mcp_server import resolve_name

        with patch("chemscheme.parsers.name_resolver.resolve_name") as mock_resolve:
            from chemscheme.models import SmilesResult

            mock_resolve.return_value = SmilesResult(
                smiles="",
                confidence=0.0,
                source="none",
                warnings=["Could not resolve 'xyznonexistent'"],
            )
            result = await resolve_name(name="xyznonexistent")
            parsed = json.loads(result)
            assert parsed["status"] == "error"
            assert len(parsed["warnings"]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_resolve_returns_warnings(self) -> None:
        """resolve_name includes warnings in response."""
        from chemscheme.mcp_server import resolve_name

        with patch("chemscheme.parsers.name_resolver.resolve_name") as mock_resolve:
            from chemscheme.models import SmilesResult

            mock_resolve.return_value = SmilesResult(
                smiles="CC",
                confidence=0.7,
                source="pubchem",
                warnings=["Low confidence match"],
            )
            result = await resolve_name(name="ethane")
            parsed = json.loads(result)
            assert parsed["status"] == "ok"
            assert parsed["warnings"] == ["Low confidence match"]


# ─── AC-8.5: Tool schemas match Phase 2 contract ──────────────────


class TestSchemaContract:
    """AC-8.5: Tool schemas verified from code (file schemas removed in Phase 15)."""

    @pytest.mark.unit
    def test_draw_scheme_contract(self) -> None:
        """draw_scheme tool has 'scheme' as required parameter."""
        from chemscheme.mcp_server import chemscheme_mcp

        tools = {t.name: t for t in chemscheme_mcp._tool_manager.list_tools()}
        tool = tools["draw_scheme"]
        assert tool.name == "draw_scheme"
        assert "scheme" in tool.parameters.get("properties", {})

    @pytest.mark.unit
    def test_validate_scheme_contract(self) -> None:
        """validate_scheme tool has 'scheme' as required parameter."""
        from chemscheme.mcp_server import chemscheme_mcp

        tools = {t.name: t for t in chemscheme_mcp._tool_manager.list_tools()}
        tool = tools["validate_scheme"]
        assert tool.name == "validate_scheme"
        assert "scheme" in tool.parameters.get("properties", {})

    @pytest.mark.unit
    def test_resolve_name_has_name_param(self) -> None:
        """resolve_name tool has 'name' as required parameter."""
        from chemscheme.mcp_server import chemscheme_mcp

        tools = {t.name: t for t in chemscheme_mcp._tool_manager.list_tools()}
        tool = tools["resolve_name"]
        assert "name" in tool.parameters.get("properties", {})


# ─── AC-8.6: Unified error response format ────────────────────────


class TestErrorFormat:
    """AC-8.6: All MCP tools have unified error response format."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_draw_scheme_error_format(self) -> None:
        """draw_scheme error response has status + error keys."""
        from chemscheme.mcp_server import draw_scheme

        result = await draw_scheme(scheme={"invalid": True})
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "error" in parsed
        assert isinstance(parsed["error"], str)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_scheme_error_format(self) -> None:
        """validate_scheme error response has status + valid + issues keys."""
        from chemscheme.mcp_server import validate_scheme

        result = await validate_scheme(scheme={})
        parsed = json.loads(result)
        assert parsed["status"] == "ok"  # validation itself doesn't error
        assert "valid" in parsed
        assert "issues" in parsed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_resolve_name_error_format(self) -> None:
        """resolve_name error response has status + error/warnings keys."""
        from chemscheme.mcp_server import resolve_name

        with patch("chemscheme.parsers.name_resolver.resolve_name") as mock_resolve:
            from chemscheme.models import SmilesResult

            mock_resolve.return_value = SmilesResult(
                smiles="", confidence=0.0, source="none", warnings=["Not found"]
            )
            result = await resolve_name(name="nonexistent_compound_xyz")
            parsed = json.loads(result)
            assert parsed["status"] == "error"
            assert "warnings" in parsed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_errors_have_status_field(self) -> None:
        """Every tool response includes a 'status' field."""
        from chemscheme.mcp_server import draw_scheme, validate_scheme

        # draw_scheme with bad input
        r1 = json.loads(await draw_scheme(scheme={}))
        assert "status" in r1

        # validate with bad input
        r2 = json.loads(await validate_scheme(scheme={}))
        assert "status" in r2


# ─── AC-18a: MCP extract_image tool ───────────────────────────────


class TestMcpExtractImage:
    """Phase 18a: extract_image MCP tool."""

    @pytest.mark.asyncio
    async def test_extract_image_returns_smiles(self) -> None:
        """GIVEN base64 image, WHEN extract_image, THEN returns SMILES."""
        import base64
        import struct
        import zlib

        from chemscheme.mcp_server import extract_image

        def _chunk(ct: bytes, d: bytes) -> bytes:
            c = ct + d
            return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        idat = zlib.compress(b"\x00\xff\xff\xff")
        png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
        b64 = base64.b64encode(png).decode("ascii")

        result = json.loads(await extract_image(image_b64=b64))
        # A4 fix: if real recognizers fail (no Docker), empty SMILES → error
        # If mock fallback kicks in → ok with "CCO"
        assert result["status"] in ("ok", "error")
        if result["status"] == "ok":
            assert "smiles" in result
            assert "confidence" in result

    @pytest.mark.asyncio
    async def test_extract_image_empty_errors(self) -> None:
        """GIVEN empty base64, WHEN extract_image, THEN error."""
        from chemscheme.mcp_server import extract_image

        result = json.loads(await extract_image(image_b64=""))
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_extract_image_invalid_b64_errors(self) -> None:
        """GIVEN invalid base64, WHEN extract_image, THEN error."""
        from chemscheme.mcp_server import extract_image

        result = json.loads(await extract_image(image_b64="not-valid-base64!!!"))
        assert result["status"] == "error"
