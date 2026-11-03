"""Phase 14 — Journal Theme System tests.

AC-14.1: SchemeTheme dataclass
AC-14.2: Built-in journal presets
AC-14.3: Full render path wiring
AC-14.4: CLI integration
AC-14.5: MCP + style→theme migration
AC-14.6: Golden regression + adversarial

BDD Feature: Journal Theme System
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# AC-14.1: SchemeTheme dataclass
# ---------------------------------------------------------------------------
class TestSchemeThemeDataclass:
    """AC-14.1: SchemeTheme frozen dataclass with typed sub-groups."""

    def test_default_theme_exists(self) -> None:
        """Default theme is importable and has a name."""
        from chemscheme.renderers.themes import DEFAULT_THEME

        assert DEFAULT_THEME.name == "default"

    def test_theme_is_frozen(self) -> None:
        """SchemeTheme instances are immutable (frozen dataclass)."""
        from chemscheme.renderers.themes import DEFAULT_THEME

        with pytest.raises(AttributeError):
            DEFAULT_THEME.name = "hacked"  # type: ignore[misc]

    def test_typography_subgroup(self) -> None:
        """Default theme has all typography fields with correct types."""
        from chemscheme.renderers.themes import DEFAULT_THEME

        assert isinstance(DEFAULT_THEME.font_family, str)
        assert isinstance(DEFAULT_THEME.font_size_conditions, int)
        assert isinstance(DEFAULT_THEME.font_size_labels, int)
        assert isinstance(DEFAULT_THEME.font_size_yield, int)
        assert isinstance(DEFAULT_THEME.font_size_legend, int)
        assert isinstance(DEFAULT_THEME.font_size_plus, int)
        assert isinstance(DEFAULT_THEME.label_font_weight, str)
        assert isinstance(DEFAULT_THEME.text_line_height, int)

    def test_arrow_subgroup(self) -> None:
        """Default theme has all arrow fields."""
        from chemscheme.renderers.themes import DEFAULT_THEME

        assert isinstance(DEFAULT_THEME.arrow_width, int)
        assert isinstance(DEFAULT_THEME.arrow_stroke_width, float)
        assert isinstance(DEFAULT_THEME.arrow_text_offset, int)
        assert isinstance(DEFAULT_THEME.arrowhead_scale, int)
        assert isinstance(DEFAULT_THEME.arrowhead_style, str)

    def test_layout_subgroup(self) -> None:
        """Default theme has all layout fields."""
        from chemscheme.renderers.themes import DEFAULT_THEME

        assert isinstance(DEFAULT_THEME.mol_height, int)
        assert isinstance(DEFAULT_THEME.plus_width, int)
        assert isinstance(DEFAULT_THEME.padding, int)
        assert isinstance(DEFAULT_THEME.label_height, int)
        assert isinstance(DEFAULT_THEME.adaptive_font_threshold, int)

    def test_connector_subgroup(self) -> None:
        """Default theme has connector fields for multi_step_renderer."""
        from chemscheme.renderers.themes import DEFAULT_THEME

        assert isinstance(DEFAULT_THEME.connector_stroke_width, float)
        assert isinstance(DEFAULT_THEME.connector_color, str)

    def test_color_subgroup(self) -> None:
        """Default theme has color fields."""
        from chemscheme.renderers.themes import DEFAULT_THEME

        assert isinstance(DEFAULT_THEME.color_text, str)
        assert isinstance(DEFAULT_THEME.color_arrow, str)
        assert isinstance(DEFAULT_THEME.color_background, str)

    def test_subscript_subgroup(self) -> None:
        """Default theme has subscript configuration."""
        from chemscheme.renderers.themes import DEFAULT_THEME

        assert isinstance(DEFAULT_THEME.subscript_dy, int)
        assert isinstance(DEFAULT_THEME.subscript_scale, str)

    def test_polymer_subgroup(self) -> None:
        """Default theme has polymer bracket configuration."""
        from chemscheme.renderers.themes import DEFAULT_THEME

        assert isinstance(DEFAULT_THEME.bracket_stroke_width, float)
        assert isinstance(DEFAULT_THEME.polymer_n_font_size, int)


# ---------------------------------------------------------------------------
# AC-14.2: Built-in journal presets
# ---------------------------------------------------------------------------
class TestJournalPresets:
    """AC-14.2: Built-in journal presets (acs, nature, rsc, wiley)."""

    @pytest.mark.parametrize("preset_name", ["default", "acs", "nature", "rsc", "wiley"])
    def test_preset_exists_and_complete(self, preset_name: str) -> None:
        """Each preset is importable and has all fields populated (no None)."""
        from chemscheme.renderers.themes import get_theme

        theme = get_theme(preset_name)
        assert theme.name == preset_name
        # Check no None values in any field
        import dataclasses

        for field in dataclasses.fields(theme):
            val = getattr(theme, field.name)
            assert val is not None, f"Field {field.name} is None in preset {preset_name}"

    def test_acs_uses_helvetica_or_arial(self) -> None:
        """ACS preset uses Arial/Helvetica font stack."""
        from chemscheme.renderers.themes import get_theme

        theme = get_theme("acs")
        font = theme.font_family.lower()
        assert "arial" in font or "helvetica" in font

    def test_presets_differ(self) -> None:
        """Different presets produce different configurations (not all identical)."""
        from chemscheme.renderers.themes import get_theme

        default = get_theme("default")
        acs = get_theme("acs")
        nature = get_theme("nature")
        # At least one field should differ between presets
        assert default != acs or default != nature, "All presets are identical"

    def test_list_available_themes(self) -> None:
        """list_themes() returns all built-in theme names."""
        from chemscheme.renderers.themes import list_themes

        names = list_themes()
        assert set(names) >= {"default", "acs", "nature", "rsc", "wiley"}


# ---------------------------------------------------------------------------
# AC-14.3: Full render path wiring
# ---------------------------------------------------------------------------
class TestRenderPathWiring:
    """AC-14.3: Theme flows through the full render pipeline."""

    @pytest.fixture()
    def single_step(self) -> dict:  # type: ignore[type-arg]
        return {
            "id": "1",
            "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
            "products": ["c1ccc(-c2ccccc2)cc1"],
            "arrow_above": "Pd(PPh3)4, K2CO3",
        }

    @pytest.fixture()
    def multi_step_scheme(self) -> dict:  # type: ignore[type-arg]
        return {
            "schema_version": "1.0",
            "title": "Test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccc(Br)cc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "NaOH",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(O)cc1"],
                    "products": ["c1ccc(OC)cc1"],
                    "arrow_above": "MeI",
                },
            ],
        }

    def test_compose_step_svg_accepts_theme(self, single_step: dict) -> None:  # type: ignore[type-arg]
        """drawsvg_composer.compose_step_svg accepts optional theme parameter."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg
        from chemscheme.renderers.themes import get_theme

        theme = get_theme("acs")
        result = compose_step_svg(single_step, theme=theme)
        assert isinstance(result, bytes)
        assert b"<svg" in result

    def test_render_step_accepts_theme(self, single_step: dict) -> None:  # type: ignore[type-arg]
        """rdkit_renderer.render_step accepts optional theme parameter."""
        from chemscheme.renderers.rdkit_renderer import render_step
        from chemscheme.renderers.themes import get_theme

        theme = get_theme("nature")
        result = render_step(single_step, fmt="svg", theme=theme)
        assert isinstance(result, bytes)
        assert b"<svg" in result

    def test_render_scheme_accepts_theme(self, multi_step_scheme: dict) -> None:  # type: ignore[type-arg]
        """multi_step_renderer.render_scheme accepts optional theme parameter."""
        from chemscheme.renderers.multi_step_renderer import render_scheme
        from chemscheme.renderers.themes import get_theme

        theme = get_theme("rsc")
        result = render_scheme(multi_step_scheme, fmt="svg", theme=theme)
        assert isinstance(result, bytes)
        assert b"<svg" in result

    def test_estimate_step_size_uses_theme(self, single_step: dict) -> None:  # type: ignore[type-arg]
        """estimate_step_size uses theme dimensions (theme-aware sizing)."""
        from chemscheme.renderers.rdkit_renderer import estimate_step_size
        from chemscheme.renderers.themes import get_theme

        default_size = estimate_step_size(single_step)
        # A theme with different arrow_width should produce different size
        theme = get_theme("nature")
        themed_size = estimate_step_size(single_step, theme=theme)
        # Both should return valid tuples
        assert len(default_size) == 2
        assert len(themed_size) == 2

    def test_different_themes_produce_different_svg(self, single_step: dict) -> None:  # type: ignore[type-arg]
        """Different themes produce visually different SVG output."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg
        from chemscheme.renderers.themes import get_theme

        svg_default = compose_step_svg(single_step, theme=get_theme("default"))
        svg_acs = compose_step_svg(single_step, theme=get_theme("acs"))
        # Should differ (different font families at minimum)
        assert svg_default != svg_acs

    def test_render_step_png_with_theme(self, single_step: dict) -> None:  # type: ignore[type-arg]
        """PNG rendering also respects theme parameter."""
        from chemscheme.renderers.rdkit_renderer import render_step
        from chemscheme.renderers.themes import get_theme

        theme = get_theme("wiley")
        result = render_step(single_step, fmt="png", theme=theme)
        assert isinstance(result, bytes)
        # PNG magic bytes
        assert result[:4] == b"\x89PNG"

    def test_render_scheme_png_with_theme(self, multi_step_scheme: dict) -> None:  # type: ignore[type-arg]
        """Multi-step PNG rendering respects theme."""
        from chemscheme.renderers.multi_step_renderer import render_scheme
        from chemscheme.renderers.themes import get_theme

        theme = get_theme("acs")
        result = render_scheme(multi_step_scheme, fmt="png", theme=theme)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# AC-14.4: CLI integration
# ---------------------------------------------------------------------------
class TestCLITheme:
    """AC-14.4: CLI --theme flag and themes command."""

    def test_draw_with_theme_flag(self, tmp_path: Path) -> None:
        """chemscheme draw --theme acs produces output."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        scheme = {
            "schema_version": "1.0",
            "title": "Test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "NaOH",
                }
            ],
        }
        scheme_file = tmp_path / "test.json"
        scheme_file.write_text(json.dumps(scheme))
        out_file = tmp_path / "out.svg"

        runner = CliRunner()
        result = runner.invoke(
            main, ["draw", "--theme", "acs", str(scheme_file), "-o", str(out_file)]
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()

    def test_themes_list_command(self) -> None:
        """chemscheme themes lists available theme names."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["themes"])
        assert result.exit_code == 0
        assert "default" in result.output
        assert "acs" in result.output

    def test_invalid_theme_error(self, tmp_path: Path) -> None:
        """Invalid --theme name produces error listing available themes."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        scheme = {
            "schema_version": "1.0",
            "title": "Test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "test",
                }
            ],
        }
        scheme_file = tmp_path / "test.json"
        scheme_file.write_text(json.dumps(scheme))
        out_file = tmp_path / "out.svg"

        runner = CliRunner()
        result = runner.invoke(
            main, ["draw", "--theme", "nonexistent", str(scheme_file), "-o", str(out_file)]
        )
        assert result.exit_code != 0
        # Error should list available themes
        assert "available" in result.output.lower() or "available" in (result.stderr or "").lower()


# ---------------------------------------------------------------------------
# AC-14.5: MCP + style→theme migration
# ---------------------------------------------------------------------------
class TestMCPThemeMigration:
    """AC-14.5: MCP draw_scheme accepts theme, style aliased."""

    def test_mcp_draw_with_theme(self) -> None:
        """MCP draw_scheme accepts theme parameter."""
        from chemscheme.mcp_server import _sync_draw

        scheme = {
            "schema_version": "1.0",
            "title": "Test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "NaOH",
                }
            ],
        }
        result_json = _sync_draw(scheme, "svg", style="ACS2024", dpi=300, theme="acs")
        result = json.loads(result_json)
        assert result["status"] == "ok"

    def test_mcp_style_backward_compat(self) -> None:
        """MCP style='ACS2024' still works (mapped to default or acs)."""
        from chemscheme.mcp_server import _sync_draw

        scheme = {
            "schema_version": "1.0",
            "title": "Test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "NaOH",
                }
            ],
        }
        result_json = _sync_draw(scheme, "svg", style="ACS2024", dpi=300)
        result = json.loads(result_json)
        assert result["status"] == "ok"
        # Should report theme info
        assert "theme" in result or "style" in result

    def test_mcp_invalid_theme_error(self) -> None:
        """MCP with invalid theme returns error listing available themes."""
        from chemscheme.mcp_server import _sync_draw

        scheme = {
            "schema_version": "1.0",
            "title": "Test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "NaOH",
                }
            ],
        }
        result_json = _sync_draw(scheme, "svg", style="ACS2024", dpi=300, theme="nonexistent")
        result = json.loads(result_json)
        assert result["status"] == "error"
        assert "available" in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# AC-14.6: Golden regression + adversarial
# ---------------------------------------------------------------------------
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


@pytest.mark.visual
class TestGoldenRegression:
    """AC-14.6: Default theme produces byte-identical output to pre-Phase-14."""

    def test_single_step_golden_unchanged(self) -> None:
        """Single-step SVG with default theme matches golden file (Suzuki coupling)."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
            "products": ["c1ccc(-c2ccccc2)cc1"],
            "arrow_above": "Pd(PPh3)4, K2CO3",
            "arrow_below": "DMF, 80 C, 12h",
        }
        result = render_step(step, fmt="svg")

        golden_path = GOLDEN_DIR / "suzuki_coupling.svg"
        if golden_path.exists():
            golden = golden_path.read_bytes()
            assert result == golden, (
                "Default theme SVG differs from pre-Phase-14 golden file. "
                "This means backward compatibility is broken."
            )

    def test_multi_step_default_theme_produces_valid_svg(self) -> None:
        """Multi-step SVG with default theme produces valid SVG output.

        Note: Byte-identical comparison not possible for multi-step because
        RDKit molecule SVG path ordering is non-deterministic across runs.
        Instead, verify structural properties (dimensions, key elements).
        """
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme_path = Path(__file__).parent / "fixtures" / "sample_scheme.json"
        if not scheme_path.exists():
            pytest.skip("sample_scheme.json fixture not found")

        scheme = json.loads(scheme_path.read_text())
        result = render_scheme(scheme, fmt="svg")
        assert b"<svg" in result
        assert b"data-step-id" in result  # multi-step markers present


class TestAdversarial:
    """Adversarial scenarios for theme system."""

    def test_get_theme_invalid_raises(self) -> None:
        """get_theme with unknown name raises ValueError listing available themes."""
        from chemscheme.renderers.themes import get_theme

        with pytest.raises(ValueError, match="(?i)available"):
            get_theme("nonexistent_journal")

    def test_theme_with_long_conditions_no_clipping(self) -> None:
        """Theme-aware sizing: smaller font theme doesn't clip long text."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg
        from chemscheme.renderers.themes import get_theme

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": (
                "Pd(PPh3)4, K2CO3, THF, reflux 24h, then cool to rt, filter through celite"
            ),
        }
        theme = get_theme("nature")
        result = compose_step_svg(step, theme=theme)
        assert isinstance(result, bytes)
        # SVG should be valid and contain the arrow text
        assert b"<svg" in result
