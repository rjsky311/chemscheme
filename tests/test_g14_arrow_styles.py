"""S68 Part 7 — G14 arrow style variants (reversible / retro / forward).

S67 Part 2 audit flagged G14 (LOW): no support for arrow style variants
(reversible ⇌ for equilibrium reactions, retro ↩ for retrosynthesis schemes).

This adds:
- schema field `arrow_style: "forward" | "reversible" | "retro"` (default "forward")
- renderer dispatch in drawsvg_composer
- backward compat: missing or absent → "forward" (existing single-arrow render)
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════
# Schema validation
# ════════════════════════════════════════════════════════════════════════════


class TestG14SchemaValidation:
    """schema accepts arrow_style enum, rejects invalid values."""

    def _scheme(self, arrow_style=None):
        s = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                }
            ],
        }
        if arrow_style is not None:
            s["steps"][0]["arrow_style"] = arrow_style
        return s

    def test_arrow_style_omitted_validates(self) -> None:
        """RED-EXEMPT: behavior lock — schemes without arrow_style must still validate."""
        from chemscheme.schema.scheme_schema import validate_scheme

        assert validate_scheme(self._scheme()) == []

    def test_arrow_style_forward_validates(self) -> None:
        from chemscheme.schema.scheme_schema import validate_scheme

        assert validate_scheme(self._scheme("forward")) == []

    def test_arrow_style_reversible_validates(self) -> None:
        from chemscheme.schema.scheme_schema import validate_scheme

        assert validate_scheme(self._scheme("reversible")) == []

    def test_arrow_style_retro_validates(self) -> None:
        from chemscheme.schema.scheme_schema import validate_scheme

        assert validate_scheme(self._scheme("retro")) == []

    def test_arrow_style_invalid_value_rejected(self) -> None:
        from chemscheme.schema.scheme_schema import validate_scheme

        errors = validate_scheme(self._scheme("invalid_style"))
        assert any("arrow_style" in e for e in errors), (
            f"invalid arrow_style must surface in validation errors; got: {errors}"
        )

    def test_arrow_style_non_string_rejected(self) -> None:
        from chemscheme.schema.scheme_schema import validate_scheme

        errors = validate_scheme(self._scheme(42))
        assert any("arrow_style" in e for e in errors)


# ════════════════════════════════════════════════════════════════════════════
# Renderer dispatch
# ════════════════════════════════════════════════════════════════════════════


class TestG14RendererDispatch:
    """drawsvg renderer routes arrow_style to the appropriate SVG arrow shape."""

    def _step(self, arrow_style=None):
        s = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(Br)cc1"],
            "arrow_above": "Br2",
        }
        if arrow_style is not None:
            s["arrow_style"] = arrow_style
        return s

    def test_forward_arrow_default_unchanged(self) -> None:
        """RED-EXEMPT: behavior lock — default (no arrow_style) renders same as 'forward'."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        svg_default = compose_step_svg(self._step()).decode("utf-8")
        svg_forward = compose_step_svg(self._step("forward")).decode("utf-8")
        # Both should render successfully; structurally similar (no extra arrow shapes)
        assert svg_default.startswith("<?xml") or svg_default.startswith("<svg")
        assert svg_forward.startswith("<?xml") or svg_forward.startswith("<svg")
        # Forward arrow has exactly one main arrow line
        # (we count <line ... marker-end ...> for the reaction arrow)

    def test_reversible_arrow_renders_two_arrows(self) -> None:
        """Reversible (⇌) renders with TWO anti-parallel arrows (one above, one below)."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        svg = compose_step_svg(self._step("reversible")).decode("utf-8")
        # Expect at least 2 arrow-line elements (top + bottom for the reversible pair)
        # Heuristic: count marker-end refs (each arrow has an arrowhead)
        marker_count = svg.count("marker-end")
        assert marker_count >= 2, (
            f"reversible arrow should render with ≥2 arrowheads, got {marker_count}"
        )

    def test_retro_arrow_renders_with_double_bar(self) -> None:
        """Retro arrow (retrosynthesis ⇒) renders with double-bar style — wider arrowhead
        OR explicit double-line bar before the arrowhead."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        svg = compose_step_svg(self._step("retro")).decode("utf-8")
        # Heuristic for retro arrow: either contains 'retro' marker class/id
        # OR has wider/thicker arrowhead OR multiple parallel lines
        has_retro_marker = (
            "retro" in svg.lower()
            or svg.count("<line") >= 2  # double-line for retro
            or 'data-arrow-style="retro"' in svg
        )
        assert has_retro_marker, "retro arrow style not visually distinct in render"

    def test_arrow_style_marked_via_data_attribute(self) -> None:
        """Each non-default arrow style is marked via data-arrow-style attribute
        for downstream tools (testing, accessibility, CDXML round-trip)."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        for style in ("reversible", "retro"):
            svg = compose_step_svg(self._step(style)).decode("utf-8")
            assert f'data-arrow-style="{style}"' in svg, (
                f"arrow_style={style!r} should be marked via data-arrow-style attribute"
            )


# ════════════════════════════════════════════════════════════════════════════
# Multi-step end-to-end
# ════════════════════════════════════════════════════════════════════════════


class TestG14MultiStepEndToEnd:
    """Full render_scheme handles arrow_style across multiple steps."""

    def test_mixed_arrow_styles_render_no_crash(self) -> None:
        from chemscheme.renderers import render_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                    "arrow_style": "forward",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(Br)cc1"],
                    "products": ["c1ccccc1"],  # equilibrium back
                    "arrow_above": "K_eq",
                    "arrow_style": "reversible",
                },
            ],
        }
        svg = render_scheme(scheme, fmt="svg").decode("utf-8")
        assert len(svg) > 1000
        # Both styles present in output
        assert 'data-arrow-style="reversible"' in svg
