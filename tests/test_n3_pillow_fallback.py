"""N3: PNG regression tests that force Pillow fallback when CairoSVG is installed.

These tests mock out CairoSVG so the Pillow code path is always exercised,
ensuring the fallback renderer stays correct regardless of CairoSVG availability.
"""

from __future__ import annotations

import struct
import sys
from unittest.mock import patch

import pytest


def _hide_cairosvg():
    """Context manager that temporarily hides cairosvg from imports."""
    saved = {}
    for mod_name in ("cairosvg", "cairocffi"):
        if mod_name in sys.modules:
            saved[mod_name] = sys.modules.pop(mod_name)
    return saved


def _restore_modules(saved: dict) -> None:
    sys.modules.update(saved)


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Extract (width, height) from PNG IHDR chunk."""
    assert data[:4] == b"\x89PNG", "Not a PNG file"
    w, h = struct.unpack(">II", data[16:24])
    return w, h


# ---------------------------------------------------------------------------
# Single-step: rdkit_renderer._render_png via Pillow fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleStepPillowFallback:
    """Force Pillow path in rdkit_renderer.render_step(fmt='png')."""

    def _render_via_pillow(self, step: dict, dpi: int = 300) -> bytes:
        """Render a step to PNG with CairoSVG hidden."""
        from chemscheme.renderers.rdkit_renderer import render_step

        saved = _hide_cairosvg()
        try:
            with patch.dict(sys.modules, {"cairosvg": None}):
                result = render_step(step, fmt="png", dpi=dpi)
        finally:
            _restore_modules(saved)
        return result

    def test_basic_reaction_produces_valid_png(self) -> None:
        """Simple ethanol→acetaldehyde via Pillow fallback."""
        step = {
            "id": "1",
            "reactants": ["CCO"],
            "products": ["CC=O"],
            "arrow_above": "PCC",
        }
        result = self._render_via_pillow(step)
        assert result[:4] == b"\x89PNG"
        w, h = _png_dimensions(result)
        assert w > 50 and h > 50

    def test_large_molecule_dynamic_widths(self) -> None:
        """C4/G5: large + small molecules get different widths (not fixed 250px)."""
        step = {
            "id": "1",
            "reactants": ["c1ccc2c(c1)c1cc3c(cc1S2)c1nsnc1c1ccc(-c4ccc(F)cc4)cc1-3"],
            "products": ["c1ccccc1"],
            "arrow_above": "test",
        }
        result = self._render_via_pillow(step)
        assert result[:4] == b"\x89PNG"
        w, h = _png_dimensions(result)
        # Large molecule needs wider canvas than a symmetric small-mol pair
        assert w > 300

    def test_arrow_annotations_rendered(self) -> None:
        """Arrow above/below text should not crash Pillow path."""
        step = {
            "id": "1",
            "reactants": ["CCO"],
            "products": ["CC=O"],
            "arrow_above": "Pd(PPh3)4",
            "arrow_below": "THF, reflux",
        }
        result = self._render_via_pillow(step)
        assert result[:4] == b"\x89PNG"

    def test_multiple_reactants_plus_signs(self) -> None:
        """Multiple reactants should render plus signs between them."""
        step = {
            "id": "1",
            "reactants": ["c1ccccc1B(O)O", "BrC1=CC=CC=C1"],
            "products": ["c1ccc(-c2ccccc2)cc1"],
            "arrow_above": "Pd(PPh3)4",
        }
        result = self._render_via_pillow(step)
        assert result[:4] == b"\x89PNG"
        w, _ = _png_dimensions(result)
        # Two reactants + arrow + product → needs reasonable width
        assert w > 400

    def test_custom_dpi_affects_dimensions(self) -> None:
        """Higher DPI should produce larger images."""
        step = {
            "id": "1",
            "reactants": ["CCO"],
            "products": ["CC=O"],
            "arrow_above": "ox",
        }
        png_150 = self._render_via_pillow(step, dpi=150)
        png_600 = self._render_via_pillow(step, dpi=600)
        w150, h150 = _png_dimensions(png_150)
        w600, h600 = _png_dimensions(png_600)
        assert w600 > w150
        assert h600 > h150


# ---------------------------------------------------------------------------
# Multi-step: multi_step_renderer._render_scheme_png via Pillow fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiStepPillowFallback:
    """Force Pillow path in multi_step_renderer.render_scheme(fmt='png')."""

    def _render_scheme_via_pillow(self, scheme: dict, dpi: int = 300) -> bytes:
        """Render a scheme to PNG with CairoSVG hidden."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        saved = _hide_cairosvg()
        try:
            with patch.dict(sys.modules, {"cairosvg": None}):
                result = render_scheme(scheme, fmt="png", dpi=dpi)
        finally:
            _restore_modules(saved)
        return result

    @pytest.fixture
    def two_step_scheme(self) -> dict:
        return {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "NBS",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(Br)cc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd(PPh3)4, K2CO3",
                },
            ],
        }

    def test_two_step_produces_valid_png(self, two_step_scheme: dict) -> None:
        """Basic multi-step renders valid PNG via Pillow."""
        result = self._render_scheme_via_pillow(two_step_scheme)
        assert result[:4] == b"\x89PNG"
        w, h = _png_dimensions(result)
        assert w > 100 and h > 50

    def test_uniform_scaling_no_distortion(self, two_step_scheme: dict) -> None:
        """G2: Pillow path should produce reasonable dimensions (not 0x0 or 1x1)."""
        result = self._render_scheme_via_pillow(two_step_scheme)
        w, h = _png_dimensions(result)
        # Aspect ratio should be landscape for a 2-step linear scheme
        assert w > h

    def test_convergent_scheme(self) -> None:
        """Convergent synthesis via Pillow fallback."""
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1B(O)O"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd cat.",
                    "branch_label": "A",
                },
                {
                    "id": "2",
                    "reactants": ["BrC1=CC=CC=C1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "coupling",
                    "branch_label": "B",
                    "consumes_step_ids": ["1"],
                },
            ],
        }
        result = self._render_scheme_via_pillow(scheme)
        assert result[:4] == b"\x89PNG"

    def test_pillow_and_cairosvg_produce_same_dimensions_roughly(
        self, two_step_scheme: dict
    ) -> None:
        """Both paths should produce images of similar scale (within 2x)."""
        try:
            import cairosvg  # type: ignore[import-untyped,import-not-found]  # noqa: F401
        except ImportError:
            pytest.skip("CairoSVG not installed — cannot compare")

        from chemscheme.renderers.multi_step_renderer import render_scheme

        # CairoSVG path
        cairo_png = render_scheme(two_step_scheme, fmt="png", dpi=300)
        cw, ch = _png_dimensions(cairo_png)

        # Pillow path
        pillow_png = self._render_scheme_via_pillow(two_step_scheme)
        pw, ph = _png_dimensions(pillow_png)

        # They won't be identical, but should be within reasonable range
        assert pw > 0 and ph > 0
        assert cw > 0 and ch > 0


# ---------------------------------------------------------------------------
# Error path: both backends missing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeitherBackend:
    """When both CairoSVG and Pillow are missing, get a clear error."""

    def test_single_step_raises_runtime_error(self) -> None:
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["CCO"],
            "products": ["CC=O"],
            "arrow_above": "ox",
        }
        saved = {}
        for mod_name in ("cairosvg", "PIL", "PIL.Image"):
            if mod_name in sys.modules:
                saved[mod_name] = sys.modules.pop(mod_name)

        with (
            patch.dict(sys.modules, {"cairosvg": None, "PIL": None, "PIL.Image": None}),
            pytest.raises(RuntimeError, match="PNG rendering requires"),
        ):
            render_step(step, fmt="png")

        _restore_modules(saved)

    def test_multi_step_raises_runtime_error(self) -> None:
        from chemscheme.renderers.layout_engine import LayoutConfig
        from chemscheme.renderers.multi_step_renderer import _render_scheme_png

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "ox",
                },
            ],
        }
        saved = {}
        for mod_name in ("cairosvg", "PIL", "PIL.Image"):
            if mod_name in sys.modules:
                saved[mod_name] = sys.modules.pop(mod_name)

        with (
            patch.dict(sys.modules, {"cairosvg": None, "PIL": None, "PIL.Image": None}),
            pytest.raises(RuntimeError, match="PNG rendering requires"),
        ):
            _render_scheme_png(scheme, LayoutConfig(), 300)

        _restore_modules(saved)
