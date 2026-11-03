"""MR13 regression tests — Phase 14 review findings.

G1: Mathpix file size guard
G2: Indigo list arrow_above normalization
C2: PNG theme parity (themed PNG rendering)
C3: estimate_step_size annotation mismatch

# RED-EXEMPT: review finding regressions
"""

from __future__ import annotations

from pathlib import Path

import pytest

_has_indigo = True
try:
    import indigo as _indigo_check  # noqa: F401
except ImportError:
    _has_indigo = False


class TestMR13G1MathpixFileSize:
    """G1: Mathpix rejects oversized image files."""

    # RED-EXEMPT: bugfix characterization

    def test_large_file_rejected(self, tmp_path: Path) -> None:
        """Files > 50 MB are rejected with ValueError."""
        from chemscheme.parsers.mathpix_client import MathpixClient

        # Create a fake large file (just write sparse)
        large_file = tmp_path / "huge.png"
        with large_file.open("wb") as f:
            f.seek(51 * 1024 * 1024)  # 51 MB
            f.write(b"\x00")

        client = MathpixClient(app_id="test", app_key="test")
        with pytest.raises(ValueError, match="too large"):
            client.extract_text_from_file(large_file)


@pytest.mark.skipif(not _has_indigo, reason="indigo not installed")
class TestMR13G2IndigoListArrowAbove:
    """G2: Indigo normalizes list-form arrow_above."""

    # RED-EXEMPT: bugfix characterization

    def test_list_arrow_above_no_crash(self) -> None:
        """Indigo render_step handles arrow_above as list without crash."""
        from chemscheme.renderers.indigo_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": ["NaOH", "H2O"],
        }
        result = render_step(step, fmt="svg")
        assert b"<svg" in result


class TestMR13C2PNGThemeParity:
    """C2: PNG rendering respects theme when specified."""

    # RED-EXEMPT: review finding

    def test_themed_png_produces_png_bytes(self) -> None:
        """render_step with theme + fmt=png produces valid PNG.

        With CairoSVG: fully themed SVG→PNG conversion.
        Without CairoSVG: falls back to Pillow legacy renderer (partial theme).
        Either way, produces valid PNG bytes.
        """
        from chemscheme.renderers.rdkit_renderer import render_step
        from chemscheme.renderers.themes import get_theme

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "NaOH",
        }
        theme = get_theme("acs")
        result = render_step(step, fmt="png", theme=theme)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"


class TestMR13C3EstimateSizeAnnotations:
    """C3: estimate_step_size accounts for annotations."""

    # RED-EXEMPT: review finding

    def test_annotations_increase_height(self) -> None:
        """Step with labels/yield/legend should estimate taller than plain step."""
        from chemscheme.renderers.rdkit_renderer import estimate_step_size

        plain = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "NaOH",
        }
        annotated = {
            "id": "2",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "NaOH",
            "arrow_below": "THF",
            "yield": "85%",
            "compound_labels": {"c1ccccc1": "1a"},
        }
        _, plain_h = estimate_step_size(plain)
        _, annotated_h = estimate_step_size(annotated)
        assert annotated_h > plain_h, "Annotated step should be taller"

    def test_multiline_above_increases_height(self) -> None:
        """Multi-line arrow_above should estimate taller."""
        from chemscheme.renderers.rdkit_renderer import estimate_step_size

        single = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "NaOH",
        }
        multi = {
            "id": "2",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "NaOH\nTHF\n80 C",
        }
        _, single_h = estimate_step_size(single)
        _, multi_h = estimate_step_size(multi)
        assert multi_h > single_h, "Multi-line above should be taller"
