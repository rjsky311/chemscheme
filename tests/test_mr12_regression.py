"""MR12 regression tests — Post-Phase 13 review fixes.

C1: CDXML export normalizes list-form arrow_above
C3: polymer_info n coerced to string
C4: Polymer detection requires 2+ dummy atoms
G2: Output path directory check
"""

import json
from pathlib import Path


class TestC1CdxmlArrowAboveList:
    """C1 (P1): CDXML export must handle arrow_above as list[str]."""

    def test_cdxml_export_list_arrow_above(self) -> None:
        """List-form arrow_above is normalized to newline-joined string in CDXML."""
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": ["K2CO3", "DMSO", "80 °C"],
                }
            ],
        }
        cdxml = scheme_to_cdxml(scheme)
        # Should contain the conditions as text, not a Python list repr
        assert "K2CO3" in cdxml
        assert "['K2CO3'" not in cdxml  # No raw list serialization


class TestC3PolymerInfoNCoercion:
    """C3 (P2): Non-string polymer_info n values don't crash rendering."""

    def test_integer_n_label(self) -> None:
        """polymer_info with integer n value renders without crash."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["*CC*"],
            "products": [],
            "arrow_above": "",
            "polymer_info": {"*CC*": {"n": 10}},  # integer, not string
        }
        svg = compose_step_svg(step).decode("utf-8")
        assert ">10<" in svg


class TestC4PolymerDetectionTwoDummies:
    """C4 (P2): Only molecules with 2+ dummy atoms are polymer units."""

    def test_single_dummy_not_polymer(self) -> None:
        """Single * atom (attachment point) is NOT a polymer repeat unit."""
        from chemscheme.renderers.drawsvg_composer import _is_polymer_unit

        assert _is_polymer_unit("*c1ccccc1") is False  # R-group, not polymer

    def test_two_dummies_is_polymer(self) -> None:
        """Two * atoms is a polymer repeat unit."""
        from chemscheme.renderers.drawsvg_composer import _is_polymer_unit

        assert _is_polymer_unit("*CC*") is True
        assert _is_polymer_unit("*c1cc(*)sc1") is True

    def test_single_dummy_no_brackets_in_svg(self) -> None:
        """SVG for single-dummy molecule has no polymer brackets."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["*c1ccccc1"],
            "products": [],
            "arrow_above": "",
        }
        svg = compose_step_svg(step).decode("utf-8")
        assert "polymer-bracket" not in svg


class TestG2OutputPathDirectory:
    """G2 (P2): CLI rejects directory as output path."""

    def test_draw_rejects_directory_output(self, tmp_path: Path) -> None:
        """draw command errors when output path is a directory."""
        from click.testing import CliRunner

        from chemscheme.cli import draw

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "test",
                }
            ],
        }
        scheme_file = tmp_path / "test.json"
        scheme_file.write_text(json.dumps(scheme))

        runner = CliRunner()
        result = runner.invoke(draw, [str(scheme_file), "-o", str(tmp_path)])
        assert result.exit_code != 0
        assert "directory" in result.output.lower() or "directory" in (result.stderr or "").lower()
