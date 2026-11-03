"""MR10 review regression tests.

Covers fixes for G1, C2, C4, C6 findings from Phase 11 review.
"""

# RED-EXEMPT: review finding regressions

import pytest

# ===========================================================================
# MR10-G1: _build_arrow_lines filters empty strings from \n split
# ===========================================================================


@pytest.mark.unit
class TestMR10G1EmptyLineSplit:
    """Empty lines from \\n split should be filtered out."""

    def test_double_newline_no_empty_lines(self) -> None:
        """GIVEN arrow_above='Line1\\n\\nLine2', WHEN _build_arrow_lines, THEN 2 lines."""
        from chemscheme.renderers.drawsvg_composer import _build_arrow_lines

        step = {"arrow_above": "Line1\n\nLine2"}
        lines = _build_arrow_lines(step)
        assert lines == ["Line1", "Line2"]

    def test_trailing_newline_no_empty(self) -> None:
        """GIVEN arrow_above='Line1\\n', WHEN _build_arrow_lines, THEN 1 line."""
        from chemscheme.renderers.drawsvg_composer import _build_arrow_lines

        step = {"arrow_above": "Line1\n"}
        lines = _build_arrow_lines(step)
        assert lines == ["Line1"]


# ===========================================================================
# MR10-C2: PNG path handles list-form arrow_above
# ===========================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestMR10C2PngListArrowAbove:
    """PNG render should not crash on list-form arrow_above."""

    def test_list_arrow_above_png_no_crash(self) -> None:
        """GIVEN step with arrow_above=['a', 'b'], WHEN render_step(png), THEN valid PNG."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": ["Pd(PPh3)4", "K2CO3"],
        }
        result = render_step(step, fmt="png")
        assert result[:4] == b"\x89PNG"


# ===========================================================================
# MR10-C4: compound_labels schema validation
# ===========================================================================


@pytest.mark.unit
class TestMR10C4CompoundLabelsSchema:
    """compound_labels should be validated as dict[str, str]."""

    def test_valid_compound_labels_passes(self) -> None:
        """GIVEN step with valid compound_labels dict, WHEN validate, THEN no errors."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "title": "test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "test",
                    "compound_labels": {"C": "1", "CC": "2"},
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert not errors

    def test_non_dict_compound_labels_rejected(self) -> None:
        """GIVEN step with compound_labels='bad', WHEN validate, THEN error."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "title": "test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "test",
                    "compound_labels": "bad",
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert any("compound_labels" in e and "dict" in e for e in errors)

    def test_non_string_value_rejected(self) -> None:
        """GIVEN compound_labels with int value, WHEN validate, THEN error."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "title": "test",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "test",
                    "compound_labels": {"C": 1},
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert any("compound_labels value" in e for e in errors)


# ===========================================================================
# MR10-C6: temperature=0 / time=0 not treated as absent
# ===========================================================================


@pytest.mark.unit
class TestMR10C6ZeroValueConditions:
    """Numeric 0 values for temperature/time should not be treated as absent."""

    def test_temperature_zero_rendered(self) -> None:
        """GIVEN step with temperature=0, WHEN _build_arrow_lines, THEN '0' in output."""
        from chemscheme.renderers.drawsvg_composer import _build_arrow_lines

        step = {"temperature": 0, "arrow_above": "fallback"}
        lines = _build_arrow_lines(step)
        assert any("0" in line for line in lines), f"temperature=0 should appear: {lines}"

    def test_time_zero_rendered(self) -> None:
        """GIVEN step with time=0, WHEN _build_arrow_lines, THEN '0' in output."""
        from chemscheme.renderers.drawsvg_composer import _build_arrow_lines

        step = {"time": 0, "arrow_above": "fallback"}
        lines = _build_arrow_lines(step)
        assert any("0" in line for line in lines), f"time=0 should appear: {lines}"

    def test_zero_temp_does_not_fall_back(self) -> None:
        """GIVEN temperature=0 + arrow_above, WHEN _build_arrow_lines, THEN no fallback."""
        from chemscheme.renderers.drawsvg_composer import _build_arrow_lines

        step = {"temperature": 0, "arrow_above": "should not appear"}
        lines = _build_arrow_lines(step)
        assert "should not appear" not in " ".join(lines)
