"""Tests for Phase 11 — Publication Annotations & Typography.

BDD Scenarios:

AC-11.1: Compound number labels rendered below molecules
  GIVEN step with compound_labels={"c1ccccc1": "1", "c1ccc(O)cc1": "2"},
  WHEN compose_step_svg(step),
  THEN SVG contains label text "1" and "2" below molecule positions

  GIVEN step without compound_labels,
  WHEN compose_step_svg(step),
  THEN no label text rendered (no crash)

  GIVEN multi-step scheme with labels,
  WHEN render_scheme(svg),
  THEN all labels visible in composed SVG

  Adversarial: empty compound_labels dict → no crash

AC-11.2: Yield field rendered below arrow
  GIVEN step with yield="85%",
  WHEN compose_step_svg(step),
  THEN SVG contains "85%" text below the arrow

  GIVEN step without yield field,
  WHEN compose_step_svg(step),
  THEN no yield text rendered

  GIVEN step with yield="95%" and arrow_below="DMF",
  WHEN compose_step_svg(step),
  THEN both yield and arrow_below are visible (not overlapping)

AC-11.3: Arrow text multi-line + adaptive font size
  GIVEN arrow_above="Pd(PPh3)4\\nK2CO3",
  WHEN compose_step_svg(step),
  THEN SVG has two separate text lines above arrow

  GIVEN arrow_above=["Pd(PPh3)4", "K2CO3"],
  WHEN compose_step_svg(step),
  THEN SVG has two separate text lines above arrow

  GIVEN arrow_above with very long text (>50 chars),
  WHEN compose_step_svg(step),
  THEN font-size is reduced (adaptive)

  Adversarial: arrow_above=[""] (list with empty string) → no crash

AC-11.4: Structured condition fields auto-formatted
  GIVEN step with reagents=["Pd(PPh3)4"], solvents=["DMF"], temperature="80°C", time="12h",
  WHEN compose_step_svg(step),
  THEN arrow region contains all fields formatted as lines

  GIVEN step with only temperature="rt",
  WHEN compose_step_svg(step),
  THEN only temperature rendered (partial fields ok)

  GIVEN step with reagents + arrow_above both set,
  WHEN compose_step_svg(step),
  THEN structured fields take precedence over raw arrow_above

  Adversarial: all structured fields empty → no crash, falls back to arrow_above

AC-11.5: Scheme S1 end-to-end
  GIVEN a realistic 3-step scheme with labels, yields, and conditions,
  WHEN render_scheme(svg),
  THEN valid SVG with all annotations

  GIVEN same scheme,
  WHEN render_scheme(png),
  THEN valid PNG output

  GIVEN scheme with mixed annotated + bare steps,
  WHEN render_scheme(svg),
  THEN bare steps still render correctly
"""

import re
import xml.etree.ElementTree as ET

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LABELED_STEP = {
    "id": "1",
    "reactants": ["c1ccccc1"],
    "products": ["c1ccc(O)cc1"],
    "arrow_above": "H2O2",
    "compound_labels": {"c1ccccc1": "1", "c1ccc(O)cc1": "2"},
}

YIELD_STEP = {
    "id": "1",
    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
    "products": ["c1ccc(-c2ccccc2)cc1"],
    "arrow_above": "Pd(PPh3)4",
    "yield": "85%",
}

MULTILINE_STEP_NEWLINE = {
    "id": "1",
    "reactants": ["c1ccccc1"],
    "products": ["c1ccc(O)cc1"],
    "arrow_above": "Pd(PPh3)4\nK2CO3",
}

MULTILINE_STEP_LIST = {
    "id": "1",
    "reactants": ["c1ccccc1"],
    "products": ["c1ccc(O)cc1"],
    "arrow_above": ["Pd(PPh3)4", "K2CO3"],
}

STRUCTURED_STEP = {
    "id": "1",
    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
    "products": ["c1ccc(-c2ccccc2)cc1"],
    "arrow_above": "old text",
    "reagents": ["Pd(PPh3)4"],
    "solvents": ["DMF"],
    "temperature": "80°C",
    "time": "12h",
}

REALISTIC_SCHEME = {
    "schema_version": 1,
    "title": "Test Publication Scheme",
    "steps": [
        {
            "id": "1",
            "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
            "products": ["c1ccc(-c2ccccc2)cc1"],
            "arrow_above": "Pd(PPh3)4, K2CO3",
            "arrow_below": "DMF, 80°C, 12h",
            "yield": "85%",
            "compound_labels": {
                "c1ccc(Br)cc1": "1",
                "OB(O)c1ccccc1": "2",
                "c1ccc(-c2ccccc2)cc1": "3",
            },
        },
        {
            "id": "2",
            "reactants": ["c1ccc(-c2ccccc2)cc1"],
            "products": ["c1ccc(-c2ccc(O)cc2)cc1"],
            "reagents": ["mCPBA"],
            "solvents": ["DCM"],
            "temperature": "0°C",
            "time": "2h",
            "yield": "72%",
            "compound_labels": {
                "c1ccc(-c2ccc(O)cc2)cc1": "4",
            },
        },
        {
            "id": "3",
            "reactants": ["c1ccc(-c2ccc(O)cc2)cc1"],
            "products": ["c1ccc(-c2ccc(OC)cc2)cc1"],
            "arrow_above": "CH3I, K2CO3",
            "yield": "91%",
            "compound_labels": {
                "c1ccc(-c2ccc(OC)cc2)cc1": "5",
            },
        },
    ],
}


# =============================================================================
# AC-11.1: Compound number labels below molecules
# =============================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestAC111CompoundLabels:
    """Compound number labels should render below molecules."""

    def test_labeled_step_has_label_text(self) -> None:
        """GIVEN step with compound_labels, WHEN compose_step_svg, THEN labels in SVG."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(LABELED_STEP)
        # Look for text elements with data-role="compound-label"
        root = ET.fromstring(result)
        labels = root.findall(".//{http://www.w3.org/2000/svg}text[@data-role='compound-label']")
        if not labels:
            labels = root.findall(".//text[@data-role='compound-label']")
        assert len(labels) >= 2, f"Expected ≥2 compound-label text elements, got {len(labels)}"

    def test_no_labels_no_crash(self) -> None:
        """GIVEN step without compound_labels, WHEN compose_step_svg, THEN no crash."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "test",
        }
        result = compose_step_svg(step)
        assert b"<svg" in result
        # No compound-label elements
        root = ET.fromstring(result)
        labels = root.findall(".//{http://www.w3.org/2000/svg}text[@data-role='compound-label']")
        if not labels:
            labels = root.findall(".//text[@data-role='compound-label']")
        assert len(labels) == 0

    def test_empty_labels_dict_no_crash(self) -> None:
        """GIVEN step with compound_labels={}, WHEN compose_step_svg, THEN no crash."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "test",
            "compound_labels": {},
        }
        result = compose_step_svg(step)
        assert b"<svg" in result

    def test_labels_in_multi_step_scheme(self) -> None:
        """GIVEN multi-step scheme with labels, WHEN render_scheme, THEN labels visible."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
            "schema_version": 1,
            "title": "labeled",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "H2O2",
                    "compound_labels": {"c1ccccc1": "1", "c1ccc(O)cc1": "2"},
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(O)cc1"],
                    "products": ["c1ccc(OC)cc1"],
                    "arrow_above": "CH3I",
                    "compound_labels": {"c1ccc(OC)cc1": "3"},
                },
            ],
        }
        svg_bytes = render_scheme(scheme, fmt="svg")
        svg_str = svg_bytes.decode("utf-8")
        assert "compound-label" in svg_str, "Multi-step SVG should contain compound labels"


# =============================================================================
# AC-11.2: Yield field below arrow
# =============================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestAC112YieldBelowArrow:
    """Yield field should render below the reaction arrow."""

    def test_yield_rendered_in_svg(self) -> None:
        """GIVEN step with yield='85%', WHEN compose_step_svg, THEN '85%' in SVG."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(YIELD_STEP)
        svg_str = result.decode("utf-8")
        assert "85%" in svg_str, "Yield text should appear in SVG"

    def test_no_yield_no_text(self) -> None:
        """GIVEN step without yield, WHEN compose_step_svg, THEN no yield text."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "test",
        }
        result = compose_step_svg(step)
        svg_str = result.decode("utf-8")
        # Should not contain any percentage text
        assert 'data-role="yield"' not in svg_str

    def test_yield_and_arrow_below_both_visible(self) -> None:
        """GIVEN step with yield + arrow_below, WHEN composed, THEN both visible."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "reagent",
            "arrow_below": "DMF, rt",
            "yield": "95%",
        }
        result = compose_step_svg(step)
        svg_str = result.decode("utf-8")
        assert "95%" in svg_str, "Yield should be present"
        assert "DMF" in svg_str, "Arrow below text should be present"


# =============================================================================
# AC-11.3: Arrow text multi-line + adaptive font size
# =============================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestAC113MultilineArrowText:
    """Arrow text should support multi-line and adaptive font size."""

    def test_newline_produces_multiple_lines(self) -> None:
        r"""GIVEN arrow_above with \n, WHEN compose_step_svg, THEN multiple text lines."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(MULTILINE_STEP_NEWLINE)
        svg_str = result.decode("utf-8")
        # We expect "Pd(PPh3)4" and "K2CO3" on separate lines
        plain = re.sub(r"<[^>]+>", "", svg_str)
        assert "Pd" in plain and "K" in plain, "Both lines should be present"
        # Check for multiple y-positions in text elements (multiline)
        # Look for dy attributes or multiple text elements
        text_y_values = re.findall(r'<text[^>]*y="([^"]+)"', svg_str)
        # With multiline, there should be more text y-values than without
        assert len(text_y_values) >= 2, "Should have multiple text line positions"

    def test_list_produces_multiple_lines(self) -> None:
        """GIVEN arrow_above as list, WHEN compose_step_svg, THEN multiple text lines."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(MULTILINE_STEP_LIST)
        svg_str = result.decode("utf-8")
        plain = re.sub(r"<[^>]+>", "", svg_str)
        assert "Pd" in plain and "K" in plain, "Both list items should be present"

    def test_long_text_adaptive_font(self) -> None:
        """GIVEN arrow_above >50 chars, WHEN compose_step_svg, THEN smaller font."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "Pd(dppf)Cl2, Cs2CO3, 1,4-dioxane/H2O (3:1), 100 degrees, 24h reflux",
        }
        result = compose_step_svg(step)
        svg_str = result.decode("utf-8")
        # Font size should be smaller than default 11
        font_sizes = re.findall(r'font-size="([\d.]+)"', svg_str)
        # At least one font-size should be < 11 for the arrow text
        has_small = any(float(fs) < 11 for fs in font_sizes)
        assert has_small, f"Long text should trigger smaller font. Sizes found: {font_sizes}"

    def test_list_with_empty_string_no_crash(self) -> None:
        """GIVEN arrow_above=[''], WHEN compose_step_svg, THEN no crash."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": [""],
        }
        result = compose_step_svg(step)
        assert b"<svg" in result


# =============================================================================
# AC-11.4: Structured condition fields
# =============================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestAC114StructuredConditions:
    """Structured condition fields should auto-format to arrow area."""

    def test_structured_fields_rendered(self) -> None:
        """GIVEN step with reagents/solvents/temp/time, WHEN composed, THEN all present."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(STRUCTURED_STEP)
        svg_str = result.decode("utf-8")
        plain = re.sub(r"<[^>]+>", "", svg_str)
        # All structured fields should appear
        assert "Pd" in plain, "Reagent should be present"
        assert "DMF" in plain, "Solvent should be present"
        assert "80" in plain, "Temperature should be present"
        assert "12h" in plain, "Time should be present"

    def test_partial_fields_ok(self) -> None:
        """GIVEN step with only temperature, WHEN composed, THEN temp rendered."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "",
            "temperature": "rt",
        }
        result = compose_step_svg(step)
        svg_str = result.decode("utf-8")
        assert "rt" in svg_str, "Temperature should render even as only field"

    def test_structured_over_raw_arrow_above(self) -> None:
        """GIVEN step with reagents + arrow_above, WHEN composed, THEN structured wins."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        result = compose_step_svg(STRUCTURED_STEP)
        svg_str = result.decode("utf-8")
        plain = re.sub(r"<[^>]+>", "", svg_str)
        # "old text" from arrow_above should NOT appear when structured fields present
        assert "old text" not in plain, "Structured fields should replace raw arrow_above"

    def test_all_empty_structured_falls_back(self) -> None:
        """GIVEN empty structured fields, WHEN composed, THEN falls back to arrow_above."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(O)cc1"],
            "arrow_above": "NaOH",
            "reagents": [],
            "solvents": [],
        }
        result = compose_step_svg(step)
        svg_str = result.decode("utf-8")
        plain = re.sub(r"<[^>]+>", "", svg_str)
        assert "NaOH" in plain, "Should fall back to arrow_above when structured fields empty"


# =============================================================================
# AC-11.5: Scheme S1 end-to-end
# =============================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestAC115SchemeS1EndToEnd:
    """Realistic multi-step scheme with all annotations renders correctly."""

    def test_realistic_scheme_valid_svg(self) -> None:
        """GIVEN realistic 3-step scheme, WHEN render_scheme(svg), THEN valid SVG."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        svg_bytes = render_scheme(REALISTIC_SCHEME, fmt="svg")
        root = ET.fromstring(svg_bytes)
        assert root.tag.endswith("svg")

    def test_realistic_scheme_has_annotations(self) -> None:
        """GIVEN realistic scheme, WHEN render_scheme(svg), THEN has labels + yields."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        svg_bytes = render_scheme(REALISTIC_SCHEME, fmt="svg")
        svg_str = svg_bytes.decode("utf-8")
        # Should contain compound labels
        assert "compound-label" in svg_str, "Should have compound labels"
        # Should contain yield percentages
        assert "85%" in svg_str, "Should have yield for step 1"

    def test_realistic_scheme_png(self) -> None:
        """GIVEN realistic scheme, WHEN render_scheme(png), THEN valid PNG."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        png_bytes = render_scheme(REALISTIC_SCHEME, fmt="png")
        assert png_bytes[:4] == b"\x89PNG"

    def test_mixed_annotated_and_bare_steps(self) -> None:
        """GIVEN scheme with annotated + bare steps, WHEN render_scheme, THEN all render."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
            "schema_version": 1,
            "title": "mixed",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "H2O2",
                    "yield": "90%",
                    "compound_labels": {"c1ccccc1": "1"},
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(O)cc1"],
                    "products": ["c1ccc(OC)cc1"],
                    "arrow_above": "CH3I",
                    # No yield, no labels — bare step
                },
            ],
        }
        svg_bytes = render_scheme(scheme, fmt="svg")
        root = ET.fromstring(svg_bytes)
        assert root.tag.endswith("svg")
        # Both steps should be present
        groups = root.findall(".//{http://www.w3.org/2000/svg}g[@data-step-id]")
        if not groups:
            groups = root.findall(".//g[@data-step-id]")
        assert len(groups) == 2
