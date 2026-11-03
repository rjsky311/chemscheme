"""Tests for AC-9.4: drawsvg composer XML-escapes all text output.

Phase 9 — MR7-04 (P1).
BDD-first: special characters in arrow text must not break SVG.
"""

import xml.etree.ElementTree as ET

import pytest

from chemscheme.renderers.drawsvg_composer import compose_step_svg, format_chem_text


def _make_step(arrow_above: str = "", arrow_below: str = "") -> dict:
    """Build a minimal step dict for testing."""
    return {
        "reactants": ["CCO"],
        "products": ["CC=O"],
        "arrow_above": arrow_above,
        "arrow_below": arrow_below,
    }


@pytest.mark.unit
class TestDrawsvgXmlEscaping:
    """AC-9.4: Text in SVG output is properly XML-escaped."""

    def test_ampersand_in_arrow_above(self) -> None:
        """Arrow text with '&' must produce valid SVG XML."""
        step = _make_step(arrow_above="Et2O & THF")
        svg_bytes = compose_step_svg(step)
        # Must parse as valid XML
        root = ET.fromstring(svg_bytes)
        assert root is not None

        # The text content should contain the ampersand
        svg_text = svg_bytes.decode("utf-8")
        assert "&amp;" in svg_text or "Et2O &amp; THF" in svg_text

    def test_less_than_in_arrow_below(self) -> None:
        """Arrow text with '<' must produce valid SVG XML."""
        step = _make_step(arrow_below="<5 °C")
        svg_bytes = compose_step_svg(step)
        root = ET.fromstring(svg_bytes)
        assert root is not None

    def test_greater_than_in_conditions(self) -> None:
        """Arrow text with '>' must produce valid SVG XML."""
        step = _make_step(arrow_above=">90% yield")
        svg_bytes = compose_step_svg(step)
        root = ET.fromstring(svg_bytes)
        assert root is not None

    def test_quotes_in_text(self) -> None:
        """Arrow text with quotes must produce valid SVG XML."""
        step = _make_step(arrow_above='"reflux"')
        svg_bytes = compose_step_svg(step)
        root = ET.fromstring(svg_bytes)
        assert root is not None

    def test_combined_special_chars(self) -> None:
        """Multiple special chars: K2CO3 & Et2O, <50 °C → valid SVG."""
        step = _make_step(
            arrow_above="K2CO3 & Et2O",
            arrow_below="<50 °C, >2h",
        )
        svg_bytes = compose_step_svg(step)
        root = ET.fromstring(svg_bytes)
        assert root is not None

    def test_normal_text_unchanged(self) -> None:
        """Normal chemistry text (no special chars) still works."""
        step = _make_step(arrow_above="Pd(PPh3)4, K2CO3")
        svg_bytes = compose_step_svg(step)
        root = ET.fromstring(svg_bytes)
        assert root is not None


@pytest.mark.unit
class TestFormatChemTextEscaping:
    """format_chem_text must handle XML special chars in input."""

    def test_ampersand_escaped_in_output(self) -> None:
        """Ampersand in input → &amp; in output (valid XML)."""
        result = format_chem_text("K2CO3 & NaOH")
        # The result is inserted into SVG Raw — must not break XML
        assert "&" in result
        # Wrap in XML to test validity
        test_xml = f"<text>{result}</text>"
        ET.fromstring(test_xml)  # must not raise

    def test_less_than_escaped(self) -> None:
        """'<' in input → &lt; in output."""
        result = format_chem_text("<5 equiv")
        test_xml = f"<text>{result}</text>"
        ET.fromstring(test_xml)

    def test_adversarial_all_special_chars(self) -> None:
        """Adversarial: all XML special chars at once."""
        result = format_chem_text('A & B < C > D "E"')
        test_xml = f"<text>{result}</text>"
        ET.fromstring(test_xml)
