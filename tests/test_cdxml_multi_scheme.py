"""Tests for CDXML multi-scheme import and arrow text extraction.

Covers:
- E2E-6: Multi-scheme CDXML files (ChemDraw uses multiple <scheme> elements)
- Arrow text proximity extraction (arrow_above/below from <t> near arrows)
- Backward compatibility (ChemScheme attributes take priority)
"""

from chemscheme.renderers.cdxml_writer import cdxml_to_scheme


def _minimal_cdxml(body: str) -> str:
    """Wrap CDXML body in root + page."""
    return f'<?xml version="1.0" encoding="UTF-8"?><CDXML><page id="1">{body}</page></CDXML>'


def _fragment(fid: str, smiles: str) -> str:
    """Create a fragment with ChemSchemeSMILES (round-trip friendly)."""
    return f'<fragment id="{fid}" ChemSchemeSMILES="{smiles}"></fragment>'


def _arrow(aid: str, tx: float, ty: float, hx: float, hy: float) -> str:
    """Create an arrow with Head3D/Tail3D."""
    return f'<arrow id="{aid}" Head3D="{hx} {hy} 0" Tail3D="{tx} {ty} 0"/>'


def _text(x: float, y: float, content: str, tid: str = "") -> str:
    """Create a page-level text element."""
    id_attr = f' id="{tid}"' if tid else ""
    return f'<t{id_attr} p="{x} {y}"><s>{content}</s></t>'


def _step(
    reactants: str,
    products: str,
    arrow: str,
    step_id: str = "",
    arrow_above: str = "",
) -> str:
    """Create a <step> element."""
    attrs = (
        f'ReactionStepReactants="{reactants}" '
        f'ReactionStepProducts="{products}" '
        f'ReactionStepArrows="{arrow}"'
    )
    if step_id:
        attrs += f' ChemSchemeStepId="{step_id}"'
    if arrow_above:
        attrs += f' ChemSchemeArrowAbove="{arrow_above}"'
    return f"<step {attrs}/>"


class TestMultiSchemeImport:
    """CDXML files with multiple <scheme> elements should merge all steps."""

    def test_two_schemes_merged(self):
        """Steps from both schemes are imported."""
        body = (
            _fragment("10", "C")
            + _fragment("11", "CC")
            + _fragment("20", "CCC")
            + _fragment("21", "CCCC")
            + _arrow("A1", 100, 200, 150, 200)
            + _arrow("A2", 300, 200, 350, 200)
            + '<scheme id="S1">'
            + _step("10", "11", "A1", step_id="1")
            + "</scheme>"
            + '<scheme id="S2">'
            + _step("20", "21", "A2", step_id="2")
            + "</scheme>"
        )
        scheme = cdxml_to_scheme(_minimal_cdxml(body))
        assert len(scheme["steps"]) == 2
        assert scheme["steps"][0]["id"] == "1"
        assert scheme["steps"][1]["id"] == "2"

    def test_four_steps_across_two_schemes(self):
        """4 steps across 2 schemes (2+2) all imported in order."""
        frags = "".join(_fragment(str(i), "C" * i) for i in range(10, 18))
        arrows = "".join(_arrow(f"A{i}", i * 100, 200, i * 100 + 50, 200) for i in range(1, 5))
        schemes = (
            '<scheme id="S1">'
            + _step("10", "11", "A1", step_id="1")
            + _step("12", "13", "A2", step_id="2")
            + "</scheme>"
            + '<scheme id="S2">'
            + _step("14", "15", "A3", step_id="3")
            + _step("16", "17", "A4", step_id="4")
            + "</scheme>"
        )
        scheme = cdxml_to_scheme(_minimal_cdxml(frags + arrows + schemes))
        assert len(scheme["steps"]) == 4
        assert [s["id"] for s in scheme["steps"]] == ["1", "2", "3", "4"]

    def test_single_scheme_still_works(self):
        """Backward compat: single scheme files still import correctly."""
        body = (
            _fragment("10", "C")
            + _fragment("11", "CC")
            + _arrow("A1", 100, 200, 150, 200)
            + '<scheme id="S1">'
            + _step("10", "11", "A1", step_id="1")
            + "</scheme>"
        )
        scheme = cdxml_to_scheme(_minimal_cdxml(body))
        assert len(scheme["steps"]) == 1


class TestArrowTextExtraction:
    """Text elements near arrows should be extracted as arrow_above/below."""

    def test_text_above_arrow(self):
        """Text positioned above arrow → arrow_above."""
        body = (
            _fragment("10", "C")
            + _fragment("11", "CC")
            + _arrow("A1", 400, 200, 450, 200)
            + _text(410, 180, "NBS")  # above: y=180 < arrow y=200
            + '<scheme id="S1">'
            + _step("10", "11", "A1", step_id="1")
            + "</scheme>"
        )
        scheme = cdxml_to_scheme(_minimal_cdxml(body))
        assert "NBS" in scheme["steps"][0]["arrow_above"]

    def test_text_below_arrow(self):
        """Text positioned below arrow → arrow_below."""
        body = (
            _fragment("10", "C")
            + _fragment("11", "CC")
            + _arrow("A1", 400, 200, 450, 200)
            + _text(410, 220, "THF, rt")  # below: y=220 > arrow y=200
            + '<scheme id="S1">'
            + _step("10", "11", "A1", step_id="1")
            + "</scheme>"
        )
        scheme = cdxml_to_scheme(_minimal_cdxml(body))
        assert scheme["steps"][0].get("arrow_below") == "THF, rt"

    def test_text_above_and_below(self):
        """Both above and below text extracted."""
        body = (
            _fragment("10", "C")
            + _fragment("11", "CC")
            + _arrow("A1", 400, 200, 450, 200)
            + _text(410, 175, "Pd catalyst")
            + _text(410, 225, "Toluene, reflux")
            + '<scheme id="S1">'
            + _step("10", "11", "A1", step_id="1")
            + "</scheme>"
        )
        scheme = cdxml_to_scheme(_minimal_cdxml(body))
        assert "Pd catalyst" in scheme["steps"][0]["arrow_above"]
        assert scheme["steps"][0].get("arrow_below") == "Toluene, reflux"

    def test_distant_text_not_associated(self):
        """Text far from arrow (>60pt Y or outside X range) is NOT associated."""
        body = (
            _fragment("10", "C")
            + _fragment("11", "CC")
            + _arrow("A1", 400, 200, 450, 200)
            + _text(410, 50, "Far above")  # 150pt above — too far
            + _text(100, 200, "Far left")  # x=100, arrow x=400-450 — too far
            + _text(410, 350, "Far below")  # 150pt below — too far
            + '<scheme id="S1">'
            + _step("10", "11", "A1", step_id="1")
            + "</scheme>"
        )
        scheme = cdxml_to_scheme(_minimal_cdxml(body))
        assert scheme["steps"][0]["arrow_above"] == ""
        step0 = scheme["steps"][0]
        assert "arrow_below" not in step0 or step0.get("arrow_below") == ""

    def test_chemscheme_attrs_take_priority(self):
        """ChemScheme attributes override proximity text extraction."""
        body = (
            _fragment("10", "C")
            + _fragment("11", "CC")
            + _arrow("A1", 400, 200, 450, 200)
            + _text(410, 180, "Proximity text")
            + '<scheme id="S1">'
            + _step("10", "11", "A1", step_id="1", arrow_above="Stored text")
            + "</scheme>"
        )
        scheme = cdxml_to_scheme(_minimal_cdxml(body))
        assert scheme["steps"][0]["arrow_above"] == "Stored text"


class TestMultiSchemeWithTextExtraction:
    """Combined: multi-scheme + arrow text extraction on real-like data."""

    def test_two_schemes_with_conditions(self):
        """Steps from 2 schemes each get their own arrow conditions."""
        body = (
            _fragment("10", "C")
            + _fragment("11", "CC")
            + _fragment("20", "CCC")
            + _fragment("21", "CCCC")
            + _arrow("A1", 400, 150, 450, 150)
            + _arrow("A2", 400, 350, 450, 350)
            + _text(410, 130, "NBS")
            + _text(410, 170, "THF")
            + _text(410, 330, "Pd catalyst")
            + _text(410, 370, "Toluene")
            + '<scheme id="S1">'
            + _step("10", "11", "A1", step_id="1")
            + "</scheme>"
            + '<scheme id="S2">'
            + _step("20", "21", "A2", step_id="2")
            + "</scheme>"
        )
        scheme = cdxml_to_scheme(_minimal_cdxml(body))
        assert len(scheme["steps"]) == 2
        assert "NBS" in scheme["steps"][0]["arrow_above"]
        assert scheme["steps"][0].get("arrow_below") == "THF"
        assert "Pd catalyst" in scheme["steps"][1]["arrow_above"]
        assert scheme["steps"][1].get("arrow_below") == "Toluene"
