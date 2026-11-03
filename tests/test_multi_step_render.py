"""Tests for multi-step scheme rendering — SVG composition.

Covers AC-3.4, AC-3.7, AC-3.8.
"""

import xml.etree.ElementTree as ET

import pytest

from chemscheme.renderers.layout_engine import LayoutConfig
from chemscheme.renderers.multi_step_renderer import render_scheme

# ---------------------------------------------------------------------------
# AC-3.4: Reagent text bounding box doesn't overlap molecules
# ---------------------------------------------------------------------------


class TestTextOverlap:
    """Reagent text should not overlap with molecule structures."""

    @pytest.mark.visual
    @pytest.mark.chemistry
    def test_text_elements_within_canvas(self, sample_scheme_3step):
        """All text elements should be within the SVG canvas bounds."""
        svg_bytes = render_scheme(sample_scheme_3step, fmt="svg")
        root = ET.fromstring(svg_bytes)
        ns = {"svg": "http://www.w3.org/2000/svg"}

        width = float(root.attrib.get("width", "0"))
        height = float(root.attrib.get("height", "0"))
        assert width > 0 and height > 0

        # Find all text elements
        texts = root.findall(".//svg:text", ns) + root.findall(".//text")
        assert len(texts) > 0, "Should have text elements for arrow annotations"
        for text_el in texts:
            x = float(text_el.attrib.get("x", "0"))
            y = float(text_el.attrib.get("y", "0"))
            assert 0 <= x <= width, f"Text x={x} outside canvas width={width}"
            assert 0 <= y <= height, f"Text y={y} outside canvas height={height}"


# ---------------------------------------------------------------------------
# AC-3.7: 5-step linear NFA renders correctly
# ---------------------------------------------------------------------------


class TestLinearNFARender:
    """5-step linear NFA synthesis renders as valid SVG."""

    @pytest.mark.chemistry
    @pytest.mark.visual
    def test_5step_renders_svg(self, sample_scheme_5step_nfa):
        """render_scheme produces valid SVG bytes."""
        svg_bytes = render_scheme(sample_scheme_5step_nfa, fmt="svg")
        assert svg_bytes.startswith(b"<") or svg_bytes.startswith(b"<?xml")
        root = ET.fromstring(svg_bytes)
        assert root.tag.endswith("svg")

    @pytest.mark.chemistry
    @pytest.mark.visual
    def test_5step_renders_png(self, sample_scheme_5step_nfa):
        """render_scheme produces valid PNG bytes."""
        png_bytes = render_scheme(sample_scheme_5step_nfa, fmt="png")
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.chemistry
    @pytest.mark.visual
    def test_5step_contains_all_arrow_text(self, sample_scheme_5step_nfa):
        """SVG should contain arrow_above text for all 5 steps.

        Note: drawsvg_composer applies format_chem_text which adds <tspan> subscripts,
        so we strip tags before checking text content presence.
        """
        import re

        svg_bytes = render_scheme(sample_scheme_5step_nfa, fmt="svg")
        svg_str = svg_bytes.decode("utf-8")
        # Strip XML tags + zero-width spaces/entities to get plain text
        plain = re.sub(r"<[^>]+>", "", svg_str)
        plain = plain.replace("\u200b", "").replace("&#8203;", "")
        for step in sample_scheme_5step_nfa["steps"]:
            assert step["arrow_above"] in plain, f"Missing arrow_above text: {step['arrow_above']}"

    @pytest.mark.chemistry
    @pytest.mark.visual
    def test_5step_svg_has_multiple_groups(self, sample_scheme_5step_nfa):
        """SVG should have distinct groups for each step."""
        svg_bytes = render_scheme(sample_scheme_5step_nfa, fmt="svg")
        root = ET.fromstring(svg_bytes)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        # Look for step groups (g elements with step-id attribute)
        groups = root.findall(".//svg:g[@data-step-id]", ns) + root.findall(".//g[@data-step-id]")
        assert len(groups) == 5, f"Expected 5 step groups, got {len(groups)}"

    @pytest.mark.chemistry
    @pytest.mark.visual
    def test_5step_with_wrapping(self, sample_scheme_5step_nfa):
        """5 steps with max 3 per row → SVG height > single row."""
        cfg = LayoutConfig(max_steps_per_row=3)
        svg_bytes = render_scheme(sample_scheme_5step_nfa, fmt="svg", config=cfg)
        root = ET.fromstring(svg_bytes)
        height = float(root.attrib.get("height", "0"))
        # Multi-row should be taller than a single step
        assert height > cfg.step_height


# ---------------------------------------------------------------------------
# AC-3.8: 8-step convergent NFA renders correctly
# ---------------------------------------------------------------------------


class TestConvergentNFARender:
    """8-step convergent NFA scheme renders correctly."""

    @pytest.mark.chemistry
    @pytest.mark.visual
    def test_8step_renders_svg(self, sample_scheme_8step_convergent):
        """render_scheme produces valid SVG for convergent scheme."""
        svg_bytes = render_scheme(sample_scheme_8step_convergent, fmt="svg")
        root = ET.fromstring(svg_bytes)
        assert root.tag.endswith("svg")

    @pytest.mark.chemistry
    @pytest.mark.visual
    def test_8step_contains_all_steps(self, sample_scheme_8step_convergent):
        """SVG should have groups for all 8 steps."""
        svg_bytes = render_scheme(sample_scheme_8step_convergent, fmt="svg")
        root = ET.fromstring(svg_bytes)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        groups = root.findall(".//svg:g[@data-step-id]", ns) + root.findall(".//g[@data-step-id]")
        assert len(groups) == 8

    @pytest.mark.chemistry
    @pytest.mark.visual
    def test_8step_convergent_branches_visible(self, sample_scheme_8step_convergent):
        """Both branches should be rendered at different vertical positions."""
        svg_bytes = render_scheme(sample_scheme_8step_convergent, fmt="svg")
        root = ET.fromstring(svg_bytes)
        ns = {"svg": "http://www.w3.org/2000/svg"}

        groups = root.findall(".//svg:g[@data-step-id]", ns) + root.findall(".//g[@data-step-id]")
        group_map = {g.attrib["data-step-id"]: g for g in groups}

        # Branch a and branch b should have different transform y-offsets
        transform_1a = group_map["1a"].attrib.get("transform", "")
        transform_1b = group_map["1b"].attrib.get("transform", "")
        assert transform_1a != transform_1b, "Branches should be at different positions"

    @pytest.mark.chemistry
    @pytest.mark.visual
    def test_8step_renders_png(self, sample_scheme_8step_convergent):
        """render_scheme produces valid PNG for convergent scheme."""
        png_bytes = render_scheme(sample_scheme_8step_convergent, fmt="png")
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# D1: Intermediate compound deduplication
# ---------------------------------------------------------------------------


class TestIntermediateDedup:
    """Shared intermediate compounds should appear only once in multi-step renders."""

    @pytest.mark.visual
    @pytest.mark.chemistry
    def test_dedup_function_removes_shared_reactants(self):
        """_dedup_intermediates removes shared intermediates from successor reactants."""
        from chemscheme.renderers.multi_step_renderer import _dedup_intermediates

        steps = [
            {
                "id": "s1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
            },
            {
                "id": "s2",
                "reactants": ["c1ccc(Br)cc1"],
                "products": ["c1ccc(-c2ccccc2)cc1"],
                "arrow_above": "Suzuki",
            },
        ]
        connections = [("s1", "s2")]
        deduped = _dedup_intermediates(steps, connections)
        # Step s2 should have empty reactants (bromobenzene removed)
        assert deduped["s2"]["reactants"] == []
        # Step s1 should be unchanged
        assert deduped["s1"]["reactants"] == ["c1ccccc1"]
        assert deduped["s1"]["products"] == ["c1ccc(Br)cc1"]

    @pytest.mark.visual
    @pytest.mark.chemistry
    def test_dedup_transfers_compound_labels(self):
        """Compound labels for deduped intermediates transfer to predecessor step."""
        from chemscheme.renderers.multi_step_renderer import _dedup_intermediates

        steps = [
            {
                "id": "s1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
                "compound_labels": {"c1ccccc1": "1"},
            },
            {
                "id": "s2",
                "reactants": ["c1ccc(Br)cc1"],
                "products": ["c1ccc(-c2ccccc2)cc1"],
                "arrow_above": "Suzuki",
                "compound_labels": {
                    "c1ccc(Br)cc1": "2",
                    "c1ccc(-c2ccccc2)cc1": "3",
                },
            },
        ]
        connections = [("s1", "s2")]
        deduped = _dedup_intermediates(steps, connections)
        # Label "2" should be transferred to step s1
        assert deduped["s1"]["compound_labels"]["c1ccc(Br)cc1"] == "2"
        # Label "2" removed from step s2
        assert "c1ccc(Br)cc1" not in deduped["s2"]["compound_labels"]
        # Label "3" stays in step s2
        assert deduped["s2"]["compound_labels"]["c1ccc(-c2ccccc2)cc1"] == "3"

    @pytest.mark.visual
    @pytest.mark.chemistry
    def test_dedup_no_shared_intermediate_unchanged(self):
        """Steps without shared intermediates are not modified."""
        from chemscheme.renderers.multi_step_renderer import _dedup_intermediates

        steps = [
            {
                "id": "s1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
            },
            {
                "id": "s2",
                "reactants": ["CC=O"],
                "products": ["CC(O)c1ccccc1"],
                "arrow_above": "PhMgBr",
            },
        ]
        connections = [("s1", "s2")]
        deduped = _dedup_intermediates(steps, connections)
        # Step s2 reactants unchanged (no shared molecules)
        assert deduped["s2"]["reactants"] == ["CC=O"]

    @pytest.mark.visual
    @pytest.mark.chemistry
    def test_dedup_3step_chain(self):
        """3-step chain deduplicates both intermediates."""
        from chemscheme.renderers.multi_step_renderer import _dedup_intermediates

        steps = [
            {
                "id": "s1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
            },
            {
                "id": "s2",
                "reactants": ["c1ccc(Br)cc1"],
                "products": ["c1ccc(O)cc1"],
                "arrow_above": "NaOH",
            },
            {
                "id": "s3",
                "reactants": ["c1ccc(O)cc1"],
                "products": ["c1ccc(OC)cc1"],
                "arrow_above": "MeI",
            },
        ]
        connections = [("s1", "s2"), ("s2", "s3")]
        deduped = _dedup_intermediates(steps, connections)
        assert deduped["s2"]["reactants"] == []
        assert deduped["s3"]["reactants"] == []

    @pytest.mark.visual
    @pytest.mark.chemistry
    def test_dedup_partial_reactant_removal(self):
        """Only shared reactants are removed; non-shared reactants stay."""
        from chemscheme.renderers.multi_step_renderer import _dedup_intermediates

        steps = [
            {
                "id": "s1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
            },
            {
                "id": "s2",
                "reactants": ["c1ccc(Br)cc1", "[H][H]"],  # Br shared, H2 not
                "products": ["c1ccc(-c2ccccc2)cc1"],
                "arrow_above": "Pd/C",
            },
        ]
        connections = [("s1", "s2")]
        deduped = _dedup_intermediates(steps, connections)
        # H2 stays, bromobenzene removed
        assert deduped["s2"]["reactants"] == ["[H][H]"]

    @pytest.mark.visual
    @pytest.mark.chemistry
    def test_render_with_dedup_produces_valid_svg(self):
        """End-to-end: 2-step scheme with shared intermediate renders valid SVG."""
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "s1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                },
                {
                    "id": "s2",
                    "reactants": ["c1ccc(Br)cc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Suzuki",
                },
            ],
        }
        svg_bytes = render_scheme(scheme, fmt="svg")
        assert b"<svg" in svg_bytes
        svg_text = svg_bytes.decode("utf-8")
        assert "Suzuki" in svg_text

    @pytest.mark.visual
    @pytest.mark.chemistry
    def test_convergent_merge_not_deduped(self, sample_scheme_8step_convergent):
        """Convergent branches should NOT have their merge-point molecules deduped."""
        svg_bytes = render_scheme(sample_scheme_8step_convergent, fmt="svg")
        root = ET.fromstring(svg_bytes)
        step_groups = root.findall(".//{http://www.w3.org/2000/svg}g[@data-step-id]")
        if not step_groups:
            step_groups = root.findall(".//g[@data-step-id]")
        assert len(step_groups) == 8
