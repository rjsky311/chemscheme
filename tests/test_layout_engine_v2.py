"""Tests for Layout Engine v2 improvements.

BDD Scenarios:
  P3-a: Adaptive box sizing
    GIVEN a scheme with steps of varying molecular complexity,
    WHEN compute_layout is called with per-step size overrides,
    THEN boxes should have the specified widths/heights.

    GIVEN a scheme with step_sizes for some steps,
    WHEN compute_layout is called,
    THEN steps without overrides use config defaults.

    GIVEN a scheme with adaptive sizes and row wrapping,
    WHEN compute_layout is called,
    THEN boxes should not overlap horizontally within a row.

  P3-b: Arrow routing (cross-row L-shape connectors)
    GIVEN a 5-step scheme that wraps to 2 rows (step 3 → step 4),
    WHEN rendered to SVG,
    THEN the connector between step 3 and step 4 should be a polyline (L-shape),
         not a straight diagonal line.

    GIVEN two steps on different rows with x1 < x2,
    WHEN rendered to SVG,
    THEN the connector should still be L-shaped (not diagonal).

  Adversarial:
    GIVEN step_sizes with an unknown step_id,
    WHEN compute_layout is called,
    THEN the unknown entry is silently ignored (no crash).
"""

import re

import pytest

from chemscheme.renderers.layout_engine import LayoutConfig, compute_layout

# ---------------------------------------------------------------------------
# P3-a: Adaptive box sizing via step_sizes parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdaptiveBoxSizing:
    """compute_layout should accept step_sizes to override per-step dimensions."""

    def test_step_sizes_override_width(self) -> None:
        """Steps with explicit sizes should use those dimensions."""
        scheme = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc2c(c1)cc1c3cc(Br)sc3-c3sc(Br)cc3c1c2=O"],
                    "products": ["c1ccccc1"],
                    "arrow_above": "",
                },
            ],
        }
        step_sizes = {"1": (200.0, 180.0), "2": (450.0, 200.0)}
        result = compute_layout(scheme, step_sizes=step_sizes)
        box_map = {b.step_id: b for b in result.boxes}
        assert box_map["1"].width == 200.0
        assert box_map["1"].height == 180.0
        assert box_map["2"].width == 450.0
        assert box_map["2"].height == 200.0

    def test_step_sizes_partial_override(self) -> None:
        """Steps without size overrides fall back to config defaults."""
        scheme = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "",
                },
            ],
        }
        cfg = LayoutConfig(step_width=300.0, step_height=200.0)
        step_sizes = {"1": (180.0, 180.0)}  # only step 1 overridden
        result = compute_layout(scheme, config=cfg, step_sizes=step_sizes)
        box_map = {b.step_id: b for b in result.boxes}
        assert box_map["1"].width == 180.0
        assert box_map["2"].width == 300.0  # config default

    def test_adaptive_sizes_no_overlap(self) -> None:
        """Boxes with different widths should not overlap horizontally."""
        scheme = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "",
                },
                {
                    "id": "3",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "",
                },
            ],
        }
        step_sizes = {"1": (180.0, 200.0), "2": (400.0, 200.0), "3": (250.0, 200.0)}
        cfg = LayoutConfig(h_gap=40.0)
        result = compute_layout(scheme, config=cfg, step_sizes=step_sizes)
        boxes = sorted(result.boxes, key=lambda b: b.x)
        for i in range(len(boxes) - 1):
            assert boxes[i].right + cfg.h_gap <= boxes[i + 1].x + 0.01, (
                f"Box {boxes[i].step_id} (right={boxes[i].right}) overlaps "
                f"box {boxes[i + 1].step_id} (x={boxes[i + 1].x})"
            )

    def test_canvas_width_reflects_adaptive_sizes(self) -> None:
        """Canvas width should account for actual box widths, not just config defaults."""
        scheme = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "",
                },
            ],
        }
        step_sizes = {"1": (200.0, 200.0), "2": (500.0, 200.0)}
        result = compute_layout(scheme, step_sizes=step_sizes)
        # Canvas must be wide enough for both boxes
        for box in result.boxes:
            assert box.right <= result.canvas_width

    def test_unknown_step_id_in_sizes_ignored(self) -> None:
        """step_sizes with unknown step IDs should be silently ignored."""
        scheme = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "",
                },
            ],
        }
        step_sizes = {"1": (200.0, 200.0), "nonexistent": (999.0, 999.0)}
        result = compute_layout(scheme, step_sizes=step_sizes)
        assert len(result.boxes) == 1
        assert result.boxes[0].width == 200.0


# ---------------------------------------------------------------------------
# P3-b: Arrow routing — L-shape for cross-row connectors
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.chemistry
class TestArrowRoutingCrossRow:
    """Cross-row connectors should use L-shaped polylines, not diagonal lines."""

    def test_cross_row_connector_is_polyline(self) -> None:
        """Step 3 → step 4 wraps to a new row; connector should be polyline, not line."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
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
                    "products": ["c1ccc(B(O)O)cc1"],
                    "arrow_above": "BuLi",
                },
                {
                    "id": "3",
                    "reactants": ["c1ccc(B(O)O)cc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd cat",
                },
                {
                    "id": "4",
                    "reactants": ["c1ccc(-c2ccccc2)cc1"],
                    "products": ["c1ccc(-c2ccc(O)cc2)cc1"],
                    "arrow_above": "H2O2",
                },
            ],
        }
        cfg = LayoutConfig(max_steps_per_row=3)
        svg_bytes = render_scheme(scheme, fmt="svg", config=cfg)
        svg = svg_bytes.decode("utf-8")

        # The connection from step 3 (end of row 1) to step 4 (start of row 2)
        # crosses rows, so it should be a polyline (L-shape), NOT a straight <line>
        # Find all inter-step connectors (with marker-end="url(#arrowhead)")
        connectors = re.findall(r"<(line|polyline)[^>]*marker-end[^>]*/>", svg)
        # At least one connector should be a polyline (the row-wrap one)
        polylines = [c for c in connectors if c.startswith("polyline")]
        assert len(polylines) >= 1, (
            f"Expected at least one polyline connector for cross-row routing, "
            f"but found only: {connectors}"
        )

    def test_convergent_branch_to_merge_is_polyline(self) -> None:
        """Convergent: branch steps at different y → merge step. Should be polyline."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
            "steps": [
                {
                    "id": "1a",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "NBS",
                    "branch_label": "A",
                },
                {
                    "id": "1b",
                    "reactants": ["c1ccncc1"],
                    "products": ["c1ccnc(Br)c1"],
                    "arrow_above": "NBS",
                    "branch_label": "B",
                },
                {
                    "id": "2",
                    "consumes_step_ids": ["1a", "1b"],
                    "reactants": ["c1ccc(Br)cc1", "c1ccnc(Br)c1"],
                    "products": ["c1ccc(-c2ccncc2)cc1"],
                    "arrow_above": "Pd cat",
                },
            ],
        }
        svg_bytes = render_scheme(scheme, fmt="svg")
        svg = svg_bytes.decode("utf-8")

        # Branch 1a and 1b are at different y values; their connectors to merge
        # step 2 cross rows (different y), so should be polylines not diagonal lines
        connectors = re.findall(r"<(line|polyline)[^>]*marker-end[^>]*/>", svg)
        polylines = [c for c in connectors if c.startswith("polyline")]
        assert len(polylines) >= 1, (
            f"Convergent branch-to-merge connectors should be polylines when y differs, "
            f"got: {connectors}"
        )

    def test_same_row_connector_is_straight_line(self) -> None:
        """Connectors between steps on the same row should remain straight lines."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
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
                    "arrow_above": "Suzuki",
                },
            ],
        }
        cfg = LayoutConfig(max_steps_per_row=3)
        svg_bytes = render_scheme(scheme, fmt="svg", config=cfg)
        svg = svg_bytes.decode("utf-8")

        # Same-row connector should be a <line>, not <polyline>
        connectors = re.findall(r"<(line|polyline)[^>]*marker-end[^>]*/>", svg)
        assert all(c.startswith("line") for c in connectors), (
            f"Same-row connectors should be straight lines, got: {connectors}"
        )
