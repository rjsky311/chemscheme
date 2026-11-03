"""Tests for Phase 12 — Advanced Layout & Routing.

BDD Scenarios:

  AC-12.1: Multi-convergent layout (multiple merge points)
    GIVEN a scheme with two merge points (A,B → merge1, merge1+C → merge2),
    WHEN compute_layout is called,
    THEN all branches are Y-stacked, both merge points placed correctly,
         and tail follows after the last merge.

    GIVEN a scheme with three branches merging into one step,
    WHEN compute_layout is called,
    THEN all three branches are at distinct Y positions.

    Adversarial:
    GIVEN a scheme where a merge step consumes a non-existent step_id,
    WHEN compute_layout is called,
    THEN no crash, the orphan reference is silently ignored.

  AC-12.3: Arrow routing obstacle avoidance
    GIVEN a convergent scheme where the merge connector would cross a branch's bounding box,
    WHEN rendered to SVG,
    THEN the connector routes around the obstacle (uses waypoints, not straight line).

  AC-12.4: layout/style schema fields wired to renderer
    GIVEN a scheme with layout="horizontal",
    WHEN compute_layout is called,
    THEN no row wrapping occurs (all steps in one row regardless of count).

    GIVEN a scheme with layout="grid" (or absent),
    WHEN compute_layout is called,
    THEN default wrapping behavior applies.

    Adversarial:
    GIVEN a scheme with layout="unknown_value",
    WHEN compute_layout is called,
    THEN default wrapping behavior applies (no crash).

  AC-12.5: Convergent Y-stacking symmetric alignment
    GIVEN a convergent scheme with asymmetric branch heights (branch A: 3 steps, branch B: 1 step),
    WHEN compute_layout is called,
    THEN the merge step is vertically centered using branch midpoints (not just Y averages).

    GIVEN a convergent scheme with 3 branches of different heights,
    WHEN compute_layout is called,
    THEN merge step Y is at the midpoint of the total vertical span.
"""

import re

import pytest

from chemscheme.renderers.layout_engine import (
    LayoutConfig,
    compute_layout,
)

# ===========================================================================
# AC-12.1: Multi-convergent layout — multiple merge points
# ===========================================================================


def _make_multi_convergent_scheme():
    """A,B → merge1(step3), merge1+C → merge2(step5), then tail step6.

    Branch A: step1a → step2a
    Branch B: step1b → step2b
    Merge 1: step3 consumes [2a, 2b]
    Branch C: step4c (independent)
    Merge 2: step5 consumes [3, 4c]
    Tail: step6 (linear after merge2)
    """
    return {
        "steps": [
            {
                "id": "1a",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "NBS",
                "branch_label": "A",
            },
            {
                "id": "2a",
                "reactants": ["c1ccc(Br)cc1"],
                "products": ["c1ccc(B(O)O)cc1"],
                "arrow_above": "B2pin2",
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
                "id": "2b",
                "reactants": ["c1ccnc(Br)c1"],
                "products": ["c1ccnc(I)c1"],
                "arrow_above": "NaI",
                "branch_label": "B",
            },
            {
                "id": "3",
                "consumes_step_ids": ["2a", "2b"],
                "reactants": ["c1ccc(B(O)O)cc1", "c1ccnc(I)c1"],
                "products": ["c1ccc(-c2ccncc2)cc1"],
                "arrow_above": "Suzuki",
            },
            {
                "id": "4c",
                "reactants": ["C=O"],
                "products": ["CO"],
                "arrow_above": "NaBH4",
                "branch_label": "C",
            },
            {
                "id": "5",
                "consumes_step_ids": ["3", "4c"],
                "reactants": ["c1ccc(-c2ccncc2)cc1", "CO"],
                "products": ["c1ccc(-c2ccnc(OC)c2)cc1"],
                "arrow_above": "coupling",
            },
            {
                "id": "6",
                "reactants": ["c1ccc(-c2ccnc(OC)c2)cc1"],
                "products": ["c1ccc(-c2ccnc(O)c2)cc1"],
                "arrow_above": "deprotection",
            },
        ],
    }


def _make_triple_branch_scheme():
    """Three branches A, B, C all merging into one step."""
    return {
        "steps": [
            {
                "id": "1a",
                "reactants": ["C"],
                "products": ["CC"],
                "arrow_above": "a",
                "branch_label": "A",
            },
            {
                "id": "1b",
                "reactants": ["N"],
                "products": ["NN"],
                "arrow_above": "b",
                "branch_label": "B",
            },
            {
                "id": "1c",
                "reactants": ["O"],
                "products": ["OO"],
                "arrow_above": "c",
                "branch_label": "C",
            },
            {
                "id": "2",
                "consumes_step_ids": ["1a", "1b", "1c"],
                "reactants": ["CC", "NN", "OO"],
                "products": ["CCNNOO"],
                "arrow_above": "merge",
            },
        ],
    }


@pytest.mark.unit
class TestMultiConvergentLayout:
    """AC-12.1: Schemes with multiple merge points."""

    def test_two_merge_points_all_branches_stacked(self):
        """Branches A, B, C should all be at distinct Y positions."""
        scheme = _make_multi_convergent_scheme()
        result = compute_layout(scheme)
        box_map = {b.step_id: b for b in result.boxes}
        # Branch A and B should be at different Ys
        assert box_map["1a"].y != box_map["1b"].y
        # Branch C should be at a different Y from both
        assert box_map["4c"].y != box_map["1a"].y
        assert box_map["4c"].y != box_map["1b"].y

    def test_two_merge_points_first_merge_after_branches_ab(self):
        """Merge step 3 should be to the right of branches A and B."""
        scheme = _make_multi_convergent_scheme()
        result = compute_layout(scheme)
        box_map = {b.step_id: b for b in result.boxes}
        assert box_map["3"].x > box_map["2a"].right - 1
        assert box_map["3"].x > box_map["2b"].right - 1

    def test_two_merge_points_second_merge_after_first(self):
        """Merge step 5 should be to the right of step 3 and branch C."""
        scheme = _make_multi_convergent_scheme()
        result = compute_layout(scheme)
        box_map = {b.step_id: b for b in result.boxes}
        assert box_map["5"].x > box_map["3"].right - 1
        assert box_map["5"].x > box_map["4c"].right - 1

    def test_two_merge_points_tail_after_last_merge(self):
        """Step 6 (tail) should be to the right of merge step 5."""
        scheme = _make_multi_convergent_scheme()
        result = compute_layout(scheme)
        box_map = {b.step_id: b for b in result.boxes}
        assert box_map["6"].x > box_map["5"].right - 1

    def test_two_merge_points_connections(self):
        """All connections should be correct for multi-convergent scheme."""
        scheme = _make_multi_convergent_scheme()
        result = compute_layout(scheme)
        # Branch A
        assert ("1a", "2a") in result.connections
        # Branch B
        assert ("1b", "2b") in result.connections
        # Merge 1
        assert ("2a", "3") in result.connections
        assert ("2b", "3") in result.connections
        # Branch C standalone
        # Merge 2
        assert ("3", "5") in result.connections
        assert ("4c", "5") in result.connections
        # Tail
        assert ("5", "6") in result.connections

    def test_triple_branch_three_distinct_y(self):
        """Three branches should have three distinct Y positions."""
        scheme = _make_triple_branch_scheme()
        result = compute_layout(scheme)
        box_map = {b.step_id: b for b in result.boxes}
        ys = {box_map["1a"].y, box_map["1b"].y, box_map["1c"].y}
        assert len(ys) == 3, f"Expected 3 distinct Y positions, got {ys}"

    def test_triple_branch_merge_connections(self):
        """Merge step should consume all three branches."""
        scheme = _make_triple_branch_scheme()
        result = compute_layout(scheme)
        assert ("1a", "2") in result.connections
        assert ("1b", "2") in result.connections
        assert ("1c", "2") in result.connections

    def test_all_boxes_present(self):
        """All 8 steps should have layout boxes."""
        scheme = _make_multi_convergent_scheme()
        result = compute_layout(scheme)
        assert len(result.boxes) == 8
        ids = {b.step_id for b in result.boxes}
        assert ids == {"1a", "2a", "1b", "2b", "3", "4c", "5", "6"}


# Adversarial
@pytest.mark.unit
class TestMultiConvergentAdversarial:
    """AC-12.1 adversarial: edge cases in multi-convergent layout."""

    def test_merge_consumes_nonexistent_step(self):
        """Merge referencing a non-existent step_id should not crash."""
        scheme = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "a",
                    "branch_label": "A",
                },
                {
                    "id": "2",
                    "consumes_step_ids": ["1", "ghost"],
                    "reactants": ["CC"],
                    "products": ["CCC"],
                    "arrow_above": "merge",
                },
            ],
        }
        # Should not raise — "ghost" just produces an empty branch
        result = compute_layout(scheme)
        assert len(result.boxes) == 2


# ===========================================================================
# AC-12.3: Arrow routing obstacle avoidance
# ===========================================================================


@pytest.mark.unit
class TestArrowObstacleAvoidance:
    """AC-12.3: Connectors should not pass through molecule bounding boxes."""

    def test_connector_avoids_intermediate_box(self):
        """Given a convergent layout where a connector from branch B to merge
        would pass through branch A's bounding box, the connector should route
        around it (use waypoints)."""
        # Simulate: box_a at (0,0,300,200), box_b at (0,260,300,200),
        #           merge at (340,130,300,200)
        # Straight line from box_b center_right(300,360) to merge center_left(340,230)
        # passes through box_a's area
        from chemscheme.renderers.layout_engine import LayoutBox, _route_connections

        boxes = [
            LayoutBox("a", 0, 0, 300, 200),
            LayoutBox("b", 0, 260, 300, 200),
            LayoutBox("merge", 340, 130, 300, 200),
        ]
        connections = [("a", "merge"), ("b", "merge")]
        routed = _route_connections(boxes, connections)
        # Each routed connection should be a list of waypoints
        assert len(routed) == 2
        for conn in routed:
            assert "from_id" in conn
            assert "to_id" in conn
            assert "waypoints" in conn
            assert isinstance(conn["waypoints"], list)
            assert len(conn["waypoints"]) >= 2

        # MR11-C9: Verify obstacle avoidance — the "b"→"merge" connector
        # must NOT pass through box "a"'s bounding rect
        b_conn = next(c for c in routed if c["from_id"] == "b")
        box_a = boxes[0]  # LayoutBox("a", 0, 0, 300, 200)
        for wx, wy in b_conn["waypoints"]:
            # No waypoint should be inside box_a's interior
            inside = box_a.x < wx < box_a.right and box_a.y < wy < box_a.bottom
            assert not inside, (
                f"Waypoint ({wx},{wy}) is inside obstacle box 'a' "
                f"({box_a.x},{box_a.y},{box_a.right},{box_a.bottom})"
            )

    def test_straight_connector_no_obstacle(self):
        """When no obstacle between two boxes, connector stays straight (2 waypoints)."""
        from chemscheme.renderers.layout_engine import LayoutBox, _route_connections

        boxes = [
            LayoutBox("1", 0, 0, 300, 200),
            LayoutBox("2", 340, 0, 300, 200),
        ]
        connections = [("1", "2")]
        routed = _route_connections(boxes, connections)
        assert len(routed) == 1
        # Straight line = exactly 2 waypoints (start, end)
        assert len(routed[0]["waypoints"]) == 2


# ===========================================================================
# AC-12.4: layout/style schema fields wired to renderer
# ===========================================================================


@pytest.mark.unit
class TestLayoutFieldWiring:
    """AC-12.4: layout field should control wrapping behavior."""

    def test_horizontal_layout_no_wrap(self):
        """layout='horizontal' → all steps in one row, no wrapping."""
        scheme = {
            "layout": "horizontal",
            "steps": [
                {"id": str(i), "reactants": ["C"], "products": ["CC"], "arrow_above": ""}
                for i in range(1, 7)  # 6 steps — normally would wrap at 3
            ],
        }
        result = compute_layout(scheme)
        ys = {b.y for b in result.boxes}
        assert len(ys) == 1, f"layout='horizontal' should prevent wrapping, but got {len(ys)} rows"

    def test_default_layout_wraps(self):
        """No layout field → default wrapping at max_steps_per_row."""
        scheme = {
            "steps": [
                {"id": str(i), "reactants": ["C"], "products": ["CC"], "arrow_above": ""}
                for i in range(1, 7)  # 6 steps
            ],
        }
        cfg = LayoutConfig(max_steps_per_row=3)
        result = compute_layout(scheme, config=cfg)
        ys = {b.y for b in result.boxes}
        assert len(ys) == 2, f"Default should wrap to 2 rows, got {len(ys)}"

    def test_grid_layout_wraps(self):
        """layout='grid' → same as default wrapping."""
        scheme = {
            "layout": "grid",
            "steps": [
                {"id": str(i), "reactants": ["C"], "products": ["CC"], "arrow_above": ""}
                for i in range(1, 7)
            ],
        }
        cfg = LayoutConfig(max_steps_per_row=3)
        result = compute_layout(scheme, config=cfg)
        ys = {b.y for b in result.boxes}
        assert len(ys) == 2

    def test_unknown_layout_uses_default(self):
        """Unknown layout value → default wrapping (no crash)."""
        scheme = {
            "layout": "unknown_value",
            "steps": [
                {"id": str(i), "reactants": ["C"], "products": ["CC"], "arrow_above": ""}
                for i in range(1, 7)
            ],
        }
        cfg = LayoutConfig(max_steps_per_row=3)
        result = compute_layout(scheme, config=cfg)
        ys = {b.y for b in result.boxes}
        assert len(ys) == 2


# ===========================================================================
# AC-12.5: Convergent Y-stacking symmetric alignment
# ===========================================================================


@pytest.mark.unit
class TestConvergentSymmetricAlignment:
    """AC-12.5: Merge step should be at the vertical midpoint of the total branch span."""

    def test_asymmetric_branches_merge_centered(self):
        """Branch A (3 steps, taller) + Branch B (1 step) → merge centered at midpoint."""
        scheme = {
            "steps": [
                {
                    "id": "1a",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "",
                    "branch_label": "A",
                },
                {
                    "id": "2a",
                    "reactants": ["CC"],
                    "products": ["CCC"],
                    "arrow_above": "",
                    "branch_label": "A",
                },
                {
                    "id": "3a",
                    "reactants": ["CCC"],
                    "products": ["CCCC"],
                    "arrow_above": "",
                    "branch_label": "A",
                },
                {
                    "id": "1b",
                    "reactants": ["N"],
                    "products": ["NN"],
                    "arrow_above": "",
                    "branch_label": "B",
                },
                {
                    "id": "merge",
                    "consumes_step_ids": ["3a", "1b"],
                    "reactants": ["CCCC", "NN"],
                    "products": ["CCCCNN"],
                    "arrow_above": "",
                },
            ],
        }
        result = compute_layout(scheme)
        box_map = {b.step_id: b for b in result.boxes}
        # Merge Y should be at the midpoint of the total vertical span
        all_branch_ys = [box_map["1a"].y, box_map["1b"].y]
        all_branch_bottoms = [
            box_map["1a"].y + box_map["1a"].height,
            box_map["1b"].y + box_map["1b"].height,
        ]
        span_top = min(all_branch_ys)
        span_bottom = max(all_branch_bottoms)
        span_midpoint = (span_top + span_bottom) / 2
        merge_center = box_map["merge"].y + box_map["merge"].height / 2
        # Allow some tolerance (within half a step height)
        assert abs(merge_center - span_midpoint) < box_map["merge"].height / 2, (
            f"Merge center ({merge_center}) should be near span midpoint ({span_midpoint})"
        )

    def test_three_branches_merge_centered(self):
        """3 branches → merge centered at vertical midpoint of all branches."""
        scheme = _make_triple_branch_scheme()
        result = compute_layout(scheme)
        box_map = {b.step_id: b for b in result.boxes}
        branch_ys = [box_map["1a"].y, box_map["1b"].y, box_map["1c"].y]
        branch_bottoms = [box_map[sid].y + box_map[sid].height for sid in ["1a", "1b", "1c"]]
        span_top = min(branch_ys)
        span_bottom = max(branch_bottoms)
        span_midpoint = (span_top + span_bottom) / 2
        merge_center = box_map["2"].y + box_map["2"].height / 2
        assert abs(merge_center - span_midpoint) < box_map["2"].height / 2


# ===========================================================================
# AC-12.6: Scheme S1 — 6-step convergent end-to-end render
# ===========================================================================


def _make_scheme_s1_convergent():
    """6-step convergent synthesis inspired by ITIC-like NFA.

    Branch A (donor): step1a (bromination) → step2a (Stille coupling)
    Branch B (acceptor): step1b (Knoevenagel) → step2b (formylation)
    Merge: step3 (Suzuki coupling of donor + acceptor)
    Tail: step4 (end-capping)
    """
    return {
        "title": "Scheme S1 — ITIC-type Convergent Synthesis",
        "schema_version": "1",
        "steps": [
            {
                "id": "1a",
                "reactants": ["c1ccc2[nH]ccc2c1"],
                "products": ["c1ccc2[nH]c(Br)cc2c1"],
                "arrow_above": "NBS",
                "arrow_below": "THF, rt",
                "branch_label": "donor",
            },
            {
                "id": "2a",
                "reactants": ["c1ccc2[nH]c(Br)cc2c1"],
                "products": ["c1ccc2[nH]c(-c3ccccc3)cc2c1"],
                "arrow_above": "PhB(OH)2, Pd(PPh3)4",
                "arrow_below": "K2CO3, THF/H2O",
                "branch_label": "donor",
            },
            {
                "id": "1b",
                "reactants": ["O=C1c2ccsc2-c2sccc21"],
                "products": ["O=C1c2cc(Br)sc2-c2sc(Br)cc21"],
                "arrow_above": "NBS",
                "arrow_below": "CHCl3, 0°C",
                "branch_label": "acceptor",
            },
            {
                "id": "2b",
                "reactants": ["O=C1c2cc(Br)sc2-c2sc(Br)cc21"],
                "products": ["O=Cc1cc2c(s1)C(=O)c1cc(Br)sc1-2"],
                "arrow_above": "POCl3, DMF",
                "arrow_below": "DCE, reflux",
                "branch_label": "acceptor",
            },
            {
                "id": "3",
                "consumes_step_ids": ["2a", "2b"],
                "reactants": [
                    "c1ccc2[nH]c(-c3ccccc3)cc2c1",
                    "O=Cc1cc2c(s1)C(=O)c1cc(Br)sc1-2",
                ],
                "products": ["c1ccc(-c2cc3c(s2)-c2sc(-c4cc5ccccc5[nH]4)cc2C3=O)cc1"],
                "arrow_above": "Pd2(dba)3, P(o-tol)3",
                "arrow_below": "toluene, 110°C",
            },
            {
                "id": "4",
                "reactants": ["c1ccc(-c2cc3c(s2)-c2sc(-c4cc5ccccc5[nH]4)cc2C3=O)cc1"],
                "products": ["N#C/C=C\\c1cc2c(s1)-c1sc(/C=C\\C#N)cc1C2=O"],
                "arrow_above": "malononitrile, pyridine",
                "arrow_below": "CHCl3, reflux",
            },
        ],
    }


@pytest.mark.unit
@pytest.mark.chemistry
class TestSchemeS1Convergent:
    """AC-12.6: End-to-end rendering of a 6-step convergent synthesis."""

    def test_s1_layout_has_6_boxes(self):
        """All 6 steps should produce layout boxes."""
        scheme = _make_scheme_s1_convergent()
        result = compute_layout(scheme)
        assert len(result.boxes) == 6

    def test_s1_branches_at_different_y(self):
        """Donor and acceptor branches should be at different Y positions."""
        scheme = _make_scheme_s1_convergent()
        result = compute_layout(scheme)
        box_map = {b.step_id: b for b in result.boxes}
        assert box_map["1a"].y != box_map["1b"].y

    def test_s1_merge_after_branches(self):
        """Merge step 3 should be positioned after both branches."""
        scheme = _make_scheme_s1_convergent()
        result = compute_layout(scheme)
        box_map = {b.step_id: b for b in result.boxes}
        assert box_map["3"].x > box_map["2a"].x
        assert box_map["3"].x > box_map["2b"].x

    def test_s1_no_box_overlap(self):
        """No layout boxes should overlap."""
        scheme = _make_scheme_s1_convergent()
        result = compute_layout(scheme)
        for i, a in enumerate(result.boxes):
            for b in result.boxes[i + 1 :]:
                # Check no overlap: one must be fully left/right/above/below the other
                h_sep = a.right <= b.x or b.right <= a.x
                v_sep = a.bottom <= b.y or b.bottom <= a.y
                assert h_sep or v_sep, (
                    f"Boxes {a.step_id} and {b.step_id} overlap: "
                    f"{a.step_id}=({a.x},{a.y},{a.right},{a.bottom}) "
                    f"{b.step_id}=({b.x},{b.y},{b.right},{b.bottom})"
                )

    def test_s1_renders_svg_without_error(self):
        """Full render to SVG should succeed and produce valid SVG."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = _make_scheme_s1_convergent()
        svg_bytes = render_scheme(scheme, fmt="svg")
        svg = svg_bytes.decode("utf-8")
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        # Should contain all 6 step groups
        for sid in ["1a", "2a", "1b", "2b", "3", "4"]:
            assert f'data-step-id="{sid}"' in svg

    def test_s1_svg_has_connectors(self):
        """SVG should have connector arrows between steps."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = _make_scheme_s1_convergent()
        svg_bytes = render_scheme(scheme, fmt="svg")
        svg = svg_bytes.decode("utf-8")
        # Should have arrowhead marker
        assert "arrowhead" in svg
        # Should have connectors (line or polyline with marker-end)
        connectors = re.findall(r"<(line|polyline)[^>]*marker-end[^>]*/>", svg)
        # 5 connections: 1a→2a, 1b→2b, 2a→3, 2b→3, 3→4
        assert len(connectors) == 5, f"Expected 5 connectors, got {len(connectors)}"
