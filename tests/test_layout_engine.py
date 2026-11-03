"""Tests for layout engine — coordinate calculations without rendering.

Covers AC-3.1, AC-3.2, AC-3.3, AC-3.5, AC-3.6.
"""

import pytest

from chemscheme.renderers.layout_engine import (
    LayoutBox,
    LayoutConfig,
    LayoutResult,
    compute_layout,
)

# ---------------------------------------------------------------------------
# AC-3.6: Layout engine coordinate calculation unit tests
# ---------------------------------------------------------------------------


class TestLayoutConfig:
    """LayoutConfig data structure."""

    @pytest.mark.unit
    def test_default_config(self):
        cfg = LayoutConfig()
        assert cfg.max_steps_per_row == 3
        assert cfg.step_width > 0
        assert cfg.step_height > 0
        assert cfg.h_gap > 0
        assert cfg.v_gap > 0
        assert cfg.numbering_style == "none"

    @pytest.mark.unit
    def test_custom_config(self):
        cfg = LayoutConfig(max_steps_per_row=4, numbering_style="roman")
        assert cfg.max_steps_per_row == 4
        assert cfg.numbering_style == "roman"


class TestLayoutBox:
    """LayoutBox data structure."""

    @pytest.mark.unit
    def test_layout_box_fields(self):
        box = LayoutBox(step_id="1", x=0.0, y=0.0, width=300.0, height=200.0)
        assert box.step_id == "1"
        assert box.x == 0.0
        assert box.y == 0.0
        assert box.width == 300.0
        assert box.height == 200.0

    @pytest.mark.unit
    def test_layout_box_right_edge(self):
        box = LayoutBox(step_id="1", x=10.0, y=20.0, width=300.0, height=200.0)
        assert box.right == 310.0
        assert box.bottom == 220.0


class TestLayoutResult:
    """LayoutResult data structure."""

    @pytest.mark.unit
    def test_layout_result_fields(self):
        boxes = [LayoutBox(step_id="1", x=0, y=0, width=300, height=200)]
        result = LayoutResult(
            boxes=boxes,
            canvas_width=300.0,
            canvas_height=200.0,
            connections=[],
        )
        assert len(result.boxes) == 1
        assert result.canvas_width == 300.0
        assert result.canvas_height == 200.0


# ---------------------------------------------------------------------------
# AC-3.1: 3 steps or fewer → single row
# ---------------------------------------------------------------------------


class TestLinearSingleRow:
    """Linear schemes with ≤3 steps should be laid out in one row."""

    @pytest.mark.unit
    def test_single_step_layout(self, sample_scheme):
        """1-step scheme → single box at origin."""
        result = compute_layout(sample_scheme)
        assert len(result.boxes) == 1
        assert result.boxes[0].x == 0.0
        assert result.boxes[0].y == 0.0

    @pytest.mark.unit
    def test_three_step_single_row(self, sample_scheme_3step):
        """3-step scheme → all boxes in same row (same y)."""
        result = compute_layout(sample_scheme_3step)
        assert len(result.boxes) == 3
        # All boxes should have the same y coordinate (single row)
        ys = {box.y for box in result.boxes}
        assert len(ys) == 1, f"Expected single row, got y values: {ys}"

    @pytest.mark.unit
    def test_three_step_left_to_right(self, sample_scheme_3step):
        """3-step scheme → boxes ordered left to right."""
        result = compute_layout(sample_scheme_3step)
        xs = [box.x for box in result.boxes]
        assert xs == sorted(xs), "Boxes should be in left-to-right order"
        # No overlap
        for i in range(len(result.boxes) - 1):
            assert result.boxes[i].right <= result.boxes[i + 1].x

    @pytest.mark.unit
    def test_canvas_fits_all_boxes(self, sample_scheme_3step):
        """Canvas dimensions should encompass all boxes."""
        result = compute_layout(sample_scheme_3step)
        for box in result.boxes:
            assert box.right <= result.canvas_width
            assert box.bottom <= result.canvas_height

    @pytest.mark.unit
    def test_connections_between_steps(self, sample_scheme_3step):
        """Linear scheme should have connections between consecutive steps."""
        result = compute_layout(sample_scheme_3step)
        assert len(result.connections) == 2  # 3 steps → 2 connections
        # Each connection is (from_id, to_id)
        assert result.connections[0] == ("1", "2")
        assert result.connections[1] == ("2", "3")


# ---------------------------------------------------------------------------
# AC-3.2: 4+ steps → auto wrap with visual connection
# ---------------------------------------------------------------------------


class TestLinearMultiRow:
    """Linear schemes with 4+ steps should wrap automatically."""

    @pytest.mark.unit
    def test_five_step_wraps(self, sample_scheme_5step_nfa):
        """5-step scheme with max_steps_per_row=3 → 2 rows."""
        cfg = LayoutConfig(max_steps_per_row=3)
        result = compute_layout(sample_scheme_5step_nfa, config=cfg)
        assert len(result.boxes) == 5
        # Should have 2 distinct y values (2 rows)
        ys = sorted({box.y for box in result.boxes})
        assert len(ys) == 2, f"Expected 2 rows, got y values: {ys}"

    @pytest.mark.unit
    def test_five_step_row_distribution(self, sample_scheme_5step_nfa):
        """5 steps, max 3 per row → row 1: 3 steps, row 2: 2 steps."""
        cfg = LayoutConfig(max_steps_per_row=3)
        result = compute_layout(sample_scheme_5step_nfa, config=cfg)
        ys = sorted({box.y for box in result.boxes})
        row1 = [b for b in result.boxes if b.y == ys[0]]
        row2 = [b for b in result.boxes if b.y == ys[1]]
        assert len(row1) == 3
        assert len(row2) == 2

    @pytest.mark.unit
    def test_second_row_below_first(self, sample_scheme_5step_nfa):
        """Second row y > first row y + first row height."""
        cfg = LayoutConfig(max_steps_per_row=3)
        result = compute_layout(sample_scheme_5step_nfa, config=cfg)
        ys = sorted({box.y for box in result.boxes})
        row1_bottom = max(b.bottom for b in result.boxes if b.y == ys[0])
        assert ys[1] >= row1_bottom, "Rows should not overlap vertically"

    @pytest.mark.unit
    def test_wrap_connections_include_row_transition(self, sample_scheme_5step_nfa):
        """Connections should include the row-wrap transition (step 3 → step 4)."""
        cfg = LayoutConfig(max_steps_per_row=3)
        result = compute_layout(sample_scheme_5step_nfa, config=cfg)
        # 5 steps → 4 connections
        assert len(result.connections) == 4
        assert ("3", "4") in result.connections

    @pytest.mark.unit
    def test_custom_max_steps_per_row(self, sample_scheme_5step_nfa):
        """max_steps_per_row=5 → all in one row."""
        cfg = LayoutConfig(max_steps_per_row=5)
        result = compute_layout(sample_scheme_5step_nfa, config=cfg)
        ys = {box.y for box in result.boxes}
        assert len(ys) == 1, "All 5 steps should fit in one row"


# ---------------------------------------------------------------------------
# AC-3.3: Convergent route layout
# ---------------------------------------------------------------------------


class TestConvergentLayout:
    """Convergent routes (branches → merge) layout correctly."""

    @pytest.mark.unit
    def test_convergent_two_branches(self, sample_scheme_convergent):
        """2 branches (1a, 1b) → merge at step 2."""
        result = compute_layout(sample_scheme_convergent)
        assert len(result.boxes) == 3
        box_map = {b.step_id: b for b in result.boxes}
        # Branch steps should be stacked vertically (different y)
        assert box_map["1a"].y != box_map["1b"].y
        # Merge step should be to the right of both branches
        assert box_map["2"].x > box_map["1a"].x
        assert box_map["2"].x > box_map["1b"].x

    @pytest.mark.unit
    def test_convergent_merge_connections(self, sample_scheme_convergent):
        """Merge step should have connections from both branches."""
        result = compute_layout(sample_scheme_convergent)
        assert ("1a", "2") in result.connections
        assert ("1b", "2") in result.connections

    @pytest.mark.unit
    def test_convergent_merge_vertically_centered(self, sample_scheme_convergent):
        """Merge step should be vertically centered between branches."""
        result = compute_layout(sample_scheme_convergent)
        box_map = {b.step_id: b for b in result.boxes}
        branch_center = (box_map["1a"].y + box_map["1b"].y) / 2
        merge_y = box_map["2"].y
        # Allow some tolerance
        assert abs(merge_y - branch_center) < 1.0

    @pytest.mark.unit
    def test_8step_convergent_layout(self, sample_scheme_8step_convergent):
        """8-step convergent: 2 branches (3a+3b each) → merge → final step."""
        result = compute_layout(sample_scheme_8step_convergent)
        assert len(result.boxes) == 8
        box_map = {b.step_id: b for b in result.boxes}
        # Branch a steps should be in one row
        assert box_map["1a"].y == box_map["2a"].y == box_map["3a"].y
        # Branch b steps should be in another row
        assert box_map["1b"].y == box_map["2b"].y == box_map["3b"].y
        # Branch a and b should be at different y
        assert box_map["1a"].y != box_map["1b"].y
        # Merge step 4 should be after both branches
        assert box_map["4"].x > box_map["3a"].x
        assert box_map["4"].x > box_map["3b"].x

    @pytest.mark.unit
    def test_8step_convergent_post_merge_linear(self, sample_scheme_8step_convergent):
        """After merge, step 5 should be to the right of step 4."""
        result = compute_layout(sample_scheme_8step_convergent)
        box_map = {b.step_id: b for b in result.boxes}
        assert box_map["5"].x > box_map["4"].x
        assert box_map["5"].y == box_map["4"].y

    @pytest.mark.unit
    def test_8step_convergent_connections(self, sample_scheme_8step_convergent):
        """Connections for convergent 8-step scheme."""
        result = compute_layout(sample_scheme_8step_convergent)
        # Branch a: 1a→2a→3a
        assert ("1a", "2a") in result.connections
        assert ("2a", "3a") in result.connections
        # Branch b: 1b→2b→3b
        assert ("1b", "2b") in result.connections
        assert ("2b", "3b") in result.connections
        # Merge: 3a→4, 3b→4
        assert ("3a", "4") in result.connections
        assert ("3b", "4") in result.connections
        # Post-merge: 4→5
        assert ("4", "5") in result.connections


# ---------------------------------------------------------------------------
# B2-P1a: Non-contiguous branch steps — graph-based branch detection
# ---------------------------------------------------------------------------


class TestNonContiguousBranches:
    """B2-P1a: Branch steps that are not contiguous in the list should still
    be correctly connected via branch_label grouping, not adjacency."""

    @pytest.mark.unit
    def test_noncontiguous_branch_dependencies(self, sample_scheme_noncontiguous_branches):
        """Step 2a (branch A) should depend on 1a (branch A), NOT on 1b (branch B).

        Scheme order: 1a(A), 1b(B), 2a(A), 3(merge).
        Step 2a is not adjacent to 1a but shares branch_label 'A'.
        """
        from chemscheme.renderers.layout_engine import _build_dependency_graph

        steps = sample_scheme_noncontiguous_branches["steps"]
        predecessors = _build_dependency_graph(steps)
        # 2a should have predecessor 1a (same branch), NOT 1b (adjacent but different branch)
        assert predecessors["2a"] == ["1a"]

    @pytest.mark.unit
    def test_noncontiguous_branch_connections(self, sample_scheme_noncontiguous_branches):
        """Layout connections should reflect correct branch dependencies."""
        result = compute_layout(sample_scheme_noncontiguous_branches)
        # Branch A: 1a → 2a
        assert ("1a", "2a") in result.connections
        # Branch B: 1b stands alone (no successor in branch B before merge)
        # Merge: 2a → 3, 1b → 3
        assert ("2a", "3") in result.connections
        assert ("1b", "3") in result.connections
        # MUST NOT have: 1b → 2a (wrong branch link)
        assert ("1b", "2a") not in result.connections

    @pytest.mark.unit
    def test_noncontiguous_branch_layout_y_positions(self, sample_scheme_noncontiguous_branches):
        """Branch A steps should share a y position, separate from branch B."""
        result = compute_layout(sample_scheme_noncontiguous_branches)
        box_map = {b.step_id: b for b in result.boxes}
        # 1a and 2a (both branch A) should be on same row
        assert box_map["1a"].y == box_map["2a"].y
        # 1b (branch B) should be on a different row
        assert box_map["1b"].y != box_map["1a"].y
        # Merge step 3 should be to the right of the longest branch
        assert box_map["3"].x > box_map["2a"].x
        assert box_map["3"].x > box_map["1b"].x


# ---------------------------------------------------------------------------
# AC-3.5: Step numbering (configurable)
# ---------------------------------------------------------------------------


class TestStepNumbering:
    """Step numbering styles."""

    @pytest.mark.unit
    def test_numbering_none(self, sample_scheme_3step):
        """numbering_style='none' → no step_label in boxes."""
        cfg = LayoutConfig(numbering_style="none")
        result = compute_layout(sample_scheme_3step, config=cfg)
        for box in result.boxes:
            assert box.step_label is None

    @pytest.mark.unit
    def test_numbering_arabic(self, sample_scheme_3step):
        """numbering_style='arabic' → '1', '2', '3'."""
        cfg = LayoutConfig(numbering_style="arabic")
        result = compute_layout(sample_scheme_3step, config=cfg)
        labels = [box.step_label for box in result.boxes]
        assert labels == ["1", "2", "3"]

    @pytest.mark.unit
    def test_numbering_roman(self, sample_scheme_3step):
        """numbering_style='roman' → 'i', 'ii', 'iii'."""
        cfg = LayoutConfig(numbering_style="roman")
        result = compute_layout(sample_scheme_3step, config=cfg)
        labels = [box.step_label for box in result.boxes]
        assert labels == ["i", "ii", "iii"]

    @pytest.mark.unit
    def test_numbering_roman_5steps(self, sample_scheme_5step_nfa):
        """Roman numbering for 5 steps."""
        cfg = LayoutConfig(numbering_style="roman")
        result = compute_layout(sample_scheme_5step_nfa, config=cfg)
        labels = [box.step_label for box in result.boxes]
        assert labels == ["i", "ii", "iii", "iv", "v"]
