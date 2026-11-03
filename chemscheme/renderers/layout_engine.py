"""Layout engine for multi-step reaction schemes.

Pure coordinate math — no rendering dependencies.
Supports linear (single/multi-row) and convergent layouts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutBox:
    """Position and dimensions of a single step in the layout."""

    step_id: str
    x: float
    y: float
    width: float
    height: float
    step_label: str | None = None

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class LayoutConfig:
    """Configuration for layout computation."""

    max_steps_per_row: int = 3
    step_width: float = 300.0
    step_height: float = 200.0
    h_gap: float = 40.0
    v_gap: float = 60.0
    numbering_style: str = "none"  # "none", "arabic", "roman"
    # G2 (S68): width-budget-based row wrap. None = disabled (use step-count only).
    # When set, _layout_linear starts a new row when adding the next step would
    # push cumulative row width past this budget. A single oversized step always
    # gets its own row (no infinite loop). Step-count cap and width budget are
    # both honored — whichever fires first.
    max_row_width: float | None = None


@dataclass
class LayoutResult:
    """Result of layout computation."""

    boxes: list[LayoutBox]
    canvas_width: float
    canvas_height: float
    connections: list[tuple[str, str]]


# -- Numbering helpers -------------------------------------------------------

_ROMAN = [
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
    "xiii",
    "xiv",
    "xv",
    "xvi",
    "xvii",
    "xviii",
    "xix",
    "xx",
]


def _step_label(index: int, style: str) -> str | None:
    """Generate a step label for the given 0-based index."""
    if style == "none":
        return None
    if style == "arabic":
        return str(index + 1)
    if style == "roman":
        if index < len(_ROMAN):
            return _ROMAN[index]
        return str(index + 1)  # fallback
    return None


# -- Graph analysis ----------------------------------------------------------


def _build_dependency_graph(steps: list[dict]) -> dict[str, list[str]]:  # type: ignore[type-arg]
    """Build a mapping of step_id → list of predecessor step_ids.

    Uses graph-based branch detection:
    - Steps with consumes_step_ids use those explicit predecessors.
    - Steps with a branch_label connect to the last step in the same branch
      (regardless of list position), not the adjacent step.
    - Steps without branch_label or consumes_step_ids connect to the previous
      step in the list (linear fallback).
    """
    predecessors: dict[str, list[str]] = {}

    # Track the last step_id seen for each branch_label
    branch_last: dict[str, str] = {}

    for i, step in enumerate(steps):
        sid = step["id"]

        if "consumes_step_ids" in step:
            predecessors[sid] = list(step["consumes_step_ids"])
        elif i == 0:
            predecessors[sid] = []
        else:
            curr_branch = step.get("branch_label")

            if curr_branch:
                if curr_branch in branch_last:
                    # Continue an existing branch → predecessor is last step in same branch
                    predecessors[sid] = [branch_last[curr_branch]]
                else:
                    # First step in a new branch → independent root
                    predecessors[sid] = []
            else:
                # No branch_label → linear fallback to previous step in list
                prev_id = steps[i - 1]["id"]
                predecessors[sid] = [prev_id]

        # Update branch tracking
        branch = step.get("branch_label")
        if branch:
            branch_last[branch] = sid

    return predecessors


def _find_branches(
    steps: list[dict],  # type: ignore[type-arg]
    predecessors: dict[str, list[str]],
) -> tuple[list[list[str]], list[str]]:
    """Identify parallel branches and the post-merge linear tail.

    Returns:
        (branches, tail) where branches is a list of branch step-id lists,
        and tail is the list of step-ids after the merge point (inclusive).
    """
    # Find merge steps (steps that consume multiple predecessors)
    merge_steps = [s["id"] for s in steps if len(predecessors.get(s["id"], [])) > 1]

    if not merge_steps:
        # Check for branch_label-based parallel branches without merge (MR4-4)
        branch_groups: dict[str, list[str]] = {}
        unlabeled: list[str] = []
        for s in steps:
            bl = s.get("branch_label")
            if bl:
                branch_groups.setdefault(bl, []).append(s["id"])
            else:
                unlabeled.append(s["id"])

        if len(branch_groups) >= 2:
            # Multiple branch labels → parallel layout, unlabeled steps as tail
            branches_out = list(branch_groups.values())
            return branches_out, unlabeled

        # Pure linear — single branch, no tail split needed
        return [[s["id"] for s in steps]], []

    first_merge = merge_steps[0]
    merge_preds = predecessors[first_merge]

    # Trace each branch backwards from merge predecessors
    step_index = {s["id"]: i for i, s in enumerate(steps)}
    branches: list[list[str]] = []
    # Shared visited set prevents common ancestors appearing in multiple branches (MR5-G3/C1)
    visited: set[str] = set()

    for pred_id in merge_preds:
        if pred_id not in step_index:
            continue  # Skip non-existent predecessor IDs (AC-12.1 adversarial)
        # Walk backwards collecting the branch (with cycle guard — MR3-14)
        branch: list[str] = []
        current: str | None = pred_id
        while current is not None and current not in visited:
            visited.add(current)
            branch.append(current)
            preds = predecessors.get(current, [])
            current = preds[0] if len(preds) == 1 else None
        branch.reverse()
        if branch:  # Skip empty branches from shared visited set (MR5R2-G7)
            branches.append(branch)

    # Tail: merge step and everything after it
    merge_idx = step_index[first_merge]
    tail = [s["id"] for s in steps[merge_idx:]]

    return branches, tail


# -- Layout algorithms -------------------------------------------------------


def _assign_rows(
    step_ids: list[str],
    config: LayoutConfig,
    sizes: dict[str, tuple[float, float]],
) -> list[int]:
    """Assign each step to a row index honoring step-count + width-budget caps.

    Returns a list of row indices parallel to `step_ids`.
    Both caps fire first-to-trigger:
    - step-count: row breaks every `max_steps_per_row` steps
    - width-budget: row breaks when adding next step's (gap + width) > max_row_width

    A single step wider than max_row_width still gets placed on its own row
    (no infinite-loop attempt to fit it).
    """
    use_width = config.max_row_width is not None and config.max_row_width > 0
    rows: list[int] = []
    row_idx = 0
    row_steps = 0
    row_cursor = 0.0  # cumulative width of current row

    for sid in step_ids:
        w, _ = sizes.get(sid, (config.step_width, config.step_height))

        # Compute prospective cursor IF we placed this step in current row
        prospective = row_cursor + (config.h_gap if row_steps > 0 else 0.0) + w

        # Decide break: step-count cap reached, OR width budget exceeded
        # (but never break when row is empty — single oversized step gets its own row)
        step_count_cap = row_steps >= config.max_steps_per_row
        width_cap = use_width and row_steps > 0 and prospective > config.max_row_width  # type: ignore[operator]

        if step_count_cap or width_cap:
            row_idx += 1
            row_steps = 0
            row_cursor = 0.0
            prospective = w  # reset for new row

        rows.append(row_idx)
        row_steps += 1
        row_cursor = prospective

    return rows


def _layout_linear(
    step_ids: list[str],
    config: LayoutConfig,
    start_x: float = 0.0,
    start_y: float = 0.0,
    label_offset: int = 0,
    step_sizes: dict[str, tuple[float, float]] | None = None,
) -> list[LayoutBox]:
    """Lay out steps in a linear sequence with row wrapping (step-count + width-budget)."""
    sizes = step_sizes or {}
    row_assignments = _assign_rows(step_ids, config, sizes)

    # Pre-compute max height per row for correct Y offset (C5 fix; G2 uses dynamic rows)
    row_max_h: dict[int, float] = {}
    for sid, row in zip(step_ids, row_assignments, strict=True):
        _, h = sizes.get(sid, (config.step_width, config.step_height))
        row_max_h[row] = max(row_max_h.get(row, 0.0), h)

    boxes: list[LayoutBox] = []
    row_cursor_x = start_x
    current_row = 0

    for i, (sid, row) in enumerate(zip(step_ids, row_assignments, strict=True)):
        w, h = sizes.get(sid, (config.step_width, config.step_height))

        if row != current_row:
            row_cursor_x = start_x
            current_row = row
        elif i > 0 and row_assignments[i - 1] == row:
            # Same row as previous step — add horizontal gap
            row_cursor_x += config.h_gap

        # Y offset from cumulative max heights of previous rows (C5 fix)
        y = start_y + sum(row_max_h.get(r, config.step_height) + config.v_gap for r in range(row))
        label = _step_label(label_offset + i, config.numbering_style)
        boxes.append(
            LayoutBox(
                step_id=sid,
                x=row_cursor_x,
                y=y,
                width=w,
                height=h,
                step_label=label,
            )
        )
        row_cursor_x += w
    return boxes


def _layout_convergent(
    branches: list[list[str]],
    tail: list[str],
    config: LayoutConfig,
    step_sizes: dict[str, tuple[float, float]] | None = None,
) -> list[LayoutBox]:
    """Lay out convergent branches stacked vertically, then a linear tail."""
    boxes: list[LayoutBox] = []
    label_counter = 0
    sizes = step_sizes or {}

    # Pre-compute max height per branch for correct Y stacking (C5 fix)
    branch_max_h: list[float] = []
    for branch in branches:
        max_h = max(
            (sizes.get(sid, (config.step_width, config.step_height))[1] for sid in branch),
            default=config.step_height,
        )
        branch_max_h.append(max_h)

    # Lay out each branch horizontally, stacked vertically
    branch_ys: list[float] = []
    branch_right_edges: list[float] = []

    for branch_idx, branch in enumerate(branches):
        # Y offset from cumulative max heights of previous branches (C5 fix)
        y = sum(branch_max_h[b] + config.v_gap for b in range(branch_idx))
        branch_ys.append(y)
        cursor_x = 0.0
        for col, sid in enumerate(branch):
            w, h = sizes.get(sid, (config.step_width, config.step_height))
            if col > 0:
                cursor_x += config.h_gap
            label = _step_label(label_counter, config.numbering_style)
            boxes.append(
                LayoutBox(
                    step_id=sid,
                    x=cursor_x,
                    y=y,
                    width=w,
                    height=h,
                    step_label=label,
                )
            )
            cursor_x += w
            label_counter += 1
        branch_right_edges.append(cursor_x)

    # Merge point and tail: placed after the longest branch, vertically centered
    tail_start_x = max(branch_right_edges) + config.h_gap
    center_y = sum(branch_ys) / len(branch_ys)

    cursor_x = tail_start_x
    for i, sid in enumerate(tail):
        w, h = sizes.get(sid, (config.step_width, config.step_height))
        if i > 0:
            cursor_x += config.h_gap
        label = _step_label(label_counter, config.numbering_style)
        boxes.append(
            LayoutBox(
                step_id=sid,
                x=cursor_x,
                y=center_y,
                width=w,
                height=h,
                step_label=label,
            )
        )
        cursor_x += w
        label_counter += 1

    return boxes


def _build_connections(
    steps: list[dict],  # type: ignore[type-arg]
    predecessors: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Build connection list as (from_id, to_id) tuples."""
    step_ids = {s["id"] for s in steps}
    connections: list[tuple[str, str]] = []
    for step in steps:
        sid = step["id"]
        for pred in predecessors.get(sid, []):
            if pred in step_ids:  # Skip non-existent predecessors (AC-12.1)
                connections.append((pred, sid))
    return connections


# -- Connection routing with obstacle avoidance (AC-12.3, MR7-16) -----------


def _boxes_overlap_segment(
    boxes: list[LayoutBox],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    exclude_ids: set[str],
) -> list[LayoutBox]:
    """Return boxes whose bounding rect is crossed by the line segment (x1,y1)→(x2,y2).

    Uses axis-aligned bounding box intersection with the segment's bounding rect
    plus a midpoint containment check for simple obstacle detection.
    """
    seg_left = min(x1, x2)
    seg_right = max(x1, x2)
    seg_top = min(y1, y2)
    seg_bottom = max(y1, y2)

    obstacles: list[LayoutBox] = []
    for box in boxes:
        if box.step_id in exclude_ids:
            continue
        # Check if segment bounding rect overlaps box bounding rect
        if box.right <= seg_left or box.x >= seg_right:
            continue
        if box.bottom <= seg_top or box.y >= seg_bottom:
            continue
        # Additional check: segment midpoint inside box (simple heuristic)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        if box.x <= mid_x <= box.right and box.y <= mid_y <= box.bottom:
            obstacles.append(box)
    return obstacles


def _route_connections(
    boxes: list[LayoutBox],
    connections: list[tuple[str, str]],
) -> list[dict]:  # type: ignore[type-arg]
    """Route connections between boxes, avoiding obstacles.

    Returns a list of dicts with keys: from_id, to_id, waypoints.
    Waypoints is a list of (x, y) tuples forming the connector path.
    When no obstacle exists, waypoints is just [start, end] (straight line).
    When obstacles exist, routes around them using orthogonal detour.
    """
    box_map = {b.step_id: b for b in boxes}
    routed: list[dict] = []  # type: ignore[type-arg]

    for from_id, to_id in connections:
        from_box = box_map[from_id]
        to_box = box_map[to_id]

        x1 = from_box.right
        y1 = from_box.y + from_box.height / 2
        x2 = to_box.x
        y2 = to_box.y + to_box.height / 2

        exclude = {from_id, to_id}
        obstacles = _boxes_overlap_segment(boxes, x1, y1, x2, y2, exclude)

        if not obstacles:
            # No obstacle — straight or L-shape based on y difference
            if abs(y1 - y2) > 1:
                mid_y = (y1 + y2) / 2
                waypoints = [(x1, y1), (x1, mid_y), (x2, mid_y), (x2, y2)]
            else:
                waypoints = [(x1, y1), (x2, y2)]
        else:
            # Route around obstacles: go below or above the obstacle
            obs = obstacles[0]  # handle first obstacle
            margin = 20.0
            # Decide: go above or below based on which is closer
            above_y = obs.y - margin
            below_y = obs.bottom + margin
            detour_y = above_y if abs(y1 - above_y) < abs(y1 - below_y) else below_y

            waypoints = [
                (x1, y1),
                (x1, detour_y),
                (x2, detour_y),
                (x2, y2),
            ]

        routed.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "waypoints": waypoints,
            }
        )

    return routed


# -- Public API --------------------------------------------------------------


def compute_layout(
    scheme: dict,  # type: ignore[type-arg]
    config: LayoutConfig | None = None,
    step_sizes: dict[str, tuple[float, float]] | None = None,
) -> LayoutResult:
    """Compute layout coordinates for a scheme.

    Args:
        scheme: A validated scheme dict with "steps" list.
        config: Layout configuration. Uses defaults if None.
        step_sizes: Optional per-step size overrides mapping step_id to (width, height).
            Steps not in this dict use config defaults.

    Returns:
        LayoutResult with box positions, canvas dimensions, and connections.
    """
    config_was_default = config is None
    if config is None:
        config = LayoutConfig()

    # Wire layout field: "horizontal" disables row wrapping (AC-12.4, MR7-13)
    # Only apply when no explicit config was provided (explicit config takes precedence)
    layout_mode = scheme.get("layout", "")
    if layout_mode == "horizontal" and config_was_default:
        config = LayoutConfig(
            max_steps_per_row=999,
            step_width=config.step_width,
            step_height=config.step_height,
            h_gap=config.h_gap,
            v_gap=config.v_gap,
            numbering_style=config.numbering_style,
        )

    steps = scheme["steps"]
    predecessors = _build_dependency_graph(steps)
    connections = _build_connections(steps, predecessors)

    branches, tail = _find_branches(steps, predecessors)

    if len(branches) == 1 and not tail:
        # Pure linear layout
        boxes = _layout_linear(branches[0], config, step_sizes=step_sizes)
    else:
        # Convergent layout
        boxes = _layout_convergent(branches, tail, config, step_sizes=step_sizes)

    # Compute canvas dimensions
    canvas_width = max(box.right for box in boxes) if boxes else 0.0
    canvas_height = max(box.bottom for box in boxes) if boxes else 0.0

    return LayoutResult(
        boxes=boxes,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        connections=connections,
    )
