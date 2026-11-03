"""G2 — width-budget-based row wrap for layout engine.

Closes the S67 Part 2 G2 CRITICAL gap: prior to S68, `LayoutConfig` only wrapped
by step COUNT (`max_steps_per_row`); 5+ step schemes overflowed journal column
widths (single-col ~600px, double-col ~1200px). This adds `max_row_width` —
when set, the linear layouter starts a new row when adding the next step would
exceed the cumulative width budget for the current row.

Backward compatibility: `max_row_width=None` (default) preserves existing
step-count behavior verbatim. Explicit step-count cap and width-budget cap are
both honored — whichever fires first.
"""

from __future__ import annotations

from click.testing import CliRunner

from chemscheme.cli import main as cli_main
from chemscheme.renderers.layout_engine import LayoutConfig, compute_layout


def _scheme(n_steps: int, *, layout: str = "") -> dict:
    """Build an n-step linear scheme. layout="" lets engine defaults apply."""
    steps = [
        {
            "id": str(i + 1),
            "reactants": [f"C{'C' * i}"],
            "products": [f"C{'C' * i}Br"],
            "arrow_above": "Br2",
        }
        for i in range(n_steps)
    ]
    s: dict = {"schema_version": "1", "steps": steps}
    if layout:
        s["layout"] = layout
    return s


# ──────────────────────────── Backward compatibility ────────────────────────────


def test_default_config_unchanged_max_row_width_is_none() -> None:
    """LayoutConfig() default must keep max_row_width None (no wrap-by-width)."""
    cfg = LayoutConfig()
    assert cfg.max_row_width is None


def test_wrap_by_step_count_unchanged_when_max_row_width_none() -> None:
    # RED-EXEMPT: behavior lock — pre-existing step-count wrap, no new code needed
    """6 steps, default config (max_steps_per_row=3, no width budget) → 2 rows of 3."""
    res = compute_layout(_scheme(6))
    rows: dict[float, list] = {}
    for box in res.boxes:
        rows.setdefault(box.y, []).append(box)
    assert len(rows) == 2, f"expected 2 rows by step-count, got {len(rows)}"
    for row_boxes in rows.values():
        assert len(row_boxes) == 3


def test_horizontal_layout_mode_still_disables_step_wrap() -> None:
    # RED-EXEMPT: behavior lock — pre-existing AC-12.4 horizontal mode preserved
    """`layout: horizontal` continues to disable step-count wrap (existing AC-12.4)."""
    res = compute_layout(_scheme(5, layout="horizontal"))
    rows = {box.y for box in res.boxes}
    assert len(rows) == 1, "horizontal layout must keep all steps on one row"


# ──────────────────────────── width-budget wrap ────────────────────────────


def test_max_row_width_breaks_when_next_step_exceeds_budget() -> None:
    """5 steps × 300px each + 40px gaps = 1660px raw; budget=700 → expect 2 per row.

    Step 1 (0-300) → cumulative 300, fits 700
    Step 2 (340-640) → cumulative 640, fits 700
    Step 3 would push to 980 → exceeds → break to new row.
    """
    cfg = LayoutConfig(max_row_width=700, max_steps_per_row=999)
    res = compute_layout(_scheme(5), config=cfg)
    rows: dict[float, list] = {}
    for box in res.boxes:
        rows.setdefault(box.y, []).append(box)
    # 5 steps with budget 700 → rows of 2,2,1
    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}: {rows}"
    sizes = sorted(len(r) for r in rows.values())
    assert sizes == [1, 2, 2]


def test_max_row_width_with_single_oversized_step_does_not_loop() -> None:
    """Step wider than budget: place it on its own row, don't infinite-loop."""
    cfg = LayoutConfig(max_row_width=200, step_width=500.0, max_steps_per_row=999)
    res = compute_layout(_scheme(3), config=cfg)
    rows: dict[float, list] = {}
    for box in res.boxes:
        rows.setdefault(box.y, []).append(box)
    # Each step (500px) > budget (200) → 3 separate rows
    assert len(rows) == 3
    for row_boxes in rows.values():
        assert len(row_boxes) == 1


def test_max_row_width_canvas_respects_budget_within_tolerance() -> None:
    """Canvas width with budget=700 should not exceed budget by more than one step."""
    cfg = LayoutConfig(max_row_width=700, max_steps_per_row=999)
    res = compute_layout(_scheme(8), config=cfg)
    # Each row's right-most box must end within budget
    rows: dict[float, list] = {}
    for box in res.boxes:
        rows.setdefault(box.y, []).append(box)
    for row_boxes in rows.values():
        right = max(b.right for b in row_boxes)
        assert right <= 700, f"row right {right} exceeds budget 700"


def test_max_row_width_with_per_step_sizes_uses_actual_width() -> None:
    """When step_sizes overrides per-step width, wrap uses actual widths not config default."""
    cfg = LayoutConfig(max_row_width=600, step_width=100.0, h_gap=20.0, max_steps_per_row=999)
    # Step "1" is 400 wide, "2" is 250, "3" is 250
    # 1: 0-400 (400). 2: +20+250=670 → exceed 600 → break. 2 alone: 0-250. 3: +20+250=520 fits.
    sizes = {"1": (400.0, 200.0), "2": (250.0, 200.0), "3": (250.0, 200.0)}
    res = compute_layout(_scheme(3), config=cfg, step_sizes=sizes)
    rows: dict[float, list] = {}
    for box in res.boxes:
        rows.setdefault(box.y, []).append(box)
    assert len(rows) == 2
    sizes_per_row = sorted(len(r) for r in rows.values())
    assert sizes_per_row == [1, 2]


def test_step_count_cap_still_honored_when_width_budget_set() -> None:
    """If max_steps_per_row=2 AND max_row_width=10000 (huge), step-count wins → rows of 2."""
    cfg = LayoutConfig(max_steps_per_row=2, max_row_width=10000)
    res = compute_layout(_scheme(5), config=cfg)
    rows: dict[float, list] = {}
    for box in res.boxes:
        rows.setdefault(box.y, []).append(box)
    sizes = sorted(len(r) for r in rows.values())
    assert sizes == [1, 2, 2]


def test_max_row_width_zero_treated_as_disabled() -> None:
    """Defensive: 0 width budget (nonsensical) falls back to step-count behavior."""
    cfg = LayoutConfig(max_row_width=0)
    # Should not infinite-loop, should not break every step trivially
    res = compute_layout(_scheme(3), config=cfg)
    assert len(res.boxes) == 3


# ──────────────────────────── CLI flag ────────────────────────────


def test_cli_draw_accepts_max_row_width_flag(tmp_path) -> None:
    """`chemscheme draw --max-row-width 600 scheme.json -o out.svg` works end-to-end."""
    import json

    src = tmp_path / "five_step.json"
    src.write_text(json.dumps(_scheme(5, layout="horizontal")), encoding="utf-8")
    out = tmp_path / "wrapped.svg"

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["draw", str(src), "-o", str(out), "--max-row-width", "600"],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert out.exists()
    svg = out.read_text(encoding="utf-8")
    # Expect wrapped rendering: SVG height should be > viewBox-height of single row
    # (rough check — at minimum the SVG was produced and is non-empty)
    assert len(svg) > 1000


def test_cli_draw_max_row_width_overrides_horizontal_layout_mode(tmp_path) -> None:
    """Explicit --max-row-width overrides scheme.json `layout: horizontal`.

    Without this, the user can't fix already-authored "horizontal" schemes
    without editing the JSON. CLI flag must take precedence.
    """
    import json

    src = tmp_path / "five_step_horiz.json"
    # layout=horizontal would normally force single row even with 5 steps
    src.write_text(json.dumps(_scheme(5, layout="horizontal")), encoding="utf-8")
    out_unwrapped = tmp_path / "single_row.svg"
    out_wrapped = tmp_path / "wrapped.svg"

    runner = CliRunner()
    res_a = runner.invoke(cli_main, ["draw", str(src), "-o", str(out_unwrapped)])
    res_b = runner.invoke(
        cli_main, ["draw", str(src), "-o", str(out_wrapped), "--max-row-width", "600"]
    )
    assert res_a.exit_code == 0
    assert res_b.exit_code == 0

    # Wrapped output should be visibly different (different viewBox / dimensions)
    assert out_unwrapped.read_bytes() != out_wrapped.read_bytes(), (
        "--max-row-width must change rendering vs no flag for layout=horizontal scheme"
    )


def test_cli_draw_max_row_width_help_documented() -> None:
    """`chemscheme draw --help` documents --max-row-width."""
    runner = CliRunner()
    result = runner.invoke(cli_main, ["draw", "--help"])
    assert result.exit_code == 0
    assert "max-row-width" in result.output.lower()
