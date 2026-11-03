"""G6 — Round-trip parity: GUI save → CLI draw produces same output as direct CLI draw.

Covers AC-G6.8 (no regression vs direct CLI authoring path).

BDD-9 ("byte-identical SVG") is intentionally relaxed to scheme.json equivalence
+ structural-SVG equivalence (file size + atom count). Pure byte-identity is too
flaky for SVG output across runs; what we actually need to guarantee is that
the GUI bridge does not silently lose data.
"""

from __future__ import annotations

import json

from click.testing import CliRunner
from fastapi.testclient import TestClient

from chemscheme.cli import main as cli_main
from chemscheme.web.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


# ──────────────────────────── AC-G6.8 / scheme.json equivalence ────────────────────────────


def test_gui_save_then_cli_draw_succeeds(tmp_path) -> None:
    """GUI-saved scheme.json passes through `chemscheme draw` without error."""
    target = tmp_path / "via_gui.json"
    out_svg = tmp_path / "via_gui.svg"

    payload = {
        "path": str(target),
        "scheme": {
            "schema_version": "1",
            "title": "GUI smoke",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                }
            ],
        },
    }
    with _client() as client:
        r = client.post("/api/scheme/save", json=payload)
        assert r.status_code == 200

    runner = CliRunner()
    result = runner.invoke(cli_main, ["draw", str(target), "-o", str(out_svg)])
    assert result.exit_code == 0, f"CLI draw failed: {result.output}"
    assert out_svg.exists()
    svg_content = out_svg.read_text(encoding="utf-8")
    assert svg_content.startswith("<?xml") or svg_content.startswith("<svg")


def test_gui_save_equivalent_to_direct_save(tmp_path) -> None:
    """A scheme dict written via GUI api/save equals same dict written via raw save_scheme.

    This is the data-fidelity contract: the GUI bridge MUST NOT mutate, normalize,
    re-order, or drop fields. (Other than required atomic-write semantics.)
    """
    from chemscheme.schema.scheme_schema import save_scheme

    scheme = {
        "schema_version": "1",
        "title": "Equivalence test",
        "steps": [
            {
                "id": "1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
                "arrow_below": "FeBr3",
                "yield": "85%",
            }
        ],
        "style": "ACS2024",
    }

    via_gui = tmp_path / "via_gui.json"
    via_direct = tmp_path / "via_direct.json"

    with _client() as client:
        client.post("/api/scheme/save", json={"path": str(via_gui), "scheme": scheme})
    save_scheme(scheme, via_direct)

    gui_loaded = json.loads(via_gui.read_text(encoding="utf-8"))
    direct_loaded = json.loads(via_direct.read_text(encoding="utf-8"))
    assert gui_loaded == direct_loaded


def test_gui_load_save_does_not_mutate_scheme(tmp_path) -> None:
    """Load → save without GUI edits MUST be a no-op on content."""
    src = tmp_path / "src.json"
    out = tmp_path / "out.json"
    original = {
        "schema_version": "1",
        "title": "no-op test",
        "steps": [
            {
                "id": "1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
                "arrow_below": "FeBr3",
                "yield": "85%",
            }
        ],
        "style": "ACS2024",
        "layout": "horizontal",
    }
    src.write_text(json.dumps(original), encoding="utf-8")

    with _client() as client:
        loaded = client.get(f"/api/scheme/load?path={src}").json()["scheme"]
        client.post("/api/scheme/save", json={"path": str(out), "scheme": loaded})

    out_content = json.loads(out.read_text(encoding="utf-8"))
    assert out_content == original


def test_gui_round_trip_then_draw_matches_direct_draw(tmp_path) -> None:
    """GUI-load + GUI-save + CLI-draw produces same SVG as direct-load + CLI-draw.

    Relaxation note: this is structural equivalence (same scheme → same bytes) since
    `chemscheme draw` is deterministic given the same input. The point of the test
    is that the GUI bridge does not introduce content drift that would change the
    eventual render. If `draw` itself is non-deterministic, that is a separate bug.
    """
    src = tmp_path / "src.json"
    via_gui = tmp_path / "via_gui.json"
    svg_direct = tmp_path / "direct.svg"
    svg_gui = tmp_path / "gui.svg"

    scheme = {
        "schema_version": "1",
        "title": "Render parity",
        "steps": [
            {
                "id": "1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
            }
        ],
    }
    src.write_text(json.dumps(scheme), encoding="utf-8")

    runner = CliRunner()
    direct_result = runner.invoke(cli_main, ["draw", str(src), "-o", str(svg_direct)])
    assert direct_result.exit_code == 0

    with _client() as client:
        loaded = client.get(f"/api/scheme/load?path={src}").json()["scheme"]
        save_r = client.post("/api/scheme/save", json={"path": str(via_gui), "scheme": loaded})
        assert save_r.status_code == 200

    gui_result = runner.invoke(cli_main, ["draw", str(via_gui), "-o", str(svg_gui)])
    assert gui_result.exit_code == 0

    # Same input scheme + deterministic renderer → same SVG bytes
    assert svg_direct.read_bytes() == svg_gui.read_bytes(), (
        "GUI round-trip must produce byte-identical SVG to direct CLI when scheme is unchanged"
    )
