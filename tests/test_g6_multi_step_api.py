"""G6 — Multi-step assembly + load existing scheme.json + concurrent save.

Covers AC-G6.6 (multi-molecule reaction → multi-step scheme.json save),
AC-G6.7 (load existing scheme.json with polymer/Markush per Phase 13).

BDD scenarios: BDD-3 (polymer round-trip preserves polymer_info), BDD-8 (concurrent
save → last-write-wins WITH explicit warning, no silent loss).
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from chemscheme.web.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


# ──────────────────────────── AC-G6.6 / multi-step assembly ────────────────────────────


def test_save_scheme_multi_step_with_conditions(tmp_path) -> None:
    """Two-step Suzuki coupling with arrow_above + arrow_below + yield → valid scheme.json."""
    target = tmp_path / "two_step.json"
    payload = {
        "path": str(target),
        "scheme": {
            "schema_version": "1",
            "title": "Two-step Suzuki",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccc(C)cc1"],
                    "products": ["c1ccc(-c2ccc(C)cc2)cc1"],
                    "arrow_above": "Pd(PPh3)4, K2CO3",
                    "arrow_below": "DMF, 80 C, 12h",
                    "yield": "78%",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(-c2ccc(C)cc2)cc1"],
                    "products": ["c1ccc(-c2ccc(C(=O)O)cc2)cc1"],
                    "arrow_above": "KMnO4",
                    "arrow_below": "H2O, reflux",
                    "yield": "65%",
                },
            ],
        },
    }
    with _client() as client:
        r = client.post("/api/scheme/save", json=payload)
        assert r.status_code == 200
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert len(on_disk["steps"]) == 2
    assert on_disk["steps"][0]["arrow_above"] == "Pd(PPh3)4, K2CO3"
    assert on_disk["steps"][1]["yield"] == "65%"


def test_save_scheme_with_arrow_above_list_preserved(tmp_path) -> None:
    """arrow_above as list (multi-line conditions) preserved by save endpoint."""
    target = tmp_path / "multi_line_arrow.json"
    payload = {
        "path": str(target),
        "scheme": {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": ["Br2", "FeBr3 (cat.)"],
                }
            ],
        },
    }
    with _client() as client:
        r = client.post("/api/scheme/save", json=payload)
        assert r.status_code == 200
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["steps"][0]["arrow_above"] == ["Br2", "FeBr3 (cat.)"]


# ──────────────────────────── AC-G6.7 / load existing ────────────────────────────


def test_load_existing_scheme_returns_full_content(tmp_path) -> None:
    """GET /api/scheme/load?path=... returns the parsed scheme + per-mol mol_blocks for canvas."""
    src = tmp_path / "existing.json"
    src.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "title": "Pre-existing",
                "steps": [
                    {
                        "id": "1",
                        "reactants": ["c1ccccc1"],
                        "products": ["c1ccc(Br)cc1"],
                        "arrow_above": "Br2",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with _client() as client:
        r = client.get(f"/api/scheme/load?path={src}")
        assert r.status_code == 200
        body = r.json()
        assert body["scheme"]["title"] == "Pre-existing"
        # Server should include per-SMILES mol_blocks so client can hydrate Ketcher canvas
        assert "mol_blocks" in body
        # mol_blocks keyed by SMILES; benzene + bromobenzene
        assert "c1ccccc1" in body["mol_blocks"]
        assert "M  END" in body["mol_blocks"]["c1ccccc1"]
        # mtime included for concurrent-save reconciliation
        assert "mtime" in body


def test_load_nonexistent_scheme_returns_404(tmp_path) -> None:
    """Loading a nonexistent file → 404, not 500."""
    missing = tmp_path / "does_not_exist.json"
    with _client() as client:
        r = client.get(f"/api/scheme/load?path={missing}")
        assert r.status_code == 404


def test_load_scheme_invalid_json_returns_400(tmp_path) -> None:
    """File exists but is malformed JSON → 400 with informative error."""
    bad = tmp_path / "bad.json"
    bad.write_text("not { valid json", encoding="utf-8")
    with _client() as client:
        r = client.get(f"/api/scheme/load?path={bad}")
        assert r.status_code == 400


# ──────────────────────────── BDD-3 / polymer round-trip ────────────────────────────


def test_polymer_round_trip_preserves_polymer_info(tmp_path) -> None:
    """Phase 13 polymer_info MUST round-trip through load → save unchanged.

    Schema: polymer_info is dict[SMILES, dict] per AC-13.1.
    """
    src = tmp_path / "polymer.json"
    polymer_smiles = "*CC(c1ccccc1)*"
    src.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "title": "Polymer round-trip",
                "steps": [
                    {
                        "id": "1",
                        "reactants": ["c1ccc(Br)cc1"],
                        "products": [polymer_smiles],
                        "arrow_above": "polymerize",
                        "polymer_info": {
                            polymer_smiles: {"n": "n", "min_n": 10},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "polymer_resaved.json"
    with _client() as client:
        load_r = client.get(f"/api/scheme/load?path={src}")
        assert load_r.status_code == 200
        loaded = load_r.json()["scheme"]

        save_r = client.post("/api/scheme/save", json={"path": str(out), "scheme": loaded})
        assert save_r.status_code == 200

    resaved = json.loads(out.read_text(encoding="utf-8"))
    pi = resaved["steps"][0]["polymer_info"]
    assert polymer_smiles in pi
    assert pi[polymer_smiles]["n"] == "n"
    assert pi[polymer_smiles]["min_n"] == 10


def test_substituent_map_round_trip_preserved(tmp_path) -> None:
    """Phase 13 substituent_map (Markush R-groups) round-trips."""
    src = tmp_path / "markush.json"
    src.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "steps": [
                    {
                        "id": "1",
                        "reactants": ["[*:1]c1ccccc1"],
                        "products": ["[*:1]c1ccc(Br)cc1"],
                        "arrow_above": "Br2",
                        "substituent_map": {"R1": ["CH3", "C2H5", "iPr"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "markush_resaved.json"
    with _client() as client:
        loaded = client.get(f"/api/scheme/load?path={src}").json()["scheme"]
        client.post("/api/scheme/save", json={"path": str(out), "scheme": loaded})

    resaved = json.loads(out.read_text(encoding="utf-8"))
    assert resaved["steps"][0]["substituent_map"]["R1"] == ["CH3", "C2H5", "iPr"]


# ──────────────────────────── BDD-8 / concurrent save ────────────────────────────


def test_concurrent_save_flags_external_modification(tmp_path) -> None:
    """If file's mtime changed since client loaded it → save returns 200 with WARNING.

    Last-write-wins semantics, but the response must announce the conflict explicitly
    so the GUI can show the user "another tab/process saved meanwhile". No silent loss.
    """
    target = tmp_path / "shared.json"
    initial = {
        "schema_version": "1",
        "steps": [
            {
                "id": "1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
            }
        ],
    }
    target.write_text(json.dumps(initial), encoding="utf-8")

    with _client() as client:
        # Tab 1 loads
        load_r = client.get(f"/api/scheme/load?path={target}")
        assert load_r.status_code == 200
        load_mtime = load_r.json()["mtime"]

        # External modification (tab 2 / another process) bumps mtime
        import time

        time.sleep(0.05)  # ensure mtime resolution is exceeded
        external = dict(initial)
        external["title"] = "modified by tab 2"
        target.write_text(json.dumps(external), encoding="utf-8")

        # Tab 1 saves with stale base_mtime
        save_r = client.post(
            "/api/scheme/save",
            json={
                "path": str(target),
                "scheme": initial,
                "base_mtime": load_mtime,
            },
        )
        assert save_r.status_code == 200
        body = save_r.json()
        # Last-write-wins is fine, but warning MUST surface
        assert body.get("warning"), "external-modification warning must be present"
        warn_msg = body["warning"].lower()
        assert "modif" in warn_msg or "concurrent" in warn_msg or "stale" in warn_msg


def test_save_without_base_mtime_skips_concurrency_check(tmp_path) -> None:
    """If client doesn't send base_mtime (e.g., new file), no warning generated."""
    target = tmp_path / "fresh.json"
    payload = {
        "path": str(target),
        "scheme": {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["CO"],
                    "arrow_above": "?",
                }
            ],
        },
    }
    with _client() as client:
        r = client.post("/api/scheme/save", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert not body.get("warning")
