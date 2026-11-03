"""G6 — Authoring API: SMILES↔mol-block bridge for Ketcher integration.

Covers AC-G6.3 (Ketcher canvas → scheme.json single-mol save)
and AC-G6.4 (paste SMILES → canvas-loadable mol_block).

BDD scenarios: BDD-1 (draw benzene → save → c1ccccc1), BDD-4 (invalid SMILES → error,
no canvas corruption).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from chemscheme.web.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


# ──────────────────────────── AC-G6.4 / paste path ────────────────────────────


def test_smiles_to_mol_endpoint_benzene() -> None:
    """POST /api/smiles_to_mol with benzene SMILES → MOL block + canonical SMILES."""
    with _client() as client:
        r = client.post("/api/smiles_to_mol", json={"smiles": "c1ccccc1"})
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is True
        assert "mol_block" in body and isinstance(body["mol_block"], str)
        # MOL V2000 block has these markers
        assert "M  END" in body["mol_block"]
        # Canonical form preserved
        assert body["canonical_smiles"] == "c1ccccc1"


def test_smiles_to_mol_endpoint_complex_nfa_fragment() -> None:
    """A complex NFA-relevant fragment (4,7-dibromo-2,1,3-benzothiadiazole) round-trips.

    This is a real Y-series acceptor building block; if the bridge can't handle
    it, the user can't author NFAs — defeating the whole point of S68.
    """
    nfa_fragment = "Brc1ccc2nsnc2c1Br"
    with _client() as client:
        r = client.post("/api/smiles_to_mol", json={"smiles": nfa_fragment})
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is True
        assert "M  END" in body["mol_block"]
        # Atoms preserved
        assert "Br" in body["mol_block"]
        assert "S" in body["mol_block"]
        assert "N" in body["mol_block"]


def test_smiles_to_mol_endpoint_invalid_returns_400() -> None:
    """BDD-4: invalid SMILES `C(C(C` → 400 with error message; no server crash."""
    with _client() as client:
        r = client.post("/api/smiles_to_mol", json={"smiles": "C(C(C"})
        assert r.status_code == 400
        body = r.json()
        # Error must be informative
        assert "error" in body or "detail" in body
        err_msg = (body.get("error") or body.get("detail") or "").lower()
        assert "smiles" in err_msg or "invalid" in err_msg or "parse" in err_msg


def test_smiles_to_mol_endpoint_empty_returns_400() -> None:
    """Empty SMILES is rejected, not silently accepted as empty mol."""
    with _client() as client:
        r = client.post("/api/smiles_to_mol", json={"smiles": ""})
        assert r.status_code == 400


def test_smiles_to_mol_endpoint_missing_field_returns_422() -> None:
    """Missing 'smiles' key → FastAPI validation 422 (not 500)."""
    with _client() as client:
        r = client.post("/api/smiles_to_mol", json={})
        assert r.status_code == 422


# ──────────────────────────── AC-G6.3 / Ketcher → save path ────────────────────────────


def test_mol_to_smiles_endpoint_benzene() -> None:
    """BDD-1: MOL block (as Ketcher emits) → canonical SMILES."""
    benzene_mol = """
  Mrv2014

  6  6  0  0  0  0            999 V2000
   -0.7145    0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7145   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.8250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7145   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7145    0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.8250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
M  END
"""
    with _client() as client:
        r = client.post("/api/mol_to_smiles", json={"mol_block": benzene_mol})
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is True
        # Canonical benzene
        assert body["smiles"] == "c1ccccc1"


def test_mol_to_smiles_endpoint_invalid_block_returns_400() -> None:
    """Garbage MOL block → 400, not 500; canvas state on client side is preserved."""
    with _client() as client:
        r = client.post("/api/mol_to_smiles", json={"mol_block": "not a mol block"})
        assert r.status_code == 400


def test_mol_to_smiles_endpoint_empty_returns_400() -> None:
    """Empty mol_block rejected explicitly."""
    with _client() as client:
        r = client.post("/api/mol_to_smiles", json={"mol_block": ""})
        assert r.status_code == 400


# ──────────────────────────── Save scheme.json (single-mol step) ────────────────────────────


def test_save_scheme_single_mol_step(tmp_path) -> None:
    """POST /api/scheme/save with single-step single-mol scheme writes valid scheme.json."""
    target = tmp_path / "single_mol.json"
    payload = {
        "path": str(target),
        "scheme": {
            "schema_version": "1",
            "title": "Single-mol smoke",
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
        body = r.json()
        assert body["status"] == "ok"
        assert "mtime" in body  # for concurrent-save reconciliation later
    # File written
    assert target.exists()
    import json

    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["steps"][0]["reactants"] == ["c1ccccc1"]


def test_save_scheme_validates_before_write(tmp_path) -> None:
    """Invalid scheme (missing required fields) → 400, no file written."""
    target = tmp_path / "should_not_exist.json"
    payload = {
        "path": str(target),
        "scheme": {
            # missing schema_version + steps
            "title": "broken",
        },
    }
    with _client() as client:
        r = client.post("/api/scheme/save", json=payload)
        assert r.status_code == 400
    assert not target.exists(), "invalid scheme must not produce a file"
