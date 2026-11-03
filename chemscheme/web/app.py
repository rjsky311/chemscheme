"""FastAPI application for ChemScheme authoring (G6 — S68).

Endpoints:
- GET  /                  — Ketcher-loader HTML page
- GET  /api/health        — liveness probe
- POST /api/smiles_to_mol — SMILES → MOL block (paste path)
- POST /api/mol_to_smiles — MOL block → SMILES (canvas-save path)
- POST /api/recognize     — image upload → SMILES via Phase 17 OCSR
- POST /api/scheme/save   — write scheme.json (atomic + concurrent-mod warning)
- GET  /api/scheme/load   — read scheme.json + per-SMILES mol_blocks for canvas hydration

The OCSR recognizer list is supplied via the `get_recognizers` dependency so
tests can inject stubs and production deployments can wire MolScribe / DECIMER /
Mathpix per Phase 17c.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from rdkit import RDLogger

from chemscheme.recognition.ocsr import recognize_molecule
from chemscheme.recognition.protocols import MoleculeRecognizer
from chemscheme.schema.scheme_schema import load_scheme, save_scheme, validate_scheme
from chemscheme.web.bridge import mol_block_to_smiles, smiles_to_mol_block

# RDKit emits noisy warnings to stderr for unparseable input; we surface errors
# as HTTP responses instead, so silence the underlying log.
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

STATIC_DIR = Path(__file__).parent / "static"


def get_recognizers() -> list[MoleculeRecognizer]:
    """FastAPI dependency: list of OCSR recognizers wired into /api/recognize.

    Default returns []. Tests inject stubs via app.dependency_overrides.
    Production wiring (MolScribe + DECIMER + Mathpix) is S69+ scope.
    """
    return []


def _build_index_html() -> str:
    """Read the static index.html (Ketcher-loader page).

    Done at request-time (not import-time) so a missing file produces a clean
    HTTP 500 instead of an import failure.
    """
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


def create_app() -> FastAPI:
    app = FastAPI(title="ChemScheme Authoring", version="0.1.0")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _build_index_html()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/smiles_to_mol")
    async def smiles_to_mol_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        if "smiles" not in payload:
            raise HTTPException(status_code=422, detail="missing 'smiles' field")
        smiles = payload["smiles"]
        if not isinstance(smiles, str) or not smiles:
            raise HTTPException(status_code=400, detail="invalid SMILES: empty")
        try:
            mol_block, canonical = smiles_to_mol_block(smiles)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid SMILES: {e}") from e
        return {
            "valid": True,
            "mol_block": mol_block,
            "canonical_smiles": canonical,
        }

    @app.post("/api/mol_to_smiles")
    async def mol_to_smiles_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        if "mol_block" not in payload:
            raise HTTPException(status_code=422, detail="missing 'mol_block' field")
        mol_block = payload["mol_block"]
        if not isinstance(mol_block, str) or not mol_block:
            raise HTTPException(status_code=400, detail="invalid mol_block: empty")
        try:
            smiles = mol_block_to_smiles(mol_block)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid mol_block: {e}") from e
        return {"valid": True, "smiles": smiles}

    @app.post("/api/recognize")
    async def recognize_endpoint(
        image: UploadFile = File(...),  # noqa: B008  — FastAPI dependency-injection idiom
        recognizers: list[MoleculeRecognizer] = Depends(get_recognizers),  # noqa: B008
    ) -> dict[str, Any]:
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty image upload")
        if not recognizers:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No OCSR recognizers configured. Install the 'image' extra "
                    "(`pip install chemscheme[image]`) and ensure the docker daemon is running."
                ),
            )
        try:
            result = recognize_molecule(image=data, recognizers=recognizers)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"OCSR recognizer error: {e}. Check that the docker container is reachable."
                ),
            ) from e
        mol_block = ""
        if result.value:
            try:
                mol_block, _ = smiles_to_mol_block(result.value)
            except ValueError:
                mol_block = ""
        return {
            "smiles": result.value,
            "confidence": result.confidence,
            "source": result.source,
            "agreement": result.agreement.value,
            "needs_review": bool(result.needs_human_review or result.confidence < 0.5),
            "mol_block": mol_block,
        }

    @app.post("/api/scheme/save")
    async def save_scheme_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        path_str = payload.get("path")
        scheme = payload.get("scheme")
        base_mtime = payload.get("base_mtime")
        if not isinstance(path_str, str) or not path_str:
            raise HTTPException(status_code=422, detail="'path' string is required")
        if not isinstance(scheme, dict):
            raise HTTPException(status_code=422, detail="'scheme' object is required")
        errors = validate_scheme(scheme)
        if errors:
            raise HTTPException(status_code=400, detail={"errors": errors})

        target = Path(path_str)
        warning: str | None = None
        if base_mtime is not None and target.exists():
            current = target.stat().st_mtime
            try:
                base = float(base_mtime)
            except (TypeError, ValueError):
                base = 0.0
            if current > base:
                warning = (
                    f"file modified externally since load (mtime {current} > base {base}); "
                    "your save overwrites the external change (last-write-wins)"
                )

        save_scheme(scheme, target)
        body: dict[str, Any] = {
            "status": "ok",
            "path": str(target),
            "mtime": target.stat().st_mtime,
        }
        if warning:
            body["warning"] = warning
        return body

    @app.get("/api/scheme/load")
    async def load_scheme_endpoint(path: str) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"file not found: {p}")
        try:
            scheme = load_scheme(p)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        mol_blocks: dict[str, str] = {}
        for step in scheme.get("steps", []):
            if not isinstance(step, dict):
                continue
            for s in list(step.get("reactants", [])) + list(step.get("products", [])):
                if isinstance(s, str) and s and s not in mol_blocks:
                    try:
                        mb, _ = smiles_to_mol_block(s)
                        mol_blocks[s] = mb
                    except ValueError:
                        # Non-SMILES strings (Markush wildcards, etc.) silently skipped
                        # — Ketcher will fall back to its own parse attempt.
                        continue

        return {
            "scheme": scheme,
            "mol_blocks": mol_blocks,
            "mtime": p.stat().st_mtime,
        }

    return app
