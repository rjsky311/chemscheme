"""ChemScheme web/authoring layer (G6 — S68).

Provides a local FastAPI server wrapping a Ketcher-based authoring UI plus
bridge endpoints to:
- SMILES ↔ MOL block conversion (via RDKit)
- Phase 17 OCSR (image → SMILES + mol_block for canvas hydration)
- scheme.json load/save with concurrent-modification detection

Optional dependency: install via `pip install chemscheme[web]`.
"""

from chemscheme.web.app import create_app

__all__ = ["create_app"]
