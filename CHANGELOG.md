# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-04-19

First feature-complete version. Not published on PyPI.

### Added
- `scheme.json` format with graph-aware convergent steps (`consumes_step_ids`).
- Three-layer validation: schema, chemistry (RDKit), reaction feasibility, and
  step-to-step continuity.
- Multi-step SVG/PNG rendering with linear, convergent, and L-shaped layouts;
  yields, labels, multiline conditions, polymer brackets.
- Journal theme presets: ACS, Nature, RSC, Wiley.
- Compound name resolution: local reagent aliases, OPSIN, PubChem.
- CDXML export and import for ChemDraw round trips.
- Natural-language drafting of `scheme.json` through the Claude API (schema-checked;
  chemistry checks run through `chemscheme validate`).
- Structure recognition from images and PDFs: OCSR tools in Docker,
  cross-validation, and a human review queue.
- Reagent library, condition suggestions, text extraction from experimental
  sections, and procedure drafts.
- MCP server exposing the main tools to AI agents.
- Local authoring web UI with Ketcher.
- CI on Linux, Windows, and macOS; PyPI publish workflow (unused).

### Changed (public release preparation, 2026-11)
- Pinned `mcp<2` and `rdkit<2026.9` so the test suite passes as recorded.
- Removed internal planning notes and agent session logs from the repository.
