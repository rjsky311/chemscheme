# ChemScheme

[![CI](https://github.com/rjsky311/chemscheme/actions/workflows/ci.yml/badge.svg)](https://github.com/rjsky311/chemscheme/actions/workflows/ci.yml)

> Generate publication-quality multi-step reaction schemes from SMILES, names, or natural language descriptions.

**ChemScheme** is a Python CLI + MCP server designed for chemistry researchers who need to create, validate, and render multi-step reaction schemes for publications. It combines RDKit for chemical validation and rendering, drawsvg for SVG composition, and Claude API for natural language intent parsing.

> **Status:** v0.1.0. Built March–April 2026; not under active development since
> April 2026. Not published on PyPI; install from source.

## Features

- **Multi-step scheme rendering** — SVG and PNG output with automatic layout (linear and convergent)
- **Layered validation** — Schema structure, chemical correctness (RDKit), an optional reaction check, and graph continuity
- **Compound name resolution** — Local alias database (100+ reagents), OPSIN, and PubChem fallback
- **AI intent parsing** — Generate `scheme.json` from natural language descriptions via Claude API
- **ChemDraw interop** — CDXML export/import with full round-trip fidelity (metadata, graph semantics)
- **MCP server** — Expose tools for AI agents via the Model Context Protocol
- **Convergent synthesis** — Graph-aware `consumes_step_ids` for branching and merging reaction pathways

## Installation

### From source

```bash
git clone https://github.com/rjsky311/chemscheme.git
cd chemscheme
pip install -e ".[dev]"
```

### Optional extras

```bash
pip install -e ".[mcp]"    # MCP server for AI agents
pip install -e ".[web]"    # Local authoring web UI (Ketcher + image recognition)
pip install -e ".[ai]"     # Claude API for natural language parsing
pip install -e ".[indigo]" # Indigo toolkit (alternative renderer)
pip install -e ".[image]"  # DECIMER for image-to-SMILES
```

### Supported platforms

- Python 3.13, 3.14
- Linux (Ubuntu), Windows, macOS
- RDKit 2024+

## Quick Start

### 1. Create a scheme

Write a `scheme.json` file:

```json
{
  "schema_version": "1",
  "title": "Suzuki Coupling",
  "steps": [
    {
      "id": "1",
      "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
      "products": ["c1ccc(-c2ccccc2)cc1"],
      "arrow_above": "Pd(PPh3)4, K2CO3",
      "arrow_below": "DMF, 80 °C, 12h",
      "yield": "78%"
    }
  ]
}
```

### 2. Validate

```bash
chemscheme validate scheme.json
# Valid scheme. (layers: schema, chemistry, reaction, continuity)

chemscheme validate scheme.json --json
# {"status": "ok", "valid": true, "issues": [], "warnings": [], "layers_checked": [...]}
```

### 3. Render

```bash
chemscheme draw scheme.json -o output.svg
chemscheme draw scheme.json -o output.png
```

### 4. Resolve compound names

```bash
chemscheme resolve "aspirin"
# CC(=O)Oc1ccccc1C(=O)O  (source: pubchem, confidence: 0.85)

chemscheme resolve "Pd(PPh3)4" --json
# {"status": "ok", "smiles": "...", "confidence": 0.95, "source": "local_alias", ...}
```

### 5. Generate from natural language (requires `ANTHROPIC_API_KEY`)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
chemscheme ai "Suzuki coupling of bromobenzene with phenylboronic acid using Pd catalyst" -o scheme.json
```

### 6. Edit schemes

```bash
# Add a step
chemscheme edit scheme.json --add-step '{"id": "2", "reactants": ["c1ccc(-c2ccccc2)cc1"], "products": ["O=Cc1ccc(-c2ccccc2)cc1"], "arrow_above": "DMF, POCl3"}'

# Remove a step
chemscheme edit scheme.json --remove-step "2"
```

### 7. ChemDraw interop

```bash
# Export to CDXML
chemscheme export scheme.json -o scheme.cdxml

# Import from CDXML
chemscheme import scheme.cdxml -o imported.json
```

## scheme.json Format

The intermediate format uses a graph-aware JSON structure:

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `schema_version` | Yes | string | Schema version (currently `"1"`) |
| `title` | No | string | Scheme title |
| `steps` | Yes | array | Reaction steps |

Each step:

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | Yes | string | Unique step identifier |
| `reactants` | Yes | string[] | SMILES strings for reactants |
| `products` | Yes | string[] | SMILES strings for products |
| `arrow_above` | Yes | string | Reagents/conditions above the arrow |
| `arrow_below` | No | string | Conditions below the arrow |
| `yield` | No | string | Reaction yield |
| `reagents` | No | string[] | Reagent list |
| `solvents` | No | string[] | Solvent list |
| `temperature` | No | string | Reaction temperature |
| `time` | No | string | Reaction time |
| `atmosphere` | No | string | Atmosphere (e.g., N2, Ar) |
| `branch_label` | No | string | Branch identifier for parallel synthesis |
| `consumes_step_ids` | No | string[] | Step IDs consumed (for convergent synthesis) |

### Convergent Synthesis Example

```json
{
  "schema_version": "1",
  "title": "Convergent Stille Coupling",
  "steps": [
    {
      "id": "1a",
      "reactants": ["c1ccc(Br)cc1"],
      "products": ["c1ccc(Br)c(Br)c1"],
      "arrow_above": "NBS",
      "branch_label": "donor unit"
    },
    {
      "id": "1b",
      "reactants": ["c1ccncc1"],
      "products": ["c1cc([Sn](C)(C)C)ncc1"],
      "arrow_above": "n-BuLi, SnMe3Cl",
      "branch_label": "acceptor unit"
    },
    {
      "id": "2",
      "consumes_step_ids": ["1a", "1b"],
      "reactants": ["c1ccc(Br)c(Br)c1", "c1cc([Sn](C)(C)C)ncc1"],
      "products": ["c1ccc(-c2ccncc2)c(-c2ccncc2)c1"],
      "arrow_above": "Pd(PPh3)4",
      "yield": "72%"
    }
  ]
}
```

## MCP Server

ChemScheme exposes tools via the [Model Context Protocol](https://modelcontextprotocol.io/) for AI agent consumption.

### Tools

| Tool | Description |
|------|-------------|
| `draw_scheme` | Render a scheme.json to SVG or PNG |
| `validate_scheme` | Validate a scheme for chemical correctness |
| `resolve_name` | Resolve a chemical compound name to SMILES |
| `extract_image` | Recognize a structure image and return SMILES with confidence |
| `extract_text` | Turn an experimental-section paragraph into scheme steps |
| `generate_procedure` | Draft an experimental procedure from a scheme |
| `lookup_reagent`, `suggest_conditions` | Reagent lookup and typical conditions by reaction type |
| `add_user_reagent`, `remove_user_reagent`, `list_user_reagents` | Manage a personal reagent alias list |

### Running the MCP server

```bash
chemscheme-mcp
```

Or configure in your MCP client (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "chemscheme": {
      "command": "chemscheme-mcp"
    }
  }
}
```

## Architecture

```
chemscheme/
  parsers/
    name_resolver.py      # Compound name -> SMILES (alias/OPSIN/PubChem)
    ai_intent.py          # Natural language -> scheme.json (Claude API)
    smiles_utils.py       # SMILES canonicalization utilities
  validators/
    layer1_chemistry.py   # RDKit chemical correctness
    layer2_reaction.py    # Reaction feasibility (Protocol-based)
    layer3_scheme.py      # Graph continuity validation
  renderers/
    indigo_renderer.py    # Single-step SVG/PNG rendering
    layout_engine.py      # Multi-step coordinate math (linear + convergent)
    multi_step_renderer.py# Composed multi-step rendering
    drawsvg_composer.py   # Publication-quality SVG composition
    themes.py             # Journal theme presets (ACS, Nature, RSC, Wiley)
    cdxml_writer.py       # CDXML export/import for ChemDraw
  editors/
    scheme_editor.py      # Non-mutating add/remove step operations
  schema/
    scheme_schema.py      # scheme.json validation and I/O
  data/
    reagent_aliases.json  # 100+ common reagent abbreviations with SMILES
    lab_context.json      # Lab preferences and style configuration
  models.py               # SmilesResult, ValidationResult types
  cli.py                  # Click CLI entry point
  mcp_server.py           # FastMCP server
```

### Validation Layers

1. **Schema** — JSON structure, required fields, type checks, DAG ordering
2. **Chemistry** — RDKit SMILES parsing, valence checks, stereochemistry warnings
3. **Reaction** — Optional product check through a pluggable `ReactionPredictor`; skipped with a warning when none is configured
4. **Continuity** — Product-reactant chain connectivity across steps, branch grouping

### Design Decisions

- **AI drafts, validators decide** — The `ai` command asks the Claude API for a draft `scheme.json`, SMILES included. The draft passes schema checks only; run `chemscheme validate` for the chemistry and reaction layers before using it
- **SmilesResult unified type** — All SMILES operations return a frozen dataclass with confidence and source tracking
- **Graph semantics** — `consumes_step_ids` supports convergent synthesis (OPV, NFA pathways)
- **Atomic writes** — All file saves use temp-file-then-replace with Windows retry logic
- **Thread safety** — Indigo rendering uses a lock; alias loading uses double-checked locking

## Development

### Setup

```bash
pip install -e ".[dev,indigo]"
```

### Quality gate (runs before every commit)

```bash
ruff check . && ruff format --check . && mypy chemscheme/ && pytest -v
```

### Test suite

```bash
pytest -v                    # All tests (1,340 pass; 7 skip without Docker/DECIMER)
pytest -m unit               # Unit tests only
pytest -m chemistry          # Chemistry-specific tests
pytest -m smoke              # Environment verification
```

### Test markers

| Marker | Description |
|--------|-------------|
| `unit` | Fast, no external dependencies |
| `integration` | May use external services |
| `smoke` | Environment and dependency verification |
| `chemistry` | Chemistry-specific (SMILES, reactions) |
| `visual` | SVG golden file regression |
| `slow` | Tests exceeding 5 seconds |

## Project Status

Built in about five weeks (March 14 – April 19, 2026), then paused.

| Area | Status |
|-------|--------|
| Core: schema, rendering, three-layer validation, CLI, MCP, CDXML, name resolution | Done |
| Chemical interchange (charge, isotope, stereo) and drawsvg production renderer | Done |
| Publication annotations, convergent and L-shaped layouts, polymer brackets | Done |
| Journal themes (ACS, Nature, RSC, Wiley) | Done |
| Structure recognition from images and PDFs (OCSR ensemble in Docker, cross-validation, human review queue) | Done; real-tool accuracy depends on the Docker images you build |
| Reagent library, condition suggestions, text extraction, procedure drafts | Done |
| Local authoring web UI (Ketcher) | Done |
| PyPI release | Workflow written, not published |

## How It Was Built

ChemScheme was designed and directed by one synthetic chemist, Yu-Che Lin
(PhD, organic electronic materials). Yu-Che Lin set the chemistry rules and
acceptance criteria; Claude Code (Anthropic) wrote the code and tests; OpenAI
Codex and Google Gemini reviewed the plans and code at key phases. See
[docs/CASE_STUDY.md](docs/CASE_STUDY.md).

## License

MIT. See [LICENSE](LICENSE).
