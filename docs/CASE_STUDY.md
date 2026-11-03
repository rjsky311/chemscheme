# Case Study: ChemScheme

A reaction-scheme generator specified by a synthetic chemist and built by AI
coding agents.

## The problem

Every synthesis paper needs multi-step reaction schemes. Drawing them by hand
in ChemDraw is slow, and the drawing is not checked: a wrong SMILES, a broken
step-to-step link, or a missing reagent only shows up when someone reads the
figure closely. The goal was a tool that takes structured data (SMILES, names,
or a description) and produces a publication-ready scheme that has already
been validated.

## What it does

- Reads a `scheme.json` file of steps and draws SVG/PNG schemes in linear,
  convergent, or L-shaped layouts, with journal themes (ACS, Nature, RSC,
  Wiley).
- Validates in layers: schema, chemistry (RDKit parsing, valence, stereo),
  continuity (each product feeds the next step), and an optional product
  check when a reaction predictor is configured.
- Resolves names to structures through a local reagent list, OPSIN, and
  PubChem, and records where each SMILES came from and how confident it is.
- Exports and imports CDXML for ChemDraw round trips.
- Recognizes structures from images and PDFs with several OCSR tools run in
  Docker, cross-checks their answers, and queues disagreements for human
  review.
- Exposes the main operations to AI agents as an MCP server, and offers a
  local web editor built on Ketcher.

## How it was built

One person owns the project: Yu-Che Lin, a synthetic chemist (PhD, organic
electronic materials). Most of the code was written by Anthropic Claude
(Claude Code); OpenAI Codex and Google Gemini reviewed plans and code at key
phases.

| Owner | Agents |
|---|---|
| Wrote the chemistry rules, the scheme format, and acceptance criteria for each phase | Implemented each phase test-first against those criteria |
| Decided what counts as correct output for a chemist | Reviewed each other's plans and code; findings became regression tests |

The work ran from March 14 to April 19, 2026: about 230 commits, about 11,900
lines of package code, and about 22,700 lines of tests.

## Engineering practices

- **Acceptance criteria before code.** Each phase started from written
  criteria and scenarios. Tests were written to fail first, then the
  implementation made them pass.
- **Review findings become tests.** Issues found by cross-model review (for
  example, blocking calls inside async MCP tools, unknown theme names, CDXML
  round trips that dropped convergent links) each have a regression test in
  `tests/test_review_regressions.py`, about 60 test classes in all.
- **Provenance on every structure.** Every SMILES operation returns one result
  type that carries the source (local alias, OPSIN, PubChem, recognition tool)
  and a confidence score, so callers can always see where a structure came
  from.
- **Golden-file checks for drawings.** Reference SVGs catch unintended changes
  in rendered output.
- **Quality gate.** `ruff`, `ruff format`, `mypy`, and the full test suite run
  in CI on Linux and Windows. At release: 1,340 tests pass, 7 skip without
  Docker or DECIMER, and `ruff` and `mypy` report no issues.

## Limits

- Not published on PyPI; install from source.
- Development stopped in April 2026. Preparing this public copy in November
  2026 found two dependency changes since then (`mcp` 2.x renamed its server
  class; RDKit 2026.09 changed drawing output). Both are pinned in
  `pyproject.toml`.
- The `ai` command drafts a whole scheme, SMILES included, with the Claude API.
  The draft is schema-checked only; run `chemscheme validate` before use.
- Structure recognition accuracy depends on the OCSR images you build. The
  benchmark fixture is a starter set, not a published accuracy figure.

## Code

This repository. MIT License.
