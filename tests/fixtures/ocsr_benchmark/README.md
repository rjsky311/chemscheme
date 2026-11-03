# OCSR Benchmark Fixture

Starter ground-truth set for `chemscheme benchmark-ocsr` (S68 Part 4).

## Structure

- `ground_truth.json` — image filename stem → canonical SMILES (RDKit-validated)
- `*.png` / `*.jpg` — molecule images matching the stems above (user-supplied)

## Usage

```bash
# Real-tool mode (requires Docker daemon + per-tool images built):
docker build -t chemscheme-molscribe docker/molscribe/
docker build -t chemscheme-decimer docker/decimer/
docker build -t chemscheme-molnextr docker/molnextr/
export CHEMSCHEME_OCSR_DOCKER=1

chemscheme benchmark-ocsr \
  --input-dir tests/fixtures/ocsr_benchmark/ \
  --ground-truth tests/fixtures/ocsr_benchmark/ground_truth.json \
  --output report.json

# Mock mode (framework smoke test only — no real accuracy):
chemscheme benchmark-ocsr \
  --input-dir tests/fixtures/ocsr_benchmark/ \
  --ground-truth tests/fixtures/ocsr_benchmark/ground_truth.json \
  --output report.json --mock
```

## Growing the set

Drop additional molecule images here, add the canonical SMILES to
`ground_truth.json`. Real chemistry-paper-grade benchmark requires
user-supplied OPV/NFA test images — the entries above verify the framework
on simple cases.

## S67 Part 5b context

The Mathpix "high accuracy" assumption was unverified marketing
(99% general-OCR claim, not chemistry-specific, not in MARCUS Oct 2025
ensemble). This fixture + `benchmark-ocsr` lets users measure the
4-tool ensemble (MolScribe + DECIMER + MolNexTR + Mathpix) empirically
on identical inputs before any keep/drop/reweight decision.
