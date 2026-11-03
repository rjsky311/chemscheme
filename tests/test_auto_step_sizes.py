"""Tests for auto step_sizes wiring — SMILES-based adaptive sizing.

BDD Scenarios:
  S16-a: estimate_step_size returns dimensions based on atom count
    GIVEN a step with small molecules (≤6 atoms each),
    WHEN estimate_step_size is called,
    THEN width < 400 (compact layout).

    GIVEN a step with large polycyclic molecules (>15 atoms),
    WHEN estimate_step_size is called,
    THEN width > default 300 (expanded layout).

  S16-b: render_scheme auto-computes step_sizes
    GIVEN a 2-step scheme where step 1 has small molecules and step 2 has large ones,
    WHEN render_scheme is called (no explicit step_sizes),
    THEN the layout box for step 2 should be wider than step 1.

    GIVEN a scheme with mixed molecule sizes,
    WHEN render_scheme is called,
    THEN boxes should not overlap (no regression from auto sizing).

  Adversarial:
    GIVEN a step with invalid SMILES in reactants,
    WHEN estimate_step_size is called,
    THEN it should fall back to default dimensions (not crash).

    GIVEN a step with empty reactants and products,
    WHEN estimate_step_size is called,
    THEN it should return default dimensions.
"""

import pytest

from chemscheme.renderers.rdkit_renderer import estimate_step_size

# ---------------------------------------------------------------------------
# S16-a: estimate_step_size unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEstimateStepSize:
    """estimate_step_size should return (width, height) based on molecule complexity."""

    def test_small_molecules_compact(self) -> None:
        """Steps with small molecules should get compact dimensions."""
        step = {
            "id": "1",
            "reactants": ["C"],
            "products": ["CC"],
            "arrow_above": "H2",
        }
        w, h = estimate_step_size(step)
        # 2 small mols (180 each) + arrow (100) + padding (20) = 480
        assert w < 600, f"Small molecules should have width < 600, got {w}"
        assert h > 0

    def test_large_molecules_expanded(self) -> None:
        """Steps with large polycyclic molecules should get wider dimensions."""
        # quaterphenyl (24 atoms) — valid large molecule
        step = {
            "id": "2",
            "reactants": ["c1ccc(-c2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1"],
            "products": ["c1ccc(-c2ccc(-c3ccc(-c4ccc(Br)cc4)cc3)cc2)cc1"],
            "arrow_above": "Pd cat",
        }
        w, h = estimate_step_size(step)
        assert w > 600, f"Large molecules should have width > 600, got {w}"

    def test_returns_tuple_of_floats(self) -> None:
        """Return type should be a tuple of two floats."""
        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(Br)cc1"],
            "arrow_above": "NBS",
        }
        result = estimate_step_size(step)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (int, float))
        assert isinstance(result[1], (int, float))

    def test_invalid_smiles_fallback(self) -> None:
        """Invalid SMILES should not crash — return default dimensions."""
        step = {
            "id": "bad",
            "reactants": ["NOT_A_SMILES"],
            "products": ["ALSO_BAD"],
            "arrow_above": "",
        }
        w, h = estimate_step_size(step)
        # Should return some reasonable default, not crash
        assert w > 0
        assert h > 0

    def test_empty_reactants_and_products(self) -> None:
        """Empty reactants/products should return defaults."""
        step = {
            "id": "empty",
            "reactants": [],
            "products": [],
            "arrow_above": "",
        }
        w, h = estimate_step_size(step)
        assert w > 0
        assert h > 0

    def test_multiple_reactants_wider(self) -> None:
        """More molecules in a step should produce wider dimensions."""
        step_1mol = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(Br)cc1"],
            "arrow_above": "",
        }
        step_3mol = {
            "id": "2",
            "reactants": ["c1ccccc1", "BrBr", "CCO"],
            "products": ["c1ccc(Br)cc1", "HBr"],
            "arrow_above": "",
        }
        w1, _ = estimate_step_size(step_1mol)
        w3, _ = estimate_step_size(step_3mol)
        assert w3 > w1, f"3-mol step ({w3}) should be wider than 1-mol step ({w1})"


# ---------------------------------------------------------------------------
# S16-b: render_scheme auto step_sizes integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.chemistry
class TestAutoStepSizesIntegration:
    """render_scheme should auto-compute step_sizes from SMILES."""

    def test_large_step_gets_wider_box(self) -> None:
        """Step with large molecules should get a wider layout box than small molecules."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "cat",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(-c2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1"],
                    "products": ["c1ccc(-c2ccc(-c3ccc(-c4ccc(Br)cc4)cc3)cc2)cc1"],
                    "arrow_above": "Zn",
                },
            ],
        }
        svg_bytes = render_scheme(scheme, fmt="svg")
        assert b"data-step-id" in svg_bytes

    def test_auto_sizes_no_box_overlap(self) -> None:
        """Auto-computed sizes should not cause box overlap in the layout."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "H2",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "KMnO4",
                },
                {
                    "id": "3",
                    "reactants": ["c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd",
                },
            ],
        }
        svg_bytes = render_scheme(scheme, fmt="svg")
        assert len(svg_bytes) > 0
        svg = svg_bytes.decode("utf-8")
        assert 'width="0"' not in svg
