"""Golden SVG regression test (AC-2.7).

BDD Feature: Visual regression for Suzuki coupling rendering.

The golden file is generated once, then future renders are compared against it.
If the golden file doesn't exist, the test generates it and marks as SKIP.
"""

from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


@pytest.mark.visual
@pytest.mark.chemistry
class TestGoldenSVGRegression:
    """AC-2.7: Golden SVG regression — Suzuki coupling."""

    def test_suzuki_coupling_golden(self) -> None:
        """Suzuki coupling SVG matches golden file."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
            "products": ["c1ccc(-c2ccccc2)cc1"],
            "arrow_above": "Pd(PPh3)4, K2CO3",
            "arrow_below": "DMF, 80 C, 12h",
        }
        result = render_step(step, fmt="svg")

        golden_path = GOLDEN_DIR / "suzuki_coupling.svg"
        if not golden_path.exists():
            # First run: generate golden file
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_bytes(result)
            pytest.skip("Golden file generated — re-run to compare")

        golden = golden_path.read_bytes()
        assert result == golden, (
            "SVG output differs from golden file. "
            "If this is intentional, delete the golden file and re-run."
        )
