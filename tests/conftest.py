"""Shared pytest fixtures for ChemScheme tests."""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_scheme(fixtures_dir: Path) -> dict:
    """Load sample_scheme.json as a dict."""
    return json.loads((fixtures_dir / "sample_scheme.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_scheme_convergent(fixtures_dir: Path) -> dict:
    """Load sample_scheme_convergent.json (with consumes_step_ids)."""
    return json.loads((fixtures_dir / "sample_scheme_convergent.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_scheme_invalid(fixtures_dir: Path) -> dict:
    """Load sample_scheme_invalid.json as a dict."""
    return json.loads((fixtures_dir / "sample_scheme_invalid.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_scheme_3step(fixtures_dir: Path) -> dict:
    """Load 3-step linear scheme."""
    return json.loads((fixtures_dir / "sample_scheme_3step.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_scheme_5step_nfa(fixtures_dir: Path) -> dict:
    """Load 5-step linear NFA scheme."""
    return json.loads((fixtures_dir / "sample_scheme_5step_nfa.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_scheme_8step_convergent(fixtures_dir: Path) -> dict:
    """Load 8-step convergent NFA scheme."""
    return json.loads(
        (fixtures_dir / "sample_scheme_8step_convergent.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def sample_scheme_unicode(fixtures_dir: Path) -> dict:
    """Load sample_scheme_unicode.json (with Unicode reagents)."""
    return json.loads((fixtures_dir / "sample_scheme_unicode.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_scheme_noncontiguous_branches(fixtures_dir: Path) -> dict:
    """Load noncontiguous branch scheme (branch A interleaved with branch B)."""
    return json.loads(
        (fixtures_dir / "sample_scheme_noncontiguous_branches.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def benzene_smiles() -> str:
    """Canonical SMILES for benzene."""
    return "c1ccccc1"


@pytest.fixture
def biphenyl_smiles() -> str:
    """Canonical SMILES for biphenyl (Suzuki coupling product)."""
    return "c1ccc(-c2ccccc2)cc1"
