"""Tests for AC-9.3: CDX import correctly assigns reactants vs products.

Phase 9 — MR7-03 (P1).
BDD-first: CDX binary files have no reaction semantics, so molecules
should be placed as reactants only, NOT duplicated into products.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from chemscheme.cli import main
from chemscheme.models import SmilesResult


@pytest.mark.unit
class TestCdxImportAssignment:
    """AC-9.3: CDX import places molecules in reactants only."""

    def test_cdx_molecules_not_duplicated_in_products(self, tmp_path: Path) -> None:
        """Molecules from CDX should appear in reactants, NOT products.

        Bug: previously all molecules were placed in both reactants AND products.
        """
        fake_cdx = tmp_path / "test.cdx"
        fake_cdx.write_bytes(b"dummy")
        out = tmp_path / "scheme.json"
        mock_results = [
            SmilesResult(smiles="CCO", confidence=1.0, source="cdx"),
            SmilesResult(smiles="CC=O", confidence=1.0, source="cdx"),
        ]

        with patch("chemscheme.parsers.structure_reader.read_cdx", return_value=mock_results):
            runner = CliRunner()
            result = runner.invoke(main, ["import", str(fake_cdx), "-o", str(out), "--force"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        scheme = json.loads(out.read_text(encoding="utf-8"))
        step = scheme["steps"][0]

        assert step["reactants"] == ["CCO", "CC=O"], "All molecules should be reactants"
        assert step["products"] != step["reactants"], "Products must NOT be a copy of reactants"

    def test_cdx_products_are_empty(self, tmp_path: Path) -> None:
        """Products should be empty list for CDX import (no reaction semantics)."""
        fake_cdx = tmp_path / "test.cdx"
        fake_cdx.write_bytes(b"dummy")
        out = tmp_path / "scheme.json"
        mock_results = [
            SmilesResult(smiles="c1ccccc1", confidence=1.0, source="cdx"),
        ]

        with patch("chemscheme.parsers.structure_reader.read_cdx", return_value=mock_results):
            runner = CliRunner()
            result = runner.invoke(main, ["import", str(fake_cdx), "-o", str(out), "--force"])

        assert result.exit_code == 0
        scheme = json.loads(out.read_text(encoding="utf-8"))
        step = scheme["steps"][0]
        assert step["products"] == [], "CDX has no reaction info → empty products"

    def test_cdx_import_warning_present(self, tmp_path: Path) -> None:
        """Import should include a warning about manual editing needed (--json output)."""
        fake_cdx = tmp_path / "test.cdx"
        fake_cdx.write_bytes(b"dummy")
        out = tmp_path / "scheme.json"
        mock_results = [
            SmilesResult(smiles="CCO", confidence=1.0, source="cdx"),
        ]

        with patch("chemscheme.parsers.structure_reader.read_cdx", return_value=mock_results):
            runner = CliRunner()
            result = runner.invoke(
                main, ["import", str(fake_cdx), "-o", str(out), "--force", "--json"]
            )

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "import_warnings" in output
        assert any("edit" in w.lower() for w in output["import_warnings"])

    def test_adversarial_single_molecule_cdx(self, tmp_path: Path) -> None:
        """Adversarial: single molecule CDX → reactants=[mol], products=[]."""
        fake_cdx = tmp_path / "test.cdx"
        fake_cdx.write_bytes(b"dummy")
        out = tmp_path / "scheme.json"
        mock_results = [
            SmilesResult(smiles="C", confidence=1.0, source="cdx"),
        ]

        with patch("chemscheme.parsers.structure_reader.read_cdx", return_value=mock_results):
            runner = CliRunner()
            result = runner.invoke(main, ["import", str(fake_cdx), "-o", str(out), "--force"])

        assert result.exit_code == 0
        scheme = json.loads(out.read_text(encoding="utf-8"))
        step = scheme["steps"][0]
        assert step["reactants"] == ["C"]
        assert step["products"] == []
