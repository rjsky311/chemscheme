"""Tests for structure_reader module — MOL/SDF/CDX → SmilesResult.

BDD Scenarios:
  GIVEN a valid MOL file, WHEN read, THEN returns SmilesResult with valid SMILES
  GIVEN a valid SDF file, WHEN read, THEN returns list of SmilesResult
  GIVEN a valid CDX binary file, WHEN read, THEN returns list of SmilesResult with correct SMILES
  GIVEN an invalid MOL file, WHEN read, THEN returns SmilesResult with confidence=0.0
  GIVEN an empty SDF file, WHEN read, THEN returns empty list
  GIVEN a non-existent file, WHEN read, THEN raises FileNotFoundError
  GIVEN an unsupported extension, WHEN read, THEN raises ValueError
"""

from pathlib import Path

import pytest
from rdkit import Chem

from chemscheme.models import SmilesResult

FIXTURES = Path(__file__).parent / "fixtures"


# --- MOL file tests ---


class TestReadMolFile:
    def test_valid_mol_returns_smiles_result(self) -> None:
        from chemscheme.parsers.structure_reader import read_mol

        result = read_mol(FIXTURES / "benzene.mol")
        assert isinstance(result, SmilesResult)
        assert result.smiles != ""
        assert result.confidence == 1.0
        assert result.source == "mol_file"

    def test_valid_mol_smiles_is_canonical(self) -> None:
        from chemscheme.parsers.structure_reader import read_mol

        result = read_mol(FIXTURES / "benzene.mol")
        # Should be canonical benzene SMILES
        mol = Chem.MolFromSmiles(result.smiles)
        assert mol is not None
        assert mol.GetNumAtoms() == 6  # benzene has 6 heavy atoms

    def test_invalid_mol_returns_zero_confidence(self) -> None:
        from chemscheme.parsers.structure_reader import read_mol

        result = read_mol(FIXTURES / "invalid.mol")
        assert result.smiles == ""
        assert result.confidence == 0.0
        assert len(result.warnings) > 0

    def test_nonexistent_mol_raises(self) -> None:
        from chemscheme.parsers.structure_reader import read_mol

        with pytest.raises(FileNotFoundError):
            read_mol(FIXTURES / "does_not_exist.mol")


# --- SDF file tests ---


class TestReadSdfFile:
    def test_valid_sdf_returns_list(self) -> None:
        from chemscheme.parsers.structure_reader import read_sdf

        results = read_sdf(FIXTURES / "multi.sdf")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, SmilesResult) for r in results)

    def test_valid_sdf_smiles_are_valid(self) -> None:
        from chemscheme.parsers.structure_reader import read_sdf

        results = read_sdf(FIXTURES / "multi.sdf")
        for r in results:
            assert r.smiles != ""
            mol = Chem.MolFromSmiles(r.smiles)
            assert mol is not None

    def test_empty_sdf_returns_empty_list(self) -> None:
        from chemscheme.parsers.structure_reader import read_sdf

        results = read_sdf(FIXTURES / "empty.sdf")
        assert results == []

    def test_sdf_source_is_sdf_file(self) -> None:
        from chemscheme.parsers.structure_reader import read_sdf

        results = read_sdf(FIXTURES / "multi.sdf")
        for r in results:
            assert r.source == "sdf_file"


# --- CDX binary file tests ---


class TestReadCdxFile:
    def test_valid_cdx_returns_smiles_results(self) -> None:
        from chemscheme.parsers.structure_reader import read_cdx

        results = read_cdx(FIXTURES / "tetrabromo_spirobifluorene.cdx")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, SmilesResult) for r in results)

    def test_cdx_spiro_formula_correct(self) -> None:
        """2,2',7,7'-Tetrabromo-9,9'-spirobifluorene from CDX should have formula C25H12Br4."""
        from rdkit.Chem import rdMolDescriptors

        from chemscheme.parsers.structure_reader import read_cdx

        results = read_cdx(FIXTURES / "tetrabromo_spirobifluorene.cdx")
        # Find the main molecule (largest)
        main = max(results, key=lambda r: len(r.smiles))
        mol = Chem.MolFromSmiles(main.smiles)
        assert mol is not None
        formula = rdMolDescriptors.CalcMolFormula(mol)
        assert formula == "C25H12Br4"

    def test_cdx_source_is_cdx_file(self) -> None:
        from chemscheme.parsers.structure_reader import read_cdx

        results = read_cdx(FIXTURES / "tetrabromo_spirobifluorene.cdx")
        for r in results:
            assert r.source == "cdx_file"

    def test_invalid_cdx_raises(self) -> None:
        from chemscheme.parsers.structure_reader import read_cdx

        with pytest.raises(ValueError, match="[Ii]nvalid CDX"):
            read_cdx(FIXTURES / "invalid.mol")  # not a CDX file

    def test_nonexistent_cdx_raises(self) -> None:
        from chemscheme.parsers.structure_reader import read_cdx

        with pytest.raises(FileNotFoundError):
            read_cdx(FIXTURES / "nonexistent.cdx")


# --- Unified read_structure dispatcher ---


class TestReadStructure:
    def test_dispatches_mol(self) -> None:
        from chemscheme.parsers.structure_reader import read_structure

        results = read_structure(FIXTURES / "benzene.mol")
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].source == "mol_file"

    def test_dispatches_sdf(self) -> None:
        from chemscheme.parsers.structure_reader import read_structure

        results = read_structure(FIXTURES / "multi.sdf")
        assert len(results) == 2

    def test_dispatches_cdx(self) -> None:
        from chemscheme.parsers.structure_reader import read_structure

        results = read_structure(FIXTURES / "tetrabromo_spirobifluorene.cdx")
        assert len(results) >= 1

    def test_unsupported_extension_raises(self) -> None:
        from chemscheme.parsers.structure_reader import read_structure

        with pytest.raises(ValueError, match="[Uu]nsupported"):
            read_structure(FIXTURES / "sample_scheme.json")

    def test_nonexistent_raises(self) -> None:
        from chemscheme.parsers.structure_reader import read_structure

        with pytest.raises(FileNotFoundError):
            read_structure(Path("/nonexistent/file.mol"))


class TestCDXSizeGuard:
    """MR15-G1: CDX file size guard prevents OOM on oversized files."""

    # RED-EXEMPT: review finding regression

    def test_oversized_cdx_raises(self, tmp_path: Path) -> None:
        """CDX file exceeding 50 MB limit raises ValueError."""
        from chemscheme.parsers.structure_reader import read_cdx

        # Create a file that reports >50MB via stat
        large_file = tmp_path / "huge.cdx"
        large_file.write_bytes(b"VjCD0100" + b"\x00" * 100)

        # Monkey-patch stat to report oversized
        import unittest.mock

        fake_stat = large_file.stat()
        with (
            unittest.mock.patch.object(
                type(large_file),
                "stat",
                return_value=type(fake_stat)(
                    (
                        fake_stat.st_mode,
                        fake_stat.st_ino,
                        fake_stat.st_dev,
                        fake_stat.st_nlink,
                        fake_stat.st_uid,
                        fake_stat.st_gid,
                        51 * 1024 * 1024,  # st_size > 50 MB
                        fake_stat.st_atime,
                        fake_stat.st_mtime,
                        fake_stat.st_ctime,
                    )
                ),
            ),
            pytest.raises(ValueError, match="too large"),
        ):
            read_cdx(large_file)
