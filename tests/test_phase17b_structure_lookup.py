"""Phase 17b: Structure Lookup Database tests.

BDD-first RED tests — all must FAIL before implementation.

Design note: Uses RDKit Morgan fingerprints + Tanimoto natively (Python 3.14
compatible). FPSim2 HDF5 backend deferred as optional high-performance path
(FPSim2 requires Python 3.10-3.13).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemscheme.models import AgreementLevel
from chemscheme.recognition.lookup import (
    StructureLookup,
    build_lookup_db,
    load_user_library,
)
from chemscheme.recognition.protocols import MoleculeRecognizer

# ===========================================================================
# AC-17b.1: Database builder
# ===========================================================================


class TestBuildLookupDb:
    """build_lookup_db creates a JSON database from SMILES list."""

    def test_build_from_smiles_list(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.json"
        count = build_lookup_db(
            smiles_list=["CCO", "c1ccccc1", "CC(=O)O"],
            output_path=db_path,
        )
        assert count == 3
        assert db_path.exists()

    def test_build_skips_invalid_smiles(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.json"
        count = build_lookup_db(
            smiles_list=["CCO", "not_valid", "c1ccccc1"],
            output_path=db_path,
        )
        assert count == 2  # invalid skipped

    def test_build_with_names(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.json"
        count = build_lookup_db(
            smiles_list=["CCO", "c1ccccc1"],
            names=["ethanol", "benzene"],
            output_path=db_path,
        )
        assert count == 2

    def test_build_empty_list(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.json"
        count = build_lookup_db(smiles_list=[], output_path=db_path)
        assert count == 0
        assert db_path.exists()

    def test_build_all_invalid(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.json"
        count = build_lookup_db(
            smiles_list=["xxx", "yyy", "zzz"],
            output_path=db_path,
        )
        assert count == 0

    def test_build_deduplicates_canonical(self, tmp_path: Path) -> None:
        """Same molecule in different notations → single entry."""
        db_path = tmp_path / "test.json"
        count = build_lookup_db(
            smiles_list=["c1ccccc1", "C1=CC=CC=C1"],
            output_path=db_path,
        )
        assert count == 1  # deduplicated


# ===========================================================================
# AC-17b.2: Dataset loaders
# ===========================================================================


class TestDatasetLoaders:
    """User library loader returns (SMILES, name) tuples."""

    def test_load_user_library_json(self, tmp_path: Path) -> None:
        lib_file = tmp_path / "library.json"
        lib_file.write_text(
            json.dumps(
                [
                    {"smiles": "CCO", "name": "ethanol"},
                    {"smiles": "c1ccccc1", "name": "benzene"},
                ]
            )
        )
        entries = load_user_library(lib_file)
        assert len(entries) == 2
        assert entries[0][1] == "ethanol"

    def test_load_user_library_csv(self, tmp_path: Path) -> None:
        lib_file = tmp_path / "library.csv"
        lib_file.write_text("smiles,name\nCCO,ethanol\nc1ccccc1,benzene\n")
        entries = load_user_library(lib_file)
        assert len(entries) == 2

    def test_load_user_library_empty(self, tmp_path: Path) -> None:
        lib_file = tmp_path / "empty.json"
        lib_file.write_text("[]")
        entries = load_user_library(lib_file)
        assert len(entries) == 0

    def test_load_user_library_nonexistent(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_user_library(Path("/nonexistent/library.json"))


# ===========================================================================
# AC-17b.3: Similarity search interface
# ===========================================================================


class TestStructureLookup:
    """StructureLookup implements MoleculeRecognizer Protocol."""

    @pytest.fixture()
    def lookup(self, tmp_path: Path) -> StructureLookup:
        db_path = tmp_path / "test.json"
        build_lookup_db(
            smiles_list=["CCO", "c1ccccc1", "CC(=O)O"],
            names=["ethanol", "benzene", "acetic acid"],
            output_path=db_path,
        )
        return StructureLookup(db_path)

    def test_implements_protocol(self, lookup: StructureLookup) -> None:
        assert isinstance(lookup, MoleculeRecognizer)

    def test_exact_match_high_confidence(self, lookup: StructureLookup) -> None:
        result = lookup.recognize(b"CCO")  # SMILES as bytes for Protocol compat
        assert result.confidence >= 0.99
        assert result.agreement == AgreementLevel.SINGLE
        assert "CCO" in result.value  # C4 fix: always canonical SMILES

    def test_lookup_method_direct(self, lookup: StructureLookup) -> None:
        """Direct lookup method (non-Protocol) returns RecognitionResult."""
        result = lookup.lookup("CCO")
        assert result.confidence >= 0.99
        assert result.source == "structure_lookup"

    def test_close_analog_medium_confidence(self, lookup: StructureLookup) -> None:
        """Propanol is similar to ethanol but not identical."""
        result = lookup.lookup("CCCO", threshold=0.3)
        assert 0.0 < result.confidence < 1.0

    def test_unknown_structure_zero_confidence(self, lookup: StructureLookup) -> None:
        """Completely different structure returns no match."""
        # Use a large, complex molecule unlikely to match small DB
        result = lookup.lookup(
            "CC(C)(C)c1ccc(cc1)-c1ccc2c(c1)c1cc(-c3ccc(cc3)C(C)(C)C)ccc1c2=O",
            threshold=0.85,
        )
        assert result.confidence == 0.0

    def test_invalid_query_smiles(self, lookup: StructureLookup) -> None:
        result = lookup.lookup("not_a_smiles")
        assert result.confidence == 0.0
        assert result.needs_human_review is False

    def test_empty_database(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.json"
        build_lookup_db(smiles_list=[], output_path=db_path)
        lookup = StructureLookup(db_path)
        result = lookup.lookup("CCO")
        assert result.confidence == 0.0

    def test_threshold_boundary(self, lookup: StructureLookup) -> None:
        """Exact match always passes any threshold."""
        result = lookup.lookup("CCO", threshold=0.999)
        assert result.confidence >= 0.99

    def test_database_info(self, lookup: StructureLookup) -> None:
        info = lookup.info()
        assert info["entry_count"] == 3
        assert "db_path" in info
