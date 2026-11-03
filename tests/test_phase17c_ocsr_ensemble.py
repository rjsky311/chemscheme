"""Phase 17c: OCSR Ensemble tests.

BDD-first RED tests. MolScribe + DECIMER adapters via subprocess,
Mathpix tiebreaker via existing client, ensemble orchestrator with
FPSim2 (17b) short-circuit.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chemscheme.models import AgreementLevel
from chemscheme.recognition.lookup import StructureLookup, build_lookup_db
from chemscheme.recognition.mocks import MockMoleculeRecognizer
from chemscheme.recognition.ocsr import (
    DecimerRecognizer,
    MathpixRecognizer,
    MolScribeRecognizer,
    recognize_molecule,
)
from chemscheme.recognition.protocols import MoleculeRecognizer

# ===========================================================================
# AC-17c.1: MolScribe adapter
# ===========================================================================


class TestMolScribeRecognizer:
    """MolScribe subprocess adapter."""

    def test_implements_protocol(self) -> None:
        r = MolScribeRecognizer()
        assert isinstance(r, MoleculeRecognizer)

    def test_recognize_with_mock_subprocess(self) -> None:
        r = MolScribeRecognizer()
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"CCO\n",
                stderr="",
                returncode=0,
                elapsed=1.0,
            )
            result = r.recognize(b"fake_image")
            assert result.value == "CCO"
            assert result.confidence > 0
            assert result.source == "molscribe"

    def test_recognize_tool_not_installed(self) -> None:
        r = MolScribeRecognizer()
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"",
                stderr="Command not found: molscribe",
                returncode=-1,
                elapsed=0.0,
            )
            result = r.recognize(b"fake_image")
            assert result.confidence == 0.0

    def test_recognize_invalid_output(self) -> None:
        r = MolScribeRecognizer()
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"not_a_smiles\n",
                stderr="",
                returncode=0,
                elapsed=1.0,
            )
            result = r.recognize(b"fake_image")
            # Should return result with low confidence for invalid SMILES
            assert result.confidence < 0.5


# ===========================================================================
# AC-17c.2: DECIMER adapter
# ===========================================================================


class TestDecimerRecognizer:
    """DECIMER subprocess adapter."""

    def test_implements_protocol(self) -> None:
        r = DecimerRecognizer()
        assert isinstance(r, MoleculeRecognizer)

    def test_recognize_with_mock_subprocess(self) -> None:
        r = DecimerRecognizer()
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"c1ccccc1\n",
                stderr="",
                returncode=0,
                elapsed=2.0,
            )
            result = r.recognize(b"fake_image")
            assert result.value == "c1ccccc1"
            assert result.source == "decimer"

    def test_recognize_tool_failure(self) -> None:
        r = DecimerRecognizer()
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"",
                stderr="CUDA error",
                returncode=1,
                elapsed=0.5,
            )
            result = r.recognize(b"fake_image")
            assert result.confidence == 0.0


# ===========================================================================
# AC-17c.3: Mathpix tiebreaker adapter
# ===========================================================================


class TestMathpixRecognizer:
    """Mathpix adapter wrapping existing MathpixClient."""

    def test_implements_protocol(self) -> None:
        r = MathpixRecognizer()
        assert isinstance(r, MoleculeRecognizer)

    def test_recognize_with_mock_client(self) -> None:
        r = MathpixRecognizer()
        mock_result = MagicMock()
        mock_result.smiles_list = ["CCO"]
        mock_result.confidence = 0.95
        mock_result.error = None

        with patch.object(r, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.extract_text_from_base64.return_value = mock_result
            mock_get.return_value = mock_client
            result = r.recognize(b"fake_image")
            assert result.value == "CCO"
            assert result.confidence > 0
            assert result.source == "mathpix"

    def test_recognize_no_api_key(self) -> None:
        """No Mathpix credentials → graceful zero confidence."""
        r = MathpixRecognizer()
        with patch.object(r, "_get_client", side_effect=Exception("No API key")):
            result = r.recognize(b"fake_image")
            assert result.confidence == 0.0

    def test_recognize_api_returns_no_smiles(self) -> None:
        r = MathpixRecognizer()
        mock_result = MagicMock()
        mock_result.smiles_list = []
        mock_result.text = ""
        mock_result.confidence = 0.5
        mock_result.error = None

        with patch.object(r, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.extract_text_from_base64.return_value = mock_result
            mock_get.return_value = mock_client
            result = r.recognize(b"fake_image")
            assert result.confidence == 0.0


# ===========================================================================
# AC-17c.4: OCSR ensemble orchestrator
# ===========================================================================


class TestRecognizeMolecule:
    """Ensemble orchestrator with FPSim2 short-circuit."""

    @pytest.fixture()
    def lookup_db(self, tmp_path: Path) -> StructureLookup:
        db_path = tmp_path / "test.json"
        build_lookup_db(
            smiles_list=["CCO", "c1ccccc1"],
            names=["ethanol", "benzene"],
            output_path=db_path,
        )
        return StructureLookup(db_path)

    def test_known_structure_short_circuits(self, lookup_db: StructureLookup) -> None:
        """FPSim2 lookup hit → skip ML tools."""
        mock_mol = MockMoleculeRecognizer(value="CCO", confidence=0.9)
        mock_dec = MockMoleculeRecognizer(value="CCO", confidence=0.9)

        result = recognize_molecule(
            image=b"img",
            smiles_hint="CCO",
            lookup=lookup_db,
            recognizers=[mock_mol, mock_dec],
        )
        assert result.confidence >= 0.99
        assert result.source == "structure_lookup"

    def test_two_recognizers_agree(self) -> None:
        """Both ML tools agree → HIGH confidence."""
        mock_a = MockMoleculeRecognizer(value="CCCO", confidence=0.85)
        mock_b = MockMoleculeRecognizer(value="CCCO", confidence=0.80)

        result = recognize_molecule(
            image=b"img",
            smiles_hint=None,
            lookup=None,
            recognizers=[mock_a, mock_b],
        )
        assert result.agreement == AgreementLevel.HIGH
        assert result.confidence >= 0.80

    def test_two_recognizers_disagree_tiebreaker(self) -> None:
        """ML tools disagree → tiebreaker invoked."""
        mock_a = MockMoleculeRecognizer(value="CCO", confidence=0.8)
        mock_b = MockMoleculeRecognizer(value="CCCO", confidence=0.7)
        mock_tie = MockMoleculeRecognizer(value="CCO", confidence=0.9)

        result = recognize_molecule(
            image=b"img",
            smiles_hint=None,
            lookup=None,
            recognizers=[mock_a, mock_b],
            tiebreaker=mock_tie,
        )
        # Tiebreaker agrees with mock_a → MEDIUM
        assert result.agreement == AgreementLevel.MEDIUM

    def test_all_disagree_needs_review(self) -> None:
        """All tools disagree → needs human review."""
        mock_a = MockMoleculeRecognizer(value="CCO", confidence=0.8)
        mock_b = MockMoleculeRecognizer(value="CCCO", confidence=0.7)
        mock_tie = MockMoleculeRecognizer(value="CC(O)C", confidence=0.6)

        result = recognize_molecule(
            image=b"img",
            smiles_hint=None,
            lookup=None,
            recognizers=[mock_a, mock_b],
            tiebreaker=mock_tie,
        )
        assert result.agreement == AgreementLevel.LOW
        assert result.needs_human_review is True

    def test_no_recognizers_raises(self) -> None:
        with pytest.raises(ValueError):
            recognize_molecule(
                image=b"img",
                smiles_hint=None,
                lookup=None,
                recognizers=[],
            )

    def test_single_recognizer(self) -> None:
        mock = MockMoleculeRecognizer(value="CCO", confidence=0.85)
        result = recognize_molecule(
            image=b"img",
            smiles_hint=None,
            lookup=None,
            recognizers=[mock],
        )
        assert result.agreement == AgreementLevel.SINGLE
        assert result.confidence == 0.85

    def test_lookup_miss_falls_through(self, lookup_db: StructureLookup) -> None:
        """Unknown structure → falls through to ML recognizers."""
        mock = MockMoleculeRecognizer(value="CCCCCCCCCCO", confidence=0.7)

        result = recognize_molecule(
            image=b"img",
            smiles_hint="CCCCCCCCCCO",  # not in lookup DB
            lookup=lookup_db,
            recognizers=[mock],
        )
        # Should come from recognizer, not lookup
        assert result.source != "structure_lookup"

    def test_corrupted_image_low_confidence(self) -> None:
        """Recognizers returning empty/invalid → low confidence."""
        mock_a = MockMoleculeRecognizer(value="", confidence=0.1)
        mock_b = MockMoleculeRecognizer(value="", confidence=0.05)

        result = recognize_molecule(
            image=b"img",
            smiles_hint=None,
            lookup=None,
            recognizers=[mock_a, mock_b],
        )
        assert result.confidence < 0.5
