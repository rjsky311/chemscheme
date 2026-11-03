"""Tests for SmilesResult dataclass (AC-1.3, AC-1.5).

BDD Feature: SmilesResult unified return type
"""

import pytest


@pytest.mark.unit
class TestSmilesResultDataclass:
    """AC-1.3: SmilesResult dataclass with smiles, confidence, source, warnings."""

    def test_create_valid_result(self) -> None:
        """SmilesResult can be created with all required fields."""
        from chemscheme.models import SmilesResult

        result = SmilesResult(
            smiles="c1ccccc1",
            confidence=0.99,
            source="rdkit",
            warnings=[],
        )
        assert result.smiles == "c1ccccc1"
        assert result.confidence == 0.99
        assert result.source == "rdkit"
        assert result.warnings == ()

    def test_frozen_immutable(self) -> None:
        """SmilesResult is frozen (immutable)."""
        from chemscheme.models import SmilesResult

        result = SmilesResult(
            smiles="c1ccccc1",
            confidence=0.99,
            source="rdkit",
            warnings=[],
        )
        with pytest.raises(AttributeError):
            result.smiles = "CC"  # type: ignore[misc]

    def test_is_high_confidence_true(self) -> None:
        """is_high_confidence returns True when confidence >= 0.9."""
        from chemscheme.models import SmilesResult

        result = SmilesResult(
            smiles="c1ccccc1",
            confidence=0.95,
            source="rdkit",
            warnings=[],
        )
        assert result.is_high_confidence is True

    def test_is_high_confidence_false(self) -> None:
        """is_high_confidence returns False when confidence < 0.9."""
        from chemscheme.models import SmilesResult

        result = SmilesResult(
            smiles="c1ccccc1",
            confidence=0.5,
            source="pubchem",
            warnings=["low confidence"],
        )
        assert result.is_high_confidence is False

    def test_is_high_confidence_boundary(self) -> None:
        """is_high_confidence returns True at exactly 0.9."""
        from chemscheme.models import SmilesResult

        result = SmilesResult(
            smiles="c1ccccc1",
            confidence=0.9,
            source="rdkit",
            warnings=[],
        )
        assert result.is_high_confidence is True


@pytest.mark.unit
class TestSmilesResultInvalidCase:
    """AC-1.5: Invalid SMILES returns SmilesResult with confidence=0.0."""

    def test_invalid_smiles_empty_string(self) -> None:
        """Invalid case uses smiles='' (not None)."""
        from chemscheme.models import SmilesResult

        result = SmilesResult(
            smiles="",
            confidence=0.0,
            source="rdkit",
            warnings=["invalid SMILES: parse error"],
        )
        assert result.smiles == ""
        assert result.confidence == 0.0

    def test_invalid_smiles_has_warning(self) -> None:
        """Invalid case must have a warning explaining the failure."""
        from chemscheme.models import SmilesResult

        result = SmilesResult(
            smiles="",
            confidence=0.0,
            source="rdkit",
            warnings=["invalid SMILES: not_a_smiles"],
        )
        assert any("invalid SMILES" in w for w in result.warnings)

    def test_invalid_check_by_empty_smiles(self) -> None:
        """Callers check failure via result.smiles == ''."""
        from chemscheme.models import SmilesResult

        result = SmilesResult(
            smiles="",
            confidence=0.0,
            source="rdkit",
            warnings=["invalid SMILES: XYZ"],
        )
        assert result.smiles == ""
        assert result.is_high_confidence is False


@pytest.mark.unit
class TestSmilesResultSourceValues:
    """Valid source field values."""

    @pytest.mark.parametrize(
        "source",
        ["rdkit", "opsin", "pubchem", "local_alias", "decimer", "user_input"],
    )
    def test_valid_source_values(self, source: str) -> None:
        """All defined source values can be used."""
        from chemscheme.models import SmilesResult

        result = SmilesResult(
            smiles="c1ccccc1",
            confidence=0.99,
            source=source,
            warnings=[],
        )
        assert result.source == source
