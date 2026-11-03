"""Tests for smiles_utils.canonicalize() (AC-1.4, AC-1.5, AC-1.6).

BDD Feature: SMILES canonicalization
"""

import pytest


@pytest.mark.unit
@pytest.mark.chemistry
class TestCanonicalize:
    """AC-1.4: canonicalize(s) returns SmilesResult with RDKit canonical SMILES."""

    def test_benzene_canonical(self) -> None:
        """Canonical SMILES for benzene from Kekulé form."""
        from chemscheme.parsers.smiles_utils import canonicalize

        result = canonicalize("C1=CC=CC=C1")
        assert result.smiles == "c1ccccc1"
        assert result.confidence >= 0.99
        assert result.source == "rdkit"

    def test_benzene_from_aromatic(self) -> None:
        """Canonical SMILES for benzene from aromatic form."""
        from chemscheme.parsers.smiles_utils import canonicalize

        result = canonicalize("c1ccccc1")
        assert result.smiles == "c1ccccc1"
        assert result.confidence >= 0.99

    def test_ethanol(self) -> None:
        """Canonical SMILES for ethanol."""
        from chemscheme.parsers.smiles_utils import canonicalize

        result = canonicalize("OCC")
        assert result.smiles == "CCO"
        assert result.source == "rdkit"

    def test_source_is_rdkit(self) -> None:
        """Source field is always 'rdkit' for canonicalize."""
        from chemscheme.parsers.smiles_utils import canonicalize

        result = canonicalize("C")
        assert result.source == "rdkit"

    def test_no_warnings_for_valid(self) -> None:
        """No warnings for valid SMILES without issues."""
        from chemscheme.parsers.smiles_utils import canonicalize

        result = canonicalize("c1ccccc1")
        assert result.warnings == ()


@pytest.mark.unit
@pytest.mark.chemistry
class TestCanonicalizeInvalid:
    """AC-1.5: Invalid SMILES returns SmilesResult with confidence=0.0."""

    def test_invalid_smiles_string(self) -> None:
        """Non-SMILES string returns confidence=0.0."""
        from chemscheme.parsers.smiles_utils import canonicalize

        result = canonicalize("not-a-smiles")
        assert result.smiles == ""
        assert result.confidence == 0.0
        assert any("invalid SMILES" in w for w in result.warnings)

    def test_empty_string_input(self) -> None:
        """Empty string input returns confidence=0.0."""
        from chemscheme.parsers.smiles_utils import canonicalize

        result = canonicalize("")
        assert result.smiles == ""
        assert result.confidence == 0.0

    def test_invalid_preserves_source(self) -> None:
        """Even invalid input has source='rdkit'."""
        from chemscheme.parsers.smiles_utils import canonicalize

        result = canonicalize("XYZ123")
        assert result.source == "rdkit"


@pytest.mark.unit
@pytest.mark.chemistry
class TestCanonicalizeSameMolecule:
    """AC-1.6: Same molecule, different representations -> same canonical."""

    def test_benzene_representations(self) -> None:
        """Multiple benzene representations converge."""
        from chemscheme.parsers.smiles_utils import canonicalize

        r1 = canonicalize("c1ccccc1")
        r2 = canonicalize("C1=CC=CC=C1")
        r3 = canonicalize("c1ccccc1")  # same aromatic form
        assert r1.smiles == r2.smiles == r3.smiles

    def test_acetic_acid_representations(self) -> None:
        """Different acetic acid representations converge."""
        from chemscheme.parsers.smiles_utils import canonicalize

        r1 = canonicalize("CC(=O)O")
        r2 = canonicalize("CC(O)=O")
        assert r1.smiles == r2.smiles

    def test_biphenyl_representations(self) -> None:
        """Biphenyl from different atom orders converges."""
        from chemscheme.parsers.smiles_utils import canonicalize

        r1 = canonicalize("c1ccc(-c2ccccc2)cc1")
        r2 = canonicalize("c1ccc(cc1)-c1ccccc1")
        assert r1.smiles == r2.smiles


@pytest.mark.unit
@pytest.mark.chemistry
class TestCanonicalizeStereo:
    """BDD: Stereochemistry loss detection."""

    def test_stereo_preserved(self) -> None:
        """E/Z stereo is preserved in canonical SMILES."""
        from chemscheme.parsers.smiles_utils import canonicalize

        result = canonicalize(r"F/C=C\F")
        assert "/" in result.smiles or "\\" in result.smiles
