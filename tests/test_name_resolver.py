"""Tests for name resolution: local alias, OPSIN, PubChem fallback (AC-5.1 ~ AC-5.4, AC-5.7)."""

from unittest.mock import MagicMock, patch

import pytest

from chemscheme.models import SmilesResult


# ---------------------------------------------------------------------------
# AC-5.4  Local alias lookup (no network)
# ---------------------------------------------------------------------------
class TestLocalAliasLookup:
    """AC-5.4: reagent_aliases.json lookup returns SmilesResult immediately."""

    @pytest.mark.unit
    def test_known_alias_nbs(self) -> None:
        """NBS resolves to N-bromosuccinimide SMILES without API call."""
        from chemscheme.parsers.name_resolver import resolve_local_alias

        result = resolve_local_alias("NBS")
        assert isinstance(result, SmilesResult)
        assert result.smiles != ""
        assert result.source == "local_alias"
        assert result.confidence >= 0.9

    @pytest.mark.unit
    def test_known_alias_dmf(self) -> None:
        """DMF resolves correctly."""
        from chemscheme.parsers.name_resolver import resolve_local_alias

        result = resolve_local_alias("DMF")
        assert result.smiles != ""
        assert result.source == "local_alias"

    @pytest.mark.unit
    def test_unknown_alias_returns_none(self) -> None:
        """Unknown abbreviation returns None (not SmilesResult)."""
        from chemscheme.parsers.name_resolver import resolve_local_alias

        result = resolve_local_alias("UNKNOWN_COMPOUND_XYZ")
        assert result is None

    @pytest.mark.unit
    def test_case_sensitive(self) -> None:
        """Alias lookup is case-sensitive (NBS != nbs)."""
        from chemscheme.parsers.name_resolver import resolve_local_alias

        result_upper = resolve_local_alias("NBS")
        result_lower = resolve_local_alias("nbs")
        assert result_upper is not None
        assert result_lower is None


# ---------------------------------------------------------------------------
# AC-5.2  OPSIN name resolution
# ---------------------------------------------------------------------------
class TestOpsinResolver:
    """AC-5.2: OPSIN resolves IUPAC names to SmilesResult."""

    @pytest.mark.integration
    def test_opsin_iupac_dinitrophenol(self) -> None:
        """OPSIN resolves '2,4-dinitrophenol' to correct SMILES."""
        from chemscheme.parsers.name_resolver import resolve_via_opsin

        # Mock the HTTP call to OPSIN
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "O=[N+]([O-])c1ccc(O)c([N+](=O)[O-])c1"

        with patch("chemscheme.parsers.name_resolver.requests.get", return_value=mock_response):
            result = resolve_via_opsin("2,4-dinitrophenol")

        assert isinstance(result, SmilesResult)
        assert result.smiles != ""
        assert result.source == "opsin"
        assert result.confidence >= 0.8

    @pytest.mark.integration
    def test_opsin_trimethylphenol(self) -> None:
        """BDD: OPSIN resolves '2,4,6-trimethylphenol'."""
        from chemscheme.parsers.name_resolver import resolve_via_opsin

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Cc1cc(C)c(O)c(C)c1"

        with patch("chemscheme.parsers.name_resolver.requests.get", return_value=mock_response):
            result = resolve_via_opsin("2,4,6-trimethylphenol")

        assert result.smiles != ""
        assert result.source == "opsin"

    @pytest.mark.integration
    def test_opsin_failure_returns_none(self) -> None:
        """OPSIN returns None on HTTP error."""
        from chemscheme.parsers.name_resolver import resolve_via_opsin

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("chemscheme.parsers.name_resolver.requests.get", return_value=mock_response):
            result = resolve_via_opsin("totally_invalid_name")

        assert result is None

    @pytest.mark.integration
    def test_opsin_network_error_returns_none(self) -> None:
        """OPSIN returns None on network exception."""
        from chemscheme.parsers.name_resolver import resolve_via_opsin

        with patch(
            "chemscheme.parsers.name_resolver.requests.get",
            side_effect=ConnectionError("no network"),
        ):
            result = resolve_via_opsin("benzene")

        assert result is None


# ---------------------------------------------------------------------------
# AC-5.3  PubChem fallback
# ---------------------------------------------------------------------------
class TestPubChemFallback:
    """AC-5.3: PubChem fallback when OPSIN fails."""

    @pytest.mark.integration
    def test_pubchem_resolves_aspirin(self) -> None:
        """PubChem resolves 'aspirin' to correct SMILES."""
        from chemscheme.parsers.name_resolver import resolve_via_pubchem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "PropertyTable": {"Properties": [{"CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}]}
        }

        with patch("chemscheme.parsers.name_resolver.requests.get", return_value=mock_response):
            result = resolve_via_pubchem("aspirin")

        assert isinstance(result, SmilesResult)
        assert result.smiles != ""
        assert result.source == "pubchem"
        assert result.confidence >= 0.7

    @pytest.mark.integration
    def test_pubchem_failure_returns_none(self) -> None:
        """PubChem returns None on failure."""
        from chemscheme.parsers.name_resolver import resolve_via_pubchem

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("chemscheme.parsers.name_resolver.requests.get", return_value=mock_response):
            result = resolve_via_pubchem("nonexistent_compound")

        assert result is None

    @pytest.mark.integration
    def test_pubchem_network_error_returns_none(self) -> None:
        """PubChem returns None on network exception."""
        from chemscheme.parsers.name_resolver import resolve_via_pubchem

        with patch(
            "chemscheme.parsers.name_resolver.requests.get",
            side_effect=ConnectionError("no network"),
        ):
            result = resolve_via_pubchem("aspirin")

        assert result is None

    @pytest.mark.integration
    def test_fallback_chain_opsin_fails_pubchem_succeeds(self) -> None:
        """BDD: OPSIN fails for 'aspirin', PubChem resolves it."""
        from chemscheme.parsers.name_resolver import resolve_name

        with (
            patch("chemscheme.parsers.name_resolver.resolve_via_opsin", return_value=None),
            patch(
                "chemscheme.parsers.name_resolver.resolve_via_pubchem",
                return_value=SmilesResult(
                    smiles="CC(=O)Oc1ccccc1C(=O)O",
                    confidence=0.85,
                    source="pubchem",
                    warnings=[],
                ),
            ),
        ):
            result = resolve_name("aspirin")

        assert result.source == "pubchem"
        assert result.smiles != ""


# ---------------------------------------------------------------------------
# AC-5.1  resolve() orchestrator — full chain
# ---------------------------------------------------------------------------
class TestResolveOrchestrator:
    """AC-5.1: resolve_name() tries local alias -> OPSIN -> PubChem."""

    @pytest.mark.unit
    def test_resolve_local_alias_first(self) -> None:
        """Local alias is checked before any API call."""
        from chemscheme.parsers.name_resolver import resolve_name

        with (
            patch(
                "chemscheme.parsers.name_resolver.resolve_local_alias",
                return_value=SmilesResult(
                    smiles="O=C1CCC(=O)N1Br",
                    confidence=0.95,
                    source="local_alias",
                    warnings=[],
                ),
            ) as mock_alias,
            patch("chemscheme.parsers.name_resolver.resolve_via_opsin") as mock_opsin,
            patch("chemscheme.parsers.name_resolver.resolve_via_pubchem") as mock_pubchem,
        ):
            result = resolve_name("NBS")

        assert result.source == "local_alias"
        mock_alias.assert_called_once()
        mock_opsin.assert_not_called()
        mock_pubchem.assert_not_called()

    @pytest.mark.unit
    def test_resolve_all_fail_returns_error(self) -> None:
        """All resolvers fail -> SmilesResult with confidence=0."""
        from chemscheme.parsers.name_resolver import resolve_name

        with (
            patch("chemscheme.parsers.name_resolver.resolve_local_alias", return_value=None),
            patch("chemscheme.parsers.name_resolver.resolve_via_opsin", return_value=None),
            patch("chemscheme.parsers.name_resolver.resolve_via_pubchem", return_value=None),
        ):
            result = resolve_name("totally_unknown_compound")

        assert result.confidence == 0.0
        assert result.smiles == ""
        assert len(result.warnings) > 0

    @pytest.mark.unit
    def test_resolve_benzene_via_opsin(self) -> None:
        """AC-5.1: 'benzene' resolves to correct canonical SMILES."""
        from chemscheme.parsers.name_resolver import resolve_name

        with (
            patch("chemscheme.parsers.name_resolver.resolve_local_alias", return_value=None),
            patch(
                "chemscheme.parsers.name_resolver.resolve_via_opsin",
                return_value=SmilesResult(
                    smiles="c1ccccc1",
                    confidence=0.95,
                    source="opsin",
                    warnings=[],
                ),
            ),
        ):
            result = resolve_name("benzene")

        assert result.smiles == "c1ccccc1"
        assert result.source == "opsin"


# ---------------------------------------------------------------------------
# AC-5.7  Mock fixtures exist for all external APIs
# ---------------------------------------------------------------------------
class TestMockFixtures:
    """AC-5.7: All external APIs have mock fixtures."""

    @pytest.mark.unit
    def test_opsin_is_mockable(self) -> None:
        """OPSIN HTTP call can be mocked."""
        from chemscheme.parsers.name_resolver import resolve_via_opsin

        with patch("chemscheme.parsers.name_resolver.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = "c1ccccc1"
            resolve_via_opsin("benzene")
            mock_get.assert_called_once()

    @pytest.mark.unit
    def test_pubchem_is_mockable(self) -> None:
        """PubChem HTTP call can be mocked."""
        from chemscheme.parsers.name_resolver import resolve_via_pubchem

        with patch("chemscheme.parsers.name_resolver.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "PropertyTable": {"Properties": [{"CanonicalSMILES": "c1ccccc1"}]}
            }
            resolve_via_pubchem("benzene")
            mock_get.assert_called_once()
