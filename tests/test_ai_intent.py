"""Tests for AI intent parsing: Claude API -> scheme.json (AC-5.5, AC-5.6, AC-5.7)."""

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# AC-5.5  Claude AI intent parsing -> scheme.json
# ---------------------------------------------------------------------------
class TestAiIntentParser:
    """AC-5.5: Claude API produces valid scheme.json from NL description."""

    @pytest.mark.integration
    def test_suzuki_coupling_produces_valid_scheme(self) -> None:
        """BDD: NL 'Suzuki coupling of bromobenzene and phenylboronic acid' -> valid scheme."""
        from chemscheme.parsers.ai_intent import parse_intent

        mock_scheme = {
            "steps": [
                {
                    "id": "step-1",
                    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd(PPh3)4, K2CO3",
                }
            ]
        }

        with patch(
            "chemscheme.parsers.ai_intent._call_claude_api",
            return_value=mock_scheme,
        ):
            result = parse_intent("Suzuki coupling of bromobenzene and phenylboronic acid")

        assert "steps" in result
        assert len(result["steps"]) >= 1
        step = result["steps"][0]
        assert "reactants" in step
        assert "products" in step
        assert "arrow_above" in step

    @pytest.mark.integration
    def test_intent_result_has_required_fields(self) -> None:
        """Scheme from AI has id, reactants, products, arrow_above per step."""
        from chemscheme.parsers.ai_intent import parse_intent

        mock_scheme = {
            "steps": [
                {
                    "id": "step-1",
                    "reactants": ["C=C"],
                    "products": ["CCO"],
                    "arrow_above": "H2O, H+",
                }
            ]
        }

        with patch(
            "chemscheme.parsers.ai_intent._call_claude_api",
            return_value=mock_scheme,
        ):
            result = parse_intent("hydration of ethylene")

        for step in result["steps"]:
            assert "id" in step
            assert "reactants" in step
            assert "products" in step
            assert "arrow_above" in step


# ---------------------------------------------------------------------------
# AC-5.6  Claude API failure -> graceful error
# ---------------------------------------------------------------------------
class TestAiErrorHandling:
    """AC-5.6: Claude API failure -> graceful error, no crash."""

    @pytest.mark.unit
    def test_api_timeout_raises_descriptive_error(self) -> None:
        """BDD: Claude API timeout -> raise with clear message."""
        from chemscheme.parsers.ai_intent import AiServiceError, parse_intent

        with (
            patch(
                "chemscheme.parsers.ai_intent._call_claude_api",
                side_effect=TimeoutError("request timed out"),
            ),
            pytest.raises(AiServiceError, match="AI service unavailable"),
        ):
            parse_intent("anything")

    @pytest.mark.unit
    def test_api_auth_error_raises_descriptive_error(self) -> None:
        """Claude API auth failure -> AiServiceError."""
        from chemscheme.parsers.ai_intent import AiServiceError, parse_intent

        with (
            patch(
                "chemscheme.parsers.ai_intent._call_claude_api",
                side_effect=PermissionError("invalid API key"),
            ),
            pytest.raises(AiServiceError),
        ):
            parse_intent("anything")

    @pytest.mark.unit
    def test_api_returns_invalid_json_raises_error(self) -> None:
        """Claude API returns non-scheme data -> AiServiceError."""
        from chemscheme.parsers.ai_intent import AiServiceError, parse_intent

        with (
            patch(
                "chemscheme.parsers.ai_intent._call_claude_api",
                return_value={"invalid": "no steps key"},
            ),
            pytest.raises(AiServiceError, match="invalid scheme"),
        ):
            parse_intent("anything")


# ---------------------------------------------------------------------------
# AC-5.7  Claude API mock fixture
# ---------------------------------------------------------------------------
class TestClaudeMockFixture:
    """AC-5.7: Claude API has mock fixture."""

    @pytest.mark.unit
    def test_claude_api_is_mockable(self) -> None:
        """_call_claude_api can be mocked for testing."""
        from chemscheme.parsers.ai_intent import parse_intent

        mock_scheme = {
            "steps": [
                {
                    "id": "step-1",
                    "reactants": ["C"],
                    "products": ["CO"],
                    "arrow_above": "O2",
                }
            ]
        }

        with patch(
            "chemscheme.parsers.ai_intent._call_claude_api",
            return_value=mock_scheme,
        ) as mock_api:
            parse_intent("oxidize methane")
            mock_api.assert_called_once()
