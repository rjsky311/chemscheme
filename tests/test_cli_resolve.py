"""Tests for CLI resolve + ai commands (AC-5.1, AC-5.5)."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from chemscheme.cli import main
from chemscheme.models import SmilesResult


# ---------------------------------------------------------------------------
# AC-5.1  chemscheme resolve "<name>"
# ---------------------------------------------------------------------------
class TestCliResolve:
    """AC-5.1: `chemscheme resolve 'benzene'` -> correct canonical SMILES."""

    @pytest.mark.integration
    def test_resolve_benzene(self) -> None:
        """CLI resolve 'benzene' prints SMILES."""
        runner = CliRunner()
        with patch(
            "chemscheme.parsers.name_resolver.resolve_name",
            return_value=SmilesResult(
                smiles="c1ccccc1",
                confidence=0.95,
                source="opsin",
                warnings=[],
            ),
        ):
            result = runner.invoke(main, ["resolve", "benzene"])

        assert result.exit_code == 0
        assert "c1ccccc1" in result.output

    @pytest.mark.integration
    def test_resolve_json_output(self) -> None:
        """CLI resolve --json produces structured JSON."""
        import json

        runner = CliRunner()
        with patch(
            "chemscheme.parsers.name_resolver.resolve_name",
            return_value=SmilesResult(
                smiles="c1ccccc1",
                confidence=0.95,
                source="opsin",
                warnings=[],
            ),
        ):
            result = runner.invoke(main, ["resolve", "--json", "benzene"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["smiles"] == "c1ccccc1"
        assert data["source"] == "opsin"

    @pytest.mark.integration
    def test_resolve_failure_exit_code(self) -> None:
        """CLI resolve with unknown compound -> exit code 1."""
        runner = CliRunner()
        with patch(
            "chemscheme.parsers.name_resolver.resolve_name",
            return_value=SmilesResult(
                smiles="",
                confidence=0.0,
                source="none",
                warnings=["Could not resolve 'xyz'"],
            ),
        ):
            result = runner.invoke(main, ["resolve", "xyz"])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# AC-5.5  chemscheme ai "<description>"
# ---------------------------------------------------------------------------
class TestCliAi:
    """AC-5.5: `chemscheme ai '<NL>'` -> valid scheme.json."""

    @pytest.mark.integration
    def test_ai_produces_scheme_json(self) -> None:
        """CLI ai command outputs valid scheme JSON."""
        import json

        runner = CliRunner()
        mock_scheme = {
            "steps": [
                {
                    "id": "step-1",
                    "reactants": ["c1ccc(Br)cc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd(PPh3)4",
                }
            ]
        }

        with patch(
            "chemscheme.parsers.ai_intent.parse_intent",
            return_value=mock_scheme,
        ):
            result = runner.invoke(main, ["ai", "Suzuki coupling of bromobenzene"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "steps" in data

    @pytest.mark.integration
    def test_ai_failure_exit_code(self) -> None:
        """CLI ai command with API error -> exit code 1."""
        from chemscheme.parsers.ai_intent import AiServiceError

        runner = CliRunner()
        with patch(
            "chemscheme.parsers.ai_intent.parse_intent",
            side_effect=AiServiceError("AI service unavailable"),
        ):
            result = runner.invoke(main, ["ai", "anything"])

        assert result.exit_code == 1
        assert "unavailable" in result.output.lower() or "error" in result.output.lower()

    @pytest.mark.integration
    def test_ai_save_to_file(self) -> None:
        """CLI ai --output saves scheme.json to file."""
        import json

        runner = CliRunner()
        mock_scheme = {
            "steps": [
                {
                    "id": "step-1",
                    "reactants": ["C=C"],
                    "products": ["CCO"],
                    "arrow_above": "H2O",
                }
            ]
        }

        with runner.isolated_filesystem():
            with patch(
                "chemscheme.parsers.ai_intent.parse_intent",
                return_value=mock_scheme,
            ):
                result = runner.invoke(
                    main, ["ai", "--output", "out.json", "hydration of ethylene"]
                )

            assert result.exit_code == 0
            import os

            assert os.path.exists("out.json")
            with open("out.json") as f:
                data = json.load(f)
            assert "steps" in data
