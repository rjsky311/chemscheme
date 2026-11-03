"""Phase 21 Experimental Section Generator tests.

AC-21.1: Single-step → procedure paragraph (compound_labels used)
AC-21.2: Multi-step → numbered sections
AC-21.3: Condition assembly (reagents, catalyst, solvents, temperature, time, atmosphere)
AC-21.4: Domain library integration (reaction type matching)
AC-21.5: Yield + atmosphere annotation
AC-21.6: CLI generate-procedure command
AC-21.7: MCP generate_procedure tool
AC-21.8: Edge cases (empty scheme, no conditions, empty products)
"""

import json
from pathlib import Path

import pytest

from chemscheme.procedure_generator import (
    ProcedureResult,
    StepProcedure,
    generate_procedure,
)

# ---------------------------------------------------------------------------
# Helper: minimal scheme builder
# ---------------------------------------------------------------------------


def _make_scheme(
    steps: list[dict],  # type: ignore[type-arg]
    *,
    title: str = "Test Scheme",
) -> dict:  # type: ignore[type-arg]
    """Build a minimal valid scheme dict."""
    return {
        "schema_version": "1.0",
        "title": title,
        "steps": steps,
    }


def _suzuki_step(
    *,
    compound_labels: dict[str, str] | None = None,
    yield_: str | None = None,
) -> dict:  # type: ignore[type-arg]
    """A single Suzuki coupling step with typical conditions."""
    step: dict = {  # type: ignore[type-arg]
        "id": "step1",
        "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
        "products": ["c1ccc(-c2ccccc2)cc1"],
        "arrow_above": "Pd(PPh3)4, K2CO3",
        "solvents": ["THF", "H2O"],
        "temperature": "80°C",
        "time": "12h",
    }
    if compound_labels is not None:
        step["compound_labels"] = compound_labels
    if yield_ is not None:
        step["yield"] = yield_
    return step


# ---------------------------------------------------------------------------
# AC-21.1: Single-step → procedure paragraph
# ---------------------------------------------------------------------------


class TestSingleStepProcedure:
    """AC-21.1: Single-step scheme produces a procedure paragraph."""

    def test_basic_single_step(self) -> None:
        """GIVEN a 1-step scheme with compound_labels,
        WHEN generate_procedure is called,
        THEN the result text mentions compound names."""
        labels = {
            "c1ccc(Br)cc1": "1",
            "OB(O)c1ccccc1": "2",
            "c1ccc(-c2ccccc2)cc1": "3",
        }
        scheme = _make_scheme([_suzuki_step(compound_labels=labels)])
        result = generate_procedure(scheme)

        assert isinstance(result, ProcedureResult)
        assert "Compound 1" in result.text or "compound 1" in result.text.lower()
        assert len(result.steps) == 1
        assert len(result.warnings) == 0

    def test_conditions_in_text(self) -> None:
        """GIVEN a step with solvents, temperature, and time,
        THEN these appear in the result text."""
        labels = {"c1ccc(Br)cc1": "1", "OB(O)c1ccccc1": "2", "c1ccc(-c2ccccc2)cc1": "3"}
        scheme = _make_scheme([_suzuki_step(compound_labels=labels)])
        result = generate_procedure(scheme)

        text_lower = result.text.lower()
        assert "thf" in text_lower or "THF" in result.text
        assert "80" in result.text
        assert "12" in result.text

    def test_step_procedure_dataclass(self) -> None:
        """GIVEN a single-step result, THEN StepProcedure has expected fields."""
        labels = {"c1ccc(Br)cc1": "1", "OB(O)c1ccccc1": "2", "c1ccc(-c2ccccc2)cc1": "3"}
        scheme = _make_scheme([_suzuki_step(compound_labels=labels)])
        result = generate_procedure(scheme)

        sp = result.steps[0]
        assert isinstance(sp, StepProcedure)
        assert sp.step_number == 1
        assert sp.text  # non-empty


# ---------------------------------------------------------------------------
# AC-21.2: Multi-step → numbered sections
# ---------------------------------------------------------------------------


class TestMultiStepProcedure:
    """AC-21.2: Multi-step scheme → numbered step sections."""

    def test_three_step_scheme(self) -> None:
        """GIVEN a 3-step scheme, THEN result has 3 StepProcedure entries."""
        steps = [
            {
                "id": f"step{i}",
                "reactants": [f"r{i}"],
                "products": [f"p{i}"],
                "arrow_above": f"reagent{i}",
                "compound_labels": {f"r{i}": str(i), f"p{i}": str(i + 1)},
            }
            for i in range(1, 4)
        ]
        scheme = _make_scheme(steps)
        result = generate_procedure(scheme)

        assert len(result.steps) == 3
        assert result.steps[0].step_number == 1
        assert result.steps[1].step_number == 2
        assert result.steps[2].step_number == 3


# ---------------------------------------------------------------------------
# AC-21.3: Condition assembly
# ---------------------------------------------------------------------------


class TestConditionAssembly:
    """AC-21.3: Condition fields assembled into text."""

    def test_multiline_arrow_above(self) -> None:
        """GIVEN arrow_above as list[str], THEN all lines included."""
        step = {
            "id": "step1",
            "reactants": ["A"],
            "products": ["B"],
            "arrow_above": ["Pd(PPh3)4", "K2CO3", "THF/H2O"],
            "compound_labels": {"A": "1", "B": "2"},
        }
        scheme = _make_scheme([step])
        result = generate_procedure(scheme)

        assert "Pd(PPh3)4" in result.text or "Pd(PPh₃)₄" in result.text
        assert "K2CO3" in result.text or "K₂CO₃" in result.text

    def test_atmosphere_in_text(self) -> None:
        """AC-21.5: atmosphere field → 'under nitrogen atmosphere'."""
        step = {
            "id": "step1",
            "reactants": ["A"],
            "products": ["B"],
            "arrow_above": "Pd catalyst",
            "atmosphere": "N2",
            "compound_labels": {"A": "1", "B": "2"},
        }
        scheme = _make_scheme([step])
        result = generate_procedure(scheme)

        assert "nitrogen" in result.text.lower() or "N2" in result.text
        # "under ... atmosphere" pattern
        assert "atmosphere" in result.text.lower() or "N2" in result.text


# ---------------------------------------------------------------------------
# AC-21.4: Domain library integration
# ---------------------------------------------------------------------------


class TestDomainLibraryIntegration:
    """AC-21.4: Reaction type recognition from arrow_above."""

    def test_suzuki_coupling_recognized(self) -> None:
        """GIVEN arrow_above mentioning Pd(PPh3)4 + K2CO3,
        THEN identifies 'Suzuki coupling' in text."""
        labels = {"c1ccc(Br)cc1": "1", "OB(O)c1ccccc1": "2", "c1ccc(-c2ccccc2)cc1": "3"}
        scheme = _make_scheme([_suzuki_step(compound_labels=labels)])
        result = generate_procedure(scheme)

        # Should mention Suzuki coupling
        text_lower = result.text.lower()
        assert "suzuki" in text_lower

    def test_unknown_reaction_no_name(self) -> None:
        """GIVEN arrow_above with unknown reagents,
        THEN no reaction name is guessed."""
        step = {
            "id": "step1",
            "reactants": ["A"],
            "products": ["B"],
            "arrow_above": "MagicReagent-XYZ",
            "compound_labels": {"A": "1", "B": "2"},
        }
        scheme = _make_scheme([step])
        result = generate_procedure(scheme)

        # Should not contain any known reaction name
        text_lower = result.text.lower()
        assert "suzuki" not in text_lower
        assert "stille" not in text_lower


# ---------------------------------------------------------------------------
# AC-21.5: Yield annotation
# ---------------------------------------------------------------------------


class TestYieldAnnotation:
    """AC-21.5: Yield field → 'in X% yield'."""

    def test_yield_in_text(self) -> None:
        """GIVEN a step with yield '85%', THEN text includes '85% yield'."""
        labels = {"c1ccc(Br)cc1": "1", "OB(O)c1ccccc1": "2", "c1ccc(-c2ccccc2)cc1": "3"}
        scheme = _make_scheme([_suzuki_step(compound_labels=labels, yield_="85%")])
        result = generate_procedure(scheme)

        assert "85%" in result.text
        assert "yield" in result.text.lower()


# ---------------------------------------------------------------------------
# AC-21.3 (missing labels fallback)
# ---------------------------------------------------------------------------


class TestMissingLabels:
    """AC-21.1: Missing compound_labels → generic terms."""

    def test_no_compound_labels_fallback(self) -> None:
        """GIVEN no compound_labels, THEN uses generic terms + warning."""
        scheme = _make_scheme([_suzuki_step(compound_labels=None)])
        result = generate_procedure(scheme)

        # Should use generic terms
        text_lower = result.text.lower()
        assert "product" in text_lower or "starting material" in text_lower
        # Should warn
        assert len(result.warnings) > 0
        assert any("label" in w.lower() or "compound" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# AC-21.8: Edge cases (adversarial)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """AC-21.8: Edge cases and robustness."""

    def test_empty_scheme_raises(self) -> None:
        """GIVEN scheme with no steps, THEN raises ValueError."""
        scheme = _make_scheme([])
        with pytest.raises(ValueError, match="(?i)step|empty"):
            generate_procedure(scheme)

    def test_no_conditions_generic_sentence(self) -> None:
        """GIVEN a step with only reactants/products (no conditions),
        THEN uses generic conversion sentence + warning."""
        step = {
            "id": "step1",
            "reactants": ["A"],
            "products": ["B"],
            "arrow_above": "",
            "compound_labels": {"A": "1", "B": "2"},
        }
        scheme = _make_scheme([step])
        result = generate_procedure(scheme)

        assert len(result.steps) == 1
        assert result.steps[0].text  # still produces text
        assert len(result.warnings) > 0  # warns about missing conditions

    def test_empty_products_skipped(self) -> None:
        """GIVEN a step with products=[], THEN step skipped with warning."""
        step = {
            "id": "step1",
            "reactants": ["A"],
            "products": [],
            "arrow_above": "some reagent",
        }
        scheme = _make_scheme([step])
        result = generate_procedure(scheme)

        # Step should be skipped — 0 step procedures
        assert len(result.steps) == 0
        assert len(result.warnings) > 0

    def test_invalid_scheme_returns_errors(self) -> None:
        """GIVEN an invalid scheme (missing steps), THEN raises ValueError."""
        scheme = {"schema_version": "1.0"}  # no steps
        with pytest.raises(ValueError, match="(?i)step|required"):
            generate_procedure(scheme)

    def test_missing_reactants_skipped(self) -> None:
        """C2 regression: step with no reactants → skipped with warning."""
        step = {
            "id": "step1",
            "products": ["B"],
            "arrow_above": "reagent",
        }
        scheme = _make_scheme([step])
        result = generate_procedure(scheme)
        assert len(result.steps) == 0
        assert any("reactant" in w.lower() for w in result.warnings)

    def test_empty_reactants_skipped(self) -> None:
        """C2 regression: step with reactants=[] → skipped with warning."""
        step = {
            "id": "step1",
            "reactants": [],
            "products": ["B"],
            "arrow_above": "reagent",
        }
        scheme = _make_scheme([step])
        result = generate_procedure(scheme)
        assert len(result.steps) == 0
        assert any("reactant" in w.lower() for w in result.warnings)

    def test_reagents_field_in_conditions(self) -> None:
        """C3 regression: schema 'reagents' field included in procedure text."""
        step = {
            "id": "step1",
            "reactants": ["A"],
            "products": ["B"],
            "arrow_above": "",
            "reagents": ["NBS", "BPO"],
            "compound_labels": {"A": "1", "B": "2"},
        }
        scheme = _make_scheme([step])
        result = generate_procedure(scheme)
        assert "NBS" in result.text

    def test_cli_malformed_json(self, tmp_path: Path) -> None:
        """C4 regression: malformed JSON → error message, not crash."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["generate-procedure", str(bad_file)])
        assert result.exit_code != 0
        assert "invalid JSON" in result.output or "Error" in result.output


# ---------------------------------------------------------------------------
# AC-21.6: CLI generate-procedure
# ---------------------------------------------------------------------------


class TestCLIGenerateProcedure:
    """AC-21.6: CLI command outputs procedure text."""

    def test_cli_text_output(self, tmp_path: Path) -> None:
        """GIVEN a scheme.json file, WHEN CLI generate-procedure is run, THEN outputs text."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        labels = {"c1ccc(Br)cc1": "1", "OB(O)c1ccccc1": "2", "c1ccc(-c2ccccc2)cc1": "3"}
        scheme = _make_scheme([_suzuki_step(compound_labels=labels)])
        scheme_file = tmp_path / "scheme.json"
        scheme_file.write_text(json.dumps(scheme), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["generate-procedure", str(scheme_file)])

        assert result.exit_code == 0
        assert "Compound" in result.output or "compound" in result.output.lower()

    def test_cli_json_output(self, tmp_path: Path) -> None:
        """GIVEN --format json, THEN outputs valid JSON with expected fields."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        labels = {"c1ccc(Br)cc1": "1", "OB(O)c1ccccc1": "2", "c1ccc(-c2ccccc2)cc1": "3"}
        scheme = _make_scheme([_suzuki_step(compound_labels=labels)])
        scheme_file = tmp_path / "scheme.json"
        scheme_file.write_text(json.dumps(scheme), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["generate-procedure", str(scheme_file), "--format", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "text" in data
        assert "steps" in data
        assert "warnings" in data

    def test_cli_output_to_file(self, tmp_path: Path) -> None:
        """GIVEN --output flag, THEN writes to file."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        labels = {"c1ccc(Br)cc1": "1", "OB(O)c1ccccc1": "2", "c1ccc(-c2ccccc2)cc1": "3"}
        scheme = _make_scheme([_suzuki_step(compound_labels=labels)])
        scheme_file = tmp_path / "scheme.json"
        scheme_file.write_text(json.dumps(scheme), encoding="utf-8")

        out_file = tmp_path / "procedure.txt"
        runner = CliRunner()
        result = runner.invoke(
            main, ["generate-procedure", str(scheme_file), "--output", str(out_file)]
        )

        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert len(content) > 0


# ---------------------------------------------------------------------------
# AC-21.7: MCP generate_procedure
# ---------------------------------------------------------------------------


class TestMCPGenerateProcedure:
    """AC-21.7: MCP tool returns structured result."""

    @pytest.mark.asyncio
    async def test_mcp_generate_procedure(self) -> None:
        """GIVEN a valid scheme dict, WHEN MCP tool called, THEN returns JSON."""
        from chemscheme.mcp_server import generate_procedure as mcp_gen

        labels = {"c1ccc(Br)cc1": "1", "OB(O)c1ccccc1": "2", "c1ccc(-c2ccccc2)cc1": "3"}
        scheme = _make_scheme([_suzuki_step(compound_labels=labels)])

        # C5 fix: MCP now accepts dict directly (consistent with draw_scheme)
        result_json = await mcp_gen(scheme)
        data = json.loads(result_json)

        assert "text" in data
        assert "steps" in data
        assert "warnings" in data
        assert isinstance(data["steps"], list)
        assert len(data["steps"]) == 1
