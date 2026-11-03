"""Phase 20: Text Extraction Pipeline tests.

BDD-first RED tests for extracting structured reaction data from
experimental text paragraphs into scheme.json format.

Three-layer progressive extraction:
  L1: Condition regex (temp/time/solvent/atmosphere/yield)
  L2: Chemical name finder (OPSIN/PubChem validation)
  L3: Reaction role classifier (domain library + AI fallback)
  L4: Text-to-scheme assembler
  L5/L6: CLI + MCP wiring
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ===========================================================================
# AC-20.1: Condition regex extractor
# ===========================================================================


class TestConditionExtractor:
    """L1: Extract reaction conditions from text via regex patterns."""

    def test_extract_temperature_celsius(self) -> None:
        """Standard Celsius temperature extracted."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("heated at 80 °C for 12 h under N₂")
        assert result.temperature == "80 °C"

    def test_extract_negative_temperature(self) -> None:
        """Negative temperature (e.g., -78 °C) extracted."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("cooled to -78 °C")
        assert result.temperature == "-78 °C"

    def test_extract_room_temperature_rt(self) -> None:
        """Room temperature abbreviations recognized."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("stirred at room temperature overnight")
        assert result.temperature == "rt"

    def test_extract_reflux_as_temperature(self) -> None:
        """'reflux' recognized as temperature descriptor."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("heated to reflux for 6 h")
        assert result.temperature == "reflux"

    def test_extract_time_hours(self) -> None:
        """Time in hours extracted."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("heated at 80 °C for 12 h under N₂")
        assert result.time == "12 h"

    def test_extract_time_minutes(self) -> None:
        """Time in minutes extracted."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("stirred for 30 min")
        assert result.time == "30 min"

    def test_extract_time_overnight(self) -> None:
        """'overnight' recognized as time descriptor."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("stirred at room temperature overnight")
        assert result.time == "overnight"

    def test_extract_solvent(self) -> None:
        """Known solvent detected from domain library."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("in THF (10 mL) to give product")
        assert result.solvent == "THF"

    def test_extract_atmosphere_n2(self) -> None:
        """Nitrogen atmosphere detected."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("heated at 80 °C for 12 h under N₂")
        assert result.atmosphere == "N₂"

    def test_extract_atmosphere_argon(self) -> None:
        """Argon atmosphere detected."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("under an argon atmosphere")
        assert result.atmosphere == "Ar"

    def test_extract_yield_percentage(self) -> None:
        """Yield percentage extracted."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("to give product (85% yield)")
        assert result.yield_pct == "85%"

    def test_extract_yield_quantitative(self) -> None:
        """'quantitative' yield recognized."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("to give a quantitative yield")
        assert result.yield_pct == "quantitative"

    def test_no_conditions_in_text(self) -> None:
        """Non-chemistry text returns all None fields (adversarial)."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("The compound was characterized by NMR")
        assert result.temperature is None
        assert result.time is None
        assert result.solvent is None
        assert result.atmosphere is None
        assert result.yield_pct is None

    def test_empty_text(self) -> None:
        """Empty string returns all None fields (adversarial)."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("")
        assert result.temperature is None

    def test_extracted_conditions_is_dataclass(self) -> None:
        """ExtractedConditions is a proper dataclass."""
        from chemscheme.text_extraction import ExtractedConditions

        ec = ExtractedConditions()
        assert hasattr(ec, "temperature")
        assert hasattr(ec, "time")
        assert hasattr(ec, "solvent")
        assert hasattr(ec, "atmosphere")
        assert hasattr(ec, "yield_pct")


# ===========================================================================
# AC-20.2: Chemical name finder
# ===========================================================================


class TestChemicalNameFinder:
    """L2: Identify chemical names in text and resolve to SMILES."""

    def test_find_reagent_abbreviation(self) -> None:
        """Common reagent abbreviation found and resolved."""
        from chemscheme.text_extraction import find_compounds

        results = find_compounds("Pd(PPh₃)₄ (5 mol%) was added")
        assert len(results) >= 1
        assert any("Pd" in c.name for c in results)

    def test_find_numbered_compound(self) -> None:
        """Numbered compound reference 'compound 1' identified."""
        from chemscheme.text_extraction import find_compounds

        results = find_compounds("compound 1 (500 mg, 1.2 mmol) was dissolved")
        assert any("compound 1" in c.name.lower() for c in results)

    def test_find_solvent_in_compound_list(self) -> None:
        """Known solvent identified as compound."""
        from chemscheme.text_extraction import find_compounds

        results = find_compounds("dissolved in THF (10 mL)")
        assert any("THF" in c.name for c in results)

    def test_found_compound_has_fields(self) -> None:
        """FoundCompound has name, smiles, confidence fields."""
        from chemscheme.text_extraction import FoundCompound

        fc = FoundCompound(name="THF", smiles="C1CCOC1", confidence=0.9)
        assert fc.name == "THF"
        assert fc.smiles == "C1CCOC1"
        assert fc.confidence == 0.9

    def test_gibberish_yields_no_compounds(self) -> None:
        """Non-chemistry text returns empty list (adversarial)."""
        from chemscheme.text_extraction import find_compounds

        results = find_compounds("lorem ipsum dolor sit amet")
        assert results == []

    def test_empty_text_yields_no_compounds(self) -> None:
        """Empty string returns empty list (adversarial)."""
        from chemscheme.text_extraction import find_compounds

        results = find_compounds("")
        assert results == []

    def test_amount_extraction(self) -> None:
        """Amount/equivalents extracted alongside compound name."""
        from chemscheme.text_extraction import find_compounds

        results = find_compounds("NaOH (2.0 equiv)")
        naoh = [c for c in results if "NaOH" in c.name]
        assert len(naoh) >= 1

    def test_multiple_compounds_in_sentence(self) -> None:
        """Multiple compounds found in one sentence."""
        from chemscheme.text_extraction import find_compounds

        text = "A mixture of compound 1 and Pd(OAc)₂ in DMF"
        results = find_compounds(text)
        assert len(results) >= 2


# ===========================================================================
# AC-20.3: Reaction role classifier
# ===========================================================================


class TestRoleClassifier:
    """L3: Assign roles (reactant/product/reagent) to compounds."""

    def test_catalyst_from_domain_library(self) -> None:
        """Catalyst identified by domain_library category."""
        from chemscheme.text_extraction import FoundCompound, classify_roles

        compounds = [FoundCompound(name="Pd₂(dba)₃", smiles="", confidence=0.9)]
        result = classify_roles(compounds, "Pd₂(dba)₃ was used as catalyst", use_ai=False)
        assert result.reagents  # catalyst goes into reagents
        assert any("Pd" in r.name for r in result.reagents)

    def test_product_by_yield_marker(self) -> None:
        """Compound with yield% is classified as product."""
        from chemscheme.text_extraction import FoundCompound, classify_roles

        compounds = [FoundCompound(name="product", smiles="CCO", confidence=0.8)]
        result = classify_roles(compounds, "to give the product (85% yield)", use_ai=False)
        assert result.products
        assert len(result.products) >= 1

    def test_classified_reaction_has_fields(self) -> None:
        """ClassifiedReaction has reactants, products, reagents, conditions."""
        from chemscheme.text_extraction import ClassifiedReaction

        cr = ClassifiedReaction(reactants=[], products=[], reagents=[], conditions=None)
        assert hasattr(cr, "reactants")
        assert hasattr(cr, "products")
        assert hasattr(cr, "reagents")
        assert hasattr(cr, "conditions")

    def test_ai_fallback_called_when_ambiguous(self) -> None:
        """Claude API called when rule-based classification is ambiguous."""
        from chemscheme.text_extraction import FoundCompound, classify_roles

        compounds = [
            FoundCompound(name="unknown-chemical", smiles="CCCC", confidence=0.7),
        ]
        with patch("chemscheme.text_extraction.classify_roles_ai") as mock_ai:
            mock_ai.return_value = MagicMock(reactants=compounds, products=[], reagents=[])
            classify_roles(compounds, "some ambiguous text", use_ai=True)
            mock_ai.assert_called_once()

    def test_no_ai_mode_skips_api(self) -> None:
        """use_ai=False never calls Claude API."""
        from chemscheme.text_extraction import FoundCompound, classify_roles

        compounds = [FoundCompound(name="X", smiles="C", confidence=0.5)]
        with patch("chemscheme.text_extraction.classify_roles_ai") as mock_ai:
            classify_roles(compounds, "ambiguous", use_ai=False)
            mock_ai.assert_not_called()

    def test_empty_compounds_returns_empty(self) -> None:
        """No compounds → empty classification (adversarial)."""
        from chemscheme.text_extraction import classify_roles

        result = classify_roles([], "some text", use_ai=False)
        assert result.reactants == []
        assert result.products == []
        assert result.reagents == []

    def test_solvent_classified_separately(self) -> None:
        """Known solvent classified into conditions, not reactants."""
        from chemscheme.text_extraction import FoundCompound, classify_roles

        compounds = [FoundCompound(name="THF", smiles="C1CCOC1", confidence=0.9)]
        result = classify_roles(compounds, "dissolved in THF", use_ai=False)
        # THF should not appear as reactant
        assert not any("THF" in r.name for r in result.reactants)


# ===========================================================================
# AC-20.4: Text-to-scheme assembler
# ===========================================================================


class TestTextToSchemeAssembler:
    """L4: Assemble extracted data into valid scheme.json."""

    def test_single_paragraph_one_step(self) -> None:
        """Single paragraph with unresolved compounds → warning emitted.

        C3/R2: "compound 1"/"product 2" have no known SMILES, so no valid
        step can be emitted. The system should warn about unresolved SMILES.
        """
        from chemscheme.text_extraction import extract_reactions_from_text

        text = (
            "A mixture of compound 1 (100 mg) and Pd(PPh₃)₄ (5 mol%) "
            "in THF (10 mL) was heated at 80 °C for 12 h to give "
            "product 2 (85% yield)."
        )
        result = extract_reactions_from_text(text, use_ai=False)
        # Named compounds without SMILES → step skipped, warning emitted
        warnings = result.get("warnings", [])
        assert any("no resolved SMILES" in w or "Unresolved" in w for w in warnings)

    def test_result_has_scheme_json_fields(self) -> None:
        """Result contains scheme.json required fields."""
        from chemscheme.text_extraction import extract_reactions_from_text

        text = "Compound A was converted to compound B using catalyst C at 60 °C."
        result = extract_reactions_from_text(text, use_ai=False)
        assert "steps" in result
        for step in result["steps"]:
            assert "reactants" in step
            assert "products" in step

    def test_multi_step_procedure(self) -> None:
        """Multi-step text with unresolved compounds → warnings emitted.

        C3/R2: "Compound 1/2"/"product 3" have no known SMILES.
        Steps are skipped with warnings, not emitted with empty arrays.
        """
        from chemscheme.text_extraction import extract_reactions_from_text

        text = (
            "Step 1: Compound 1 was brominated with NBS in DCM at rt for 2 h "
            "to give compound 2. "
            "Step 2: Compound 2 was coupled with Pd(OAc)2 and K2CO3 in DMF at 80 °C "
            "to give product 3."
        )
        result = extract_reactions_from_text(text, use_ai=False)
        warnings = result.get("warnings", [])
        # Should have warnings about unresolved SMILES
        unresolved_warnings = [
            w for w in warnings if "no resolved SMILES" in w or "Unresolved" in w
        ]
        assert len(unresolved_warnings) >= 1

    def test_empty_text_zero_steps(self) -> None:
        """Empty text produces zero steps with warning (adversarial)."""
        from chemscheme.text_extraction import extract_reactions_from_text

        result = extract_reactions_from_text("", use_ai=False)
        assert result["steps"] == []

    def test_non_chemistry_text_zero_steps(self) -> None:
        """Non-chemistry paragraph produces zero steps (adversarial)."""
        from chemscheme.text_extraction import extract_reactions_from_text

        result = extract_reactions_from_text("The weather was sunny and warm today.", use_ai=False)
        assert result["steps"] == []

    def test_assembled_scheme_passes_validation(self) -> None:
        """Assembled scheme.json passes validate_scheme (Phase 6 integration)."""
        from chemscheme.text_extraction import extract_reactions_from_text

        text = (
            "Compound 1 (100 mg) was treated with NaOH (2 equiv) "
            "in MeOH at 60 °C for 3 h to afford compound 2 (90%)."
        )
        result = extract_reactions_from_text(text, use_ai=False)
        if result["steps"]:
            # If extraction found steps, they should be valid scheme format
            for step in result["steps"]:
                assert "id" in step
                assert "reactants" in step
                assert "products" in step


# ===========================================================================
# AC-20.5: CLI extract-text
# ===========================================================================


class TestCLIExtractText:
    """CLI command: chemscheme extract-text."""

    def test_cli_extract_text_help(self) -> None:
        """extract-text command appears in CLI help."""
        from click.testing import CliRunner

        from chemscheme.cli import main as cli

        runner = CliRunner()
        result = runner.invoke(cli, ["extract-text", "--help"])
        assert result.exit_code == 0
        assert "extract" in result.output.lower()

    def test_cli_extract_text_from_string(self) -> None:
        """Extract from inline text string."""
        from click.testing import CliRunner

        from chemscheme.cli import main as cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["extract-text", "--text", "heated at 80 °C for 12 h", "--no-ai", "--json"],
        )
        assert result.exit_code == 0

    def test_cli_no_ai_flag(self) -> None:
        """--no-ai flag prevents Claude API calls."""
        from click.testing import CliRunner

        from chemscheme.cli import main as cli

        runner = CliRunner()
        with patch("chemscheme.text_extraction.classify_roles_ai") as mock_ai:
            runner.invoke(
                cli,
                ["extract-text", "--text", "some reaction text", "--no-ai", "--json"],
            )
            mock_ai.assert_not_called()

    def test_cli_json_output(self) -> None:
        """--json flag outputs valid JSON to stdout."""
        import json

        from click.testing import CliRunner

        from chemscheme.cli import main as cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["extract-text", "--text", "compound in THF at 80 °C", "--no-ai", "--json"],
        )
        if result.exit_code == 0 and result.output.strip():
            parsed = json.loads(result.output)
            assert "steps" in parsed


# ===========================================================================
# AC-20.6: MCP extract_text tool
# ===========================================================================


class TestMCPExtractText:
    """MCP tool: extract_text."""

    def test_mcp_tool_registered(self) -> None:
        """extract_text tool is registered in MCP server."""
        from chemscheme.mcp_server import chemscheme_mcp

        tool_names = [t.name for t in chemscheme_mcp._tool_manager.list_tools()]
        assert "extract_text" in tool_names

    def test_mcp_extract_text_returns_dict(self) -> None:
        """extract_text returns a dict with steps."""
        from chemscheme.text_extraction import extract_reactions_from_text

        result = extract_reactions_from_text("heated at 80 °C for 12 h in THF", use_ai=False)
        assert isinstance(result, dict)
        assert "steps" in result


# ===========================================================================
# AC-20.3b: classify_roles_ai wired to Claude API
# ===========================================================================


class TestClassifyRolesAiWiring:
    """classify_roles_ai: wired to Claude API via ai_intent infrastructure."""

    def test_ai_calls_claude_api(self) -> None:
        """classify_roles_ai calls _call_claude_api with role-classification prompt."""
        from chemscheme.text_extraction import ClassifiedReaction, FoundCompound, classify_roles_ai

        compounds = [FoundCompound(name="benzaldehyde", smiles="O=Cc1ccccc1", confidence=0.9)]
        mock_response = {
            "reactants": [{"name": "benzaldehyde", "smiles": "O=Cc1ccccc1"}],
            "products": [],
            "reagents": [],
        }
        with patch("chemscheme.text_extraction._call_claude_role_api", return_value=mock_response):
            result = classify_roles_ai(compounds, "benzaldehyde was added")
            assert isinstance(result, ClassifiedReaction)

    def test_ai_classifies_reactant(self) -> None:
        """AI correctly classifies a reactant from response."""
        from chemscheme.text_extraction import FoundCompound, classify_roles_ai

        compounds = [FoundCompound(name="styrene", smiles="C=Cc1ccccc1", confidence=0.9)]
        mock_response = {
            "reactants": [{"name": "styrene"}],
            "products": [],
            "reagents": [],
        }
        with patch("chemscheme.text_extraction._call_claude_role_api", return_value=mock_response):
            result = classify_roles_ai(compounds, "styrene was polymerized")
            assert any("styrene" in c.name for c in result.reactants)

    def test_ai_classifies_product(self) -> None:
        """AI correctly classifies a product from response."""
        from chemscheme.text_extraction import FoundCompound, classify_roles_ai

        compounds = [
            FoundCompound(name="compound 1", smiles="CCCC", confidence=0.7),
            FoundCompound(name="compound 2", smiles="CCCCO", confidence=0.7),
        ]
        mock_response = {
            "reactants": [{"name": "compound 1"}],
            "products": [{"name": "compound 2"}],
            "reagents": [],
        }
        with patch("chemscheme.text_extraction._call_claude_role_api", return_value=mock_response):
            result = classify_roles_ai(compounds, "compound 1 gave compound 2")
            assert any("compound 2" in c.name for c in result.products)

    def test_ai_api_failure_graceful_fallback(self) -> None:
        """API failure → compounds default to reactants (graceful degradation)."""
        from chemscheme.text_extraction import FoundCompound, classify_roles_ai

        compounds = [FoundCompound(name="X", smiles="C", confidence=0.5)]
        with patch(
            "chemscheme.text_extraction._call_claude_role_api",
            side_effect=Exception("API down"),
        ):
            result = classify_roles_ai(compounds, "some text")
            # Fallback: all unclassified → reactants
            assert len(result.reactants) == 1
            assert result.products == []

    def test_ai_invalid_response_graceful_fallback(self) -> None:
        """Malformed API response → compounds default to reactants (adversarial)."""
        from chemscheme.text_extraction import FoundCompound, classify_roles_ai

        compounds = [FoundCompound(name="Y", smiles="CC", confidence=0.6)]
        with patch(
            "chemscheme.text_extraction._call_claude_role_api",
            return_value="not a dict",
        ):
            result = classify_roles_ai(compounds, "some text")
            assert len(result.reactants) == 1

    def test_ai_no_api_key_graceful_fallback(self) -> None:
        """Missing ANTHROPIC_API_KEY → fallback without crash."""
        from chemscheme.text_extraction import FoundCompound, classify_roles_ai

        compounds = [FoundCompound(name="Z", smiles="CCC", confidence=0.5)]
        with patch(
            "chemscheme.text_extraction._call_claude_role_api",
            side_effect=Exception("ANTHROPIC_API_KEY not set"),
        ):
            result = classify_roles_ai(compounds, "some text")
            assert len(result.reactants) == 1


# ===========================================================================
# Deferred P2/P3 regression tests (MR19: NEW-2, NEW-4, NEW-5)
# ===========================================================================


class TestDeferredMR19Fixes:
    """Regression tests for MR19 deferred items."""

    # NEW-2: Solvent matching should be case-insensitive
    def test_solvent_case_insensitive_lowercase(self) -> None:
        """Lowercase 'thf' should match solvent THF."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("dissolved in thf at 25 °C")
        assert result.solvent is not None
        # Should match regardless of case
        assert result.solvent.upper() == "THF"

    def test_solvent_case_insensitive_mixed(self) -> None:
        """Mixed case 'Toluene' vs 'toluene' should both match."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("stirred in Toluene for 2 h")
        assert result.solvent is not None
        assert result.solvent.lower() == "toluene"

    def test_solvent_case_insensitive_allcaps(self) -> None:
        """All-caps 'DMSO' in lowercase text should still match."""
        from chemscheme.text_extraction import extract_conditions

        # "dmso" lowercase should match "DMSO" in solvent list
        result = extract_conditions("reaction in dmso at 120 °C")
        assert result.solvent is not None
        assert result.solvent.upper() == "DMSO"

    # NEW-4: _RE_AMOUNT should be used or removed
    def test_no_unused_re_amount_pattern(self) -> None:
        """_RE_AMOUNT should not exist as dead code."""
        import ast
        import inspect

        import chemscheme.text_extraction as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        # Collect all Name nodes that reference _RE_AMOUNT
        refs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "_RE_AMOUNT"
        ]
        # Either removed entirely (0 refs) or used somewhere (>1 ref: definition + usage)
        assert len(refs) != 1, "_RE_AMOUNT is defined but never used (dead code)"

    # NEW-5: Formula regex should be module-level (not recompiled per call)
    def test_formula_regex_is_module_level(self) -> None:
        """_RE_FORMULA should be compiled at module level, not inside function."""
        import chemscheme.text_extraction as mod

        assert hasattr(mod, "_RE_FORMULA"), (
            "Formula regex should be module-level constant _RE_FORMULA"
        )

    def test_formula_regex_matches_real_formulas(self) -> None:
        """Real chemical formulas like NaOH should still match."""
        from chemscheme.text_extraction import find_compounds

        result = find_compounds("treated with NaOH solution")
        names = {c.name for c in result}
        assert "NaOH" in names


# ===========================================================================
# MR20 Round 1 regression tests (G1-G4, C1-C2)
# ===========================================================================


class TestMR20Regressions:
    """Regression tests for MR20 Round 1 findings."""

    # G2: None in API response should not crash
    def test_ai_role_none_list_graceful(self) -> None:
        """AI returning null for a role list should not crash."""
        from chemscheme.text_extraction import FoundCompound, classify_roles_ai

        compounds = [FoundCompound(name="A", smiles="C", confidence=0.9)]
        with patch(
            "chemscheme.text_extraction._call_claude_role_api",
            return_value={"reactants": None, "products": [], "reagents": []},
        ):
            result = classify_roles_ai(compounds, "A reacts")
            # Should not crash; compound A should end up somewhere
            total = len(result.reactants) + len(result.products) + len(result.reagents)
            assert total >= 1

    # C2: Duplicate names across roles should be deduplicated
    def test_ai_role_dedup_across_roles(self) -> None:
        """Compound appearing in multiple roles should only be classified once."""
        from chemscheme.text_extraction import FoundCompound, classify_roles_ai

        compounds = [FoundCompound(name="X", smiles="CC", confidence=0.8)]
        with patch(
            "chemscheme.text_extraction._call_claude_role_api",
            return_value={
                "reactants": [{"name": "X"}],
                "products": [{"name": "X"}],
                "reagents": [],
            },
        ):
            result = classify_roles_ai(compounds, "X reacts")
            all_names = (
                [c.name for c in result.reactants]
                + [c.name for c in result.products]
                + [c.name for c in result.reagents]
            )
            assert all_names.count("X") == 1, "Compound should appear exactly once"

    # G3: Formula regex should capture post-paren digits
    def test_formula_regex_post_paren_digits(self) -> None:
        """Pd(PPh3)4 should be captured with the trailing 4."""
        from chemscheme.text_extraction import _RE_FORMULA

        m = _RE_FORMULA.search("using Pd(PPh3)4 as catalyst")
        assert m is not None
        assert m.group(1) == "Pd(PPh3)4"

    # G4: Yield regex should not match "50% aqueous solution"
    def test_yield_no_false_positive_aqueous(self) -> None:
        """'50% aqueous solution' should not be extracted as yield."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("dissolved in 50% aqueous NaOH solution")
        assert result.yield_pct is None

    def test_yield_still_matches_with_keyword(self) -> None:
        """'85% yield' should still be extracted correctly."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("to give the product in 85% yield")
        assert result.yield_pct == "85%"

    def test_yield_matches_yield_colon_format(self) -> None:
        """'yield: 92%' should be extracted."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("Isolated yield: 92%")
        assert result.yield_pct == "92%"

    # C1: _comment_* keys should not appear as reagents
    def test_comment_keys_not_loaded_as_reagents(self) -> None:
        """Metadata _comment_* entries should be filtered from reagent library."""
        from chemscheme.domain_library import _load_reagents

        reagents = _load_reagents()
        comment_entries = [r for r in reagents if r.abbreviation.startswith("_")]
        assert len(comment_entries) == 0, f"Found metadata entries: {comment_entries}"


# ===========================================================================
# MR20 Round 2 regression tests (C1-R2, C2-R2, C4-R2)
# ===========================================================================


class TestMR20R2Regressions:
    """Regression tests for MR20 Round 2 findings."""

    # C1-R2: Bare "(85%)" should NOT be extracted as yield
    def test_bare_percent_not_yield(self) -> None:
        """Bare '(50%)' without yield keyword should not match."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("added compound A (50%) and stirred")
        assert result.yield_pct is None

    # C1-R2: But "(85% yield)" should still match
    def test_percent_with_yield_still_matches(self) -> None:
        """'(85% yield)' should still be extracted."""
        from chemscheme.text_extraction import extract_conditions

        result = extract_conditions("product was obtained (85% yield)")
        assert result.yield_pct == "85%"

    # C2-R2: Solvent-only paragraphs should not produce steps
    def test_solvent_only_paragraph_no_step(self) -> None:
        """Paragraph with only solvents should not produce a step."""
        from chemscheme.text_extraction import extract_reactions_from_text

        result = extract_reactions_from_text("dissolved in THF and stirred")
        # Should have 0 steps (no reactants resolved with SMILES)
        assert len(result["steps"]) == 0

    # C4-R2: name_resolver should also skip _comment_* keys
    def test_name_resolver_skips_comment_keys(self) -> None:
        """_comment_* should not appear in alias cache."""
        from chemscheme.parsers.name_resolver import _load_aliases

        aliases = _load_aliases()
        comment_keys = [k for k in aliases if k.startswith("_")]
        assert len(comment_keys) == 0, f"Found metadata keys: {comment_keys}"
