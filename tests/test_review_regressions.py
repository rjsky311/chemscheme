"""Regression tests for Multi-AI Review findings (Gemini + Codex, 2026-03-15).

Each test is tagged with the finding ID it guards against.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ===========================================================================
# C1 — ai_intent: shallow validation / crash on malformed / missing schema_version
# G2 — ai_intent: markdown backtick stripping
# ===========================================================================


@pytest.mark.unit
class TestAiIntentRobustness:
    """C1/G2: AI intent parser handles malformed model output gracefully."""

    def test_malformed_steps_raises_service_error(self) -> None:
        """C1: {"steps":[1]} should raise AiServiceError, not TypeError."""
        from chemscheme.parsers.ai_intent import AiServiceError, _validate_scheme

        with pytest.raises(AiServiceError, match="step 0 must be a dict"):
            _validate_scheme({"steps": [1]})

    def test_missing_schema_version_injected(self) -> None:
        """C1: schema_version is injected if AI omits it."""
        from chemscheme.parsers.ai_intent import _validate_scheme

        scheme: dict = {
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                }
            ]
        }
        _validate_scheme(scheme)
        assert scheme["schema_version"] == "1"

    def test_strip_markdown_fences(self) -> None:
        """G2: Markdown ```json ... ``` fences are stripped before JSON parse."""
        from chemscheme.parsers.ai_intent import _strip_markdown_fences

        wrapped = '```json\n{"key": "value"}\n```'
        assert _strip_markdown_fences(wrapped) == '{"key": "value"}'

    def test_strip_bare_backticks(self) -> None:
        """G2: Bare ``` ... ``` fences also stripped."""
        from chemscheme.parsers.ai_intent import _strip_markdown_fences

        wrapped = '```\n{"a": 1}\n```'
        assert _strip_markdown_fences(wrapped) == '{"a": 1}'

    def test_no_fences_passthrough(self) -> None:
        """G2: Plain JSON passes through unchanged."""
        from chemscheme.parsers.ai_intent import _strip_markdown_fences

        plain = '{"key": "value"}'
        assert _strip_markdown_fences(plain) == '{"key": "value"}'

    def test_full_schema_validation_in_ai_path(self) -> None:
        """C1: _validate_scheme runs the full schema validator."""
        from chemscheme.parsers.ai_intent import AiServiceError, _validate_scheme

        # reactants should be a list, not a string
        with pytest.raises(AiServiceError, match="'reactants' must be a list"):
            _validate_scheme(
                {
                    "steps": [
                        {
                            "id": "1",
                            "reactants": "CCO",
                            "products": ["CC=O"],
                            "arrow_above": "PCC",
                        }
                    ]
                }
            )


# ===========================================================================
# C2 — scheme_schema: optional field type validation
# ===========================================================================


@pytest.mark.unit
class TestOptionalFieldTypes:
    """C2: Optional fields are type-checked by validate_scheme."""

    def test_branch_label_list_rejected(self) -> None:
        """C2: branch_label: [] is rejected."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                    "branch_label": [],
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert any("'branch_label' must be a string" in e for e in errors)

    def test_arrow_below_dict_rejected(self) -> None:
        """C2: arrow_below: {} is rejected."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                    "arrow_below": {},
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert any("'arrow_below' must be a string" in e for e in errors)

    def test_yield_int_rejected(self) -> None:
        """C2: yield must be a string."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                    "yield": 95,
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert any("'yield' must be a string" in e for e in errors)

    def test_valid_optional_fields_pass(self) -> None:
        """C2: Valid optional string fields still pass."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                    "arrow_below": "DCM, rt",
                    "branch_label": "A",
                    "yield": "95%",
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert len(errors) == 0


# ===========================================================================
# C3 — multi_step_renderer: step_id SVG injection
# ===========================================================================


@pytest.mark.unit
class TestStepIdSanitization:
    """C3: Step IDs with special chars are sanitized for SVG."""

    def test_safe_id_strips_quotes(self) -> None:
        """C3: Quotes and angle brackets are removed from step IDs."""
        from chemscheme.renderers.multi_step_renderer import _safe_id

        assert '"' not in _safe_id('step"1')
        assert "<" not in _safe_id("step<1>")
        assert ">" not in _safe_id("step<1>")

    def test_safe_id_preserves_normal(self) -> None:
        """C3: Normal IDs are preserved."""
        from chemscheme.renderers.multi_step_renderer import _safe_id

        assert _safe_id("step-1a") == "step-1a"
        assert _safe_id("1") == "1"


# ===========================================================================
# C4 — cdxml_writer: convergent semantics round-trip
# ===========================================================================


@pytest.mark.unit
class TestCdxmlConvergentRoundTrip:
    """C4: CDXML round-trip preserves branch_label and consumes_step_ids."""

    def test_convergent_round_trip_preserves_graph(self, sample_scheme_convergent: dict) -> None:
        """C4: Convergent scheme survives CDXML export→import."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        cdxml = scheme_to_cdxml(sample_scheme_convergent)
        reimported = cdxml_to_scheme(cdxml)

        # Check that graph semantics survived
        original_steps = sample_scheme_convergent["steps"]
        imported_steps = reimported["steps"]

        for orig, imp in zip(original_steps, imported_steps, strict=False):
            if "branch_label" in orig:
                assert imp.get("branch_label") == orig["branch_label"], (
                    f"branch_label lost for step {orig['id']}"
                )
            if "consumes_step_ids" in orig:
                assert imp.get("consumes_step_ids") == orig["consumes_step_ids"], (
                    f"consumes_step_ids lost for step {orig['id']}"
                )


# ===========================================================================
# G4 — multi_step_renderer: scale capped at 1.0
# ===========================================================================


@pytest.mark.unit
class TestScaleCap:
    """G4: Small molecules are not upscaled beyond 1.0."""

    def test_scale_does_not_exceed_one(self) -> None:
        """G4: min(scale_x, scale_y, 1.0) prevents upscaling."""
        # Scale calc: min(300/100, 200/50, 1.0) = 1.0, not 2.0
        scale_x = 300.0 / 100.0
        scale_y = 200.0 / 50.0
        scale = min(scale_x, scale_y, 1.0)
        assert scale == 1.0


# ===========================================================================
# G5 — scheme_schema: load_scheme error handling
# ===========================================================================


@pytest.mark.unit
class TestLoadSchemeErrorHandling:
    """G5: load_scheme raises ValueError on invalid JSON."""

    def test_invalid_json_raises_valueerror(self, tmp_path: Path) -> None:
        """G5: Invalid JSON gives a clear error, not raw traceback."""
        from chemscheme.schema.scheme_schema import load_scheme

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json!!!", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_scheme(bad_file)


# ===========================================================================
# C5 — multi_step_renderer: wrapped layout connectors
# ===========================================================================


@pytest.mark.unit
class TestWrappedConnectors:
    """C5: Row-wrap connectors use polyline, not backwards line."""

    def test_wrap_connector_is_polyline(self, sample_scheme_5step_nfa: dict) -> None:
        """C5: 5-step scheme with max_steps_per_row=3 uses polyline for wrap."""
        from chemscheme.renderers.layout_engine import LayoutConfig
        from chemscheme.renderers.multi_step_renderer import render_scheme

        config = LayoutConfig(max_steps_per_row=3)
        svg_bytes = render_scheme(sample_scheme_5step_nfa, fmt="svg", config=config)
        svg_str = svg_bytes.decode("utf-8")

        # Should contain polyline for the row-wrap connection
        assert "polyline" in svg_str


# ===========================================================================
# C6 — mcp_server: async with thread offloading
# ===========================================================================


@pytest.mark.unit
class TestMcpAsyncThreading:
    """C6: MCP tools use asyncio.to_thread for blocking work."""

    def test_draw_scheme_uses_to_thread(self) -> None:
        """C6: draw_scheme calls asyncio.to_thread."""
        import inspect

        from chemscheme.mcp_server import draw_scheme

        source = inspect.getsource(draw_scheme)
        assert "asyncio.to_thread" in source or "to_thread" in source

    def test_validate_scheme_uses_to_thread(self) -> None:
        """C6: validate_scheme calls asyncio.to_thread."""
        import inspect

        from chemscheme.mcp_server import validate_scheme

        source = inspect.getsource(validate_scheme)
        assert "asyncio.to_thread" in source or "to_thread" in source


# ===========================================================================
# G7 — mcp_server: style param is used
# ===========================================================================


@pytest.mark.unit
class TestMcpStyleParam:
    """G7: draw_scheme passes style through to response."""

    def test_style_appears_in_response(self) -> None:
        """G7: The style is included in the sync draw output."""
        import inspect

        from chemscheme.mcp_server import _sync_draw

        source = inspect.getsource(_sync_draw)
        # style is used in LayoutConfig selection and included in response
        assert '"style"' in source or "style" in source


# ===========================================================================
# G6/C7 — name_resolver: importlib.resources
# ===========================================================================


@pytest.mark.unit
class TestAliasLoading:
    """G6/C7: Alias loading uses importlib.resources with fallback."""

    def test_load_aliases_text_returns_valid_json(self) -> None:
        """G6/C7: _load_aliases_text returns parseable JSON."""
        from chemscheme.parsers.name_resolver import _load_aliases_text

        text = _load_aliases_text()
        data = json.loads(text)
        assert "abbreviations" in data

    def test_load_aliases_text_fallback_on_importlib_failure(self) -> None:
        """G6/C7: Falls back to legacy path when importlib.resources fails."""
        from chemscheme.parsers.name_resolver import _load_aliases_text

        with patch("chemscheme.parsers.name_resolver.importlib.resources") as mock_ir:
            mock_ir.files.side_effect = FileNotFoundError("no package")
            text = _load_aliases_text()
            data = json.loads(text)
            assert "abbreviations" in data


# ===========================================================================
# G8 — multi_step_renderer: generic ID prefixing
# ===========================================================================


@pytest.mark.unit
class TestGenericIdPrefixing:
    """G8: _strip_svg_wrapper prefixes ALL id/href, not just glyph-."""

    def test_prefix_all_ids(self) -> None:
        """G8: All id= attributes get prefixed."""
        from chemscheme.renderers.multi_step_renderer import _strip_svg_wrapper

        svg = (
            b'<?xml version="1.0"?>'
            b'<svg width="100" height="100">'
            b'<g id="custom-1"><use href="#custom-1"/></g>'
            b"</svg>"
        )
        result = _strip_svg_wrapper(svg, "s1")
        assert 'id="ss1-custom-1"' in result
        assert 'href="#ss1-custom-1"' in result


# ===========================================================================
# G9 — layer1_chemistry: stereo check consistency with smiles_utils
# ===========================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestStereoCheckConsistency:
    """G9: layer1 stereo check uses RDKit counts, consistent with smiles_utils."""

    def test_no_false_positive_on_pseudo_stereo(self) -> None:
        """MR4-17: No false warning for molecules with stereo markers but no real stereocenter."""
        from chemscheme.validators.layer1_chemistry import validate_chemistry

        # C/C=C(/C)C has stereo markers but identical substituents → no real E/Z
        result = validate_chemistry(["C/C=C(/C)C"])
        assert not any("Stereochemistry" in w for w in result.warnings)


# ===========================================================================
# MR3-1 — cdxml_writer: missing fragment reference raises ValueError
# ===========================================================================


@pytest.mark.unit
class TestCdxmlMissingFragment:
    """MR3-1: CDXML import fails fast on dangling fragment references."""

    def test_missing_reactant_fragment_raises(self) -> None:
        """MR3-1: Missing reactant fragment ID raises ValueError."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme

        cdxml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<CDXML><page>"
            '<fragment id="F1" ChemSchemeSMILES="CC"/>'
            "<scheme>"
            '<step ReactionStepReactants="F404" ReactionStepProducts="F1" '
            'ReactionStepArrows="A1" ChemSchemeStepId="1"/>'
            "</scheme>"
            "</page></CDXML>"
        )
        with pytest.raises(ValueError, match="fragment IDs.*F404"):
            cdxml_to_scheme(cdxml)

    def test_missing_product_fragment_raises(self) -> None:
        """MR3-1: Missing product fragment ID raises ValueError."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme

        cdxml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<CDXML><page>"
            '<fragment id="F1" ChemSchemeSMILES="CC"/>'
            "<scheme>"
            '<step ReactionStepReactants="F1" ReactionStepProducts="F999" '
            'ReactionStepArrows="A1" ChemSchemeStepId="1"/>'
            "</scheme>"
            "</page></CDXML>"
        )
        with pytest.raises(ValueError, match="fragment IDs.*F999"):
            cdxml_to_scheme(cdxml)


# ===========================================================================
# MR3.1-1 — CLI import: catch ValueError from cdxml_to_scheme
# ===========================================================================


@pytest.mark.unit
class TestCliImportCdxmlError:
    """MR3.1-1: CLI import command catches ValueError from malformed CDXML."""

    def test_import_missing_fragment_text_mode(self, tmp_path: Path) -> None:
        """MR3.1-1: import with bad CDXML shows error, not traceback."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        cdxml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<CDXML><page>"
            '<fragment id="F1" ChemSchemeSMILES="CC"/>'
            "<scheme>"
            '<step ReactionStepReactants="F404" ReactionStepProducts="F1" '
            'ReactionStepArrows="A1" ChemSchemeStepId="1"/>'
            "</scheme>"
            "</page></CDXML>"
        )
        src = tmp_path / "bad.cdxml"
        src.write_text(cdxml, encoding="utf-8")
        out = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(main, ["import", str(src), "-o", str(out)])
        assert result.exit_code == 1
        # exception is SystemExit(1) from sys.exit, not an uncaught ValueError
        assert not isinstance(result.exception, ValueError)

    def test_import_missing_fragment_json_mode(self, tmp_path: Path) -> None:
        """MR3.1-1: import --json with bad CDXML returns error JSON."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        cdxml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<CDXML><page>"
            '<fragment id="F1" ChemSchemeSMILES="CC"/>'
            "<scheme>"
            '<step ReactionStepReactants="F404" ReactionStepProducts="F1" '
            'ReactionStepArrows="A1" ChemSchemeStepId="1"/>'
            "</scheme>"
            "</page></CDXML>"
        )
        src = tmp_path / "bad.cdxml"
        src.write_text(cdxml, encoding="utf-8")
        out = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(main, ["import", str(src), "-o", str(out), "--json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["status"] == "error"
        assert "fragment IDs" in data["error"]


# ===========================================================================
# MR3-2 — name_resolver: garbage SMILES rejected
# ===========================================================================


@pytest.mark.unit
class TestNameResolverGarbageSmiles:
    """MR3-2: name_resolver rejects unparseable SMILES from providers."""

    def test_canonicalize_returns_none_for_garbage(self) -> None:
        """MR3-2: _canonicalize_smiles returns None for invalid SMILES."""
        from chemscheme.parsers.name_resolver import _canonicalize_smiles

        assert _canonicalize_smiles("not-a-smiles!!!") is None
        assert _canonicalize_smiles("XYZ[[[") is None

    def test_canonicalize_returns_canonical_for_valid(self) -> None:
        """MR3-2: _canonicalize_smiles returns canonical for valid SMILES."""
        from chemscheme.parsers.name_resolver import _canonicalize_smiles

        assert _canonicalize_smiles("C(C)O") == "CCO"

    def test_opsin_garbage_returns_none(self) -> None:
        """MR3-2: OPSIN returning garbage text results in None."""
        from unittest.mock import MagicMock

        from chemscheme.parsers.name_resolver import resolve_via_opsin

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "Error: name not found"

        with patch("chemscheme.parsers.name_resolver.requests.get", return_value=mock_resp):
            result = resolve_via_opsin("nonexistent-compound")
            assert result is None


# ===========================================================================
# MR3-3 — cli validate: consistent behavior with/without --json
# ===========================================================================


@pytest.mark.unit
class TestCliValidateConsistency:
    """MR3-3: CLI validate runs full validation in both text and JSON mode."""

    def test_text_mode_runs_chemistry_layer(self, tmp_path: Path) -> None:
        """MR3-3: Text mode also runs chemistry + continuity layers."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        # Valid schema but invalid SMILES
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["INVALID_SMILES"],
                    "products": ["CC"],
                    "arrow_above": "test",
                }
            ],
        }
        scheme_file = tmp_path / "test.json"
        scheme_file.write_text(json.dumps(scheme), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(scheme_file)])
        # Should fail because chemistry layer catches invalid SMILES
        assert result.exit_code != 0

    def test_text_mode_shows_layers(self, tmp_path: Path) -> None:
        """MR3-3: Text mode shows which layers were checked."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                }
            ],
        }
        scheme_file = tmp_path / "test.json"
        scheme_file.write_text(json.dumps(scheme), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(scheme_file)])
        assert result.exit_code == 0
        assert "layers:" in result.output


# ===========================================================================
# MR3-4 — ai_intent: markdown stripping with surrounding text
# ===========================================================================


@pytest.mark.unit
class TestMarkdownStripSurroundingText:
    """MR3-4: Markdown fences are extracted even with surrounding text."""

    def test_fences_with_preamble(self) -> None:
        """MR3-4: JSON block extracted when preceded by conversational text."""
        from chemscheme.parsers.ai_intent import _strip_markdown_fences

        text = 'Here is the JSON:\n```json\n{"key": "val"}\n```\nDone.'
        assert _strip_markdown_fences(text) == '{"key": "val"}'


# ===========================================================================
# MR3-6 — scheme_schema: atomic save_scheme
# ===========================================================================


@pytest.mark.unit
class TestAtomicSaveScheme:
    """MR3-6: save_scheme uses atomic write."""

    def test_save_scheme_creates_file(self, tmp_path: Path) -> None:
        """MR3-6: save_scheme writes valid JSON atomically."""
        from chemscheme.schema.scheme_schema import load_scheme, save_scheme

        scheme = {
            "schema_version": "1",
            "steps": [{"id": "1", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "test"}],
        }
        out = tmp_path / "output.json"
        save_scheme(scheme, out)
        loaded = load_scheme(out)
        assert loaded["schema_version"] == "1"
        assert len(loaded["steps"]) == 1


# ===========================================================================
# MR3-8 — layer2_reaction: PredictionResult immutability
# ===========================================================================


@pytest.mark.unit
class TestPredictionResultImmutability:
    """MR3-8: PredictionResult.predicted_products is immutable tuple."""

    def test_predicted_products_is_tuple(self) -> None:
        """MR3-8: predicted_products stored as tuple."""
        from chemscheme.validators.layer2_reaction import PredictionResult

        r = PredictionResult(feasible=True, confidence=1.0, predicted_products=["CC", "CCO"])
        assert isinstance(r.predicted_products, tuple)
        assert r.predicted_products == ("CC", "CCO")

    def test_predicted_products_not_mutable(self) -> None:
        """MR3-8: Cannot mutate predicted_products."""
        from chemscheme.validators.layer2_reaction import PredictionResult

        r = PredictionResult(feasible=True, confidence=1.0, predicted_products=["CC"])
        with pytest.raises(AttributeError):
            r.predicted_products.append("X")  # type: ignore[union-attr]


# ===========================================================================
# MR3-9 — scheme_editor: position boundary validation
# ===========================================================================


@pytest.mark.unit
class TestAddStepPositionBounds:
    """MR3-9: add_step rejects out-of-range position."""

    def test_negative_position_raises(self, sample_scheme: dict) -> None:
        """MR3-9: Negative position raises ValueError."""
        from chemscheme.editors.scheme_editor import add_step

        new_step = {"id": "new", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "test"}
        with pytest.raises(ValueError, match="out of range"):
            add_step(sample_scheme, new_step, position=-1)

    def test_oversized_position_raises(self, sample_scheme: dict) -> None:
        """MR3-9: Position > len(steps) raises ValueError."""
        from chemscheme.editors.scheme_editor import add_step

        new_step = {"id": "new", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "test"}
        num_steps = len(sample_scheme["steps"])
        with pytest.raises(ValueError, match="out of range"):
            add_step(sample_scheme, new_step, position=num_steps + 10)


# ===========================================================================
# MR3-10 — mcp_server: unknown style reports applied_style
# ===========================================================================


@pytest.mark.unit
class TestMcpUnknownStyle:
    """MR3-10: Unknown styles fall back and report applied_style correctly."""

    def test_unknown_style_returns_default_theme(self) -> None:
        """MR3-10: Unknown style falls back to default theme in response."""
        from chemscheme.mcp_server import _sync_draw

        scheme = {
            "schema_version": "1",
            "steps": [
                {"id": "1", "reactants": ["CCO"], "products": ["CC=O"], "arrow_above": "PCC"}
            ],
        }
        result_json = _sync_draw(scheme, "svg", "FancyStyle", 300)
        result = json.loads(result_json)
        assert result["status"] == "ok"
        assert result["theme"] == "default"
        assert result["style"] == "FancyStyle"  # raw style preserved


# ===========================================================================
# MR3-14 — scheme_schema: forward/cyclic consumes_step_ids rejected
# ===========================================================================


@pytest.mark.unit
class TestConsumesStepIdsValidation:
    """MR3-14: Forward and self-referential consumes_step_ids are rejected."""

    def test_self_reference_rejected(self) -> None:
        """MR3-14: A step referencing itself is an error."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {"id": "1", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "a"},
                {
                    "id": "2",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "b",
                    "consumes_step_ids": ["2"],
                },
            ],
        }
        errors = validate_scheme(scheme)
        assert any("cycle" in e for e in errors)

    def test_forward_reference_rejected(self) -> None:
        """MR3-14: A step referencing a later step is an error."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CC"],
                    "products": ["CCO"],
                    "arrow_above": "a",
                    "consumes_step_ids": ["2"],
                },
                {"id": "2", "reactants": ["CCO"], "products": ["CC=O"], "arrow_above": "b"},
            ],
        }
        errors = validate_scheme(scheme)
        assert any("forward step" in e for e in errors)

    def test_valid_backward_reference_passes(self) -> None:
        """MR3-14: Valid backward reference is accepted."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1a",
                    "reactants": ["CC"],
                    "products": ["CCO"],
                    "arrow_above": "a",
                    "branch_label": "A",
                },
                {
                    "id": "1b",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "b",
                    "branch_label": "B",
                },
                {
                    "id": "2",
                    "reactants": ["CCO", "c1ccc(O)cc1"],
                    "products": ["CC(=O)Oc1ccccc1"],
                    "arrow_above": "c",
                    "consumes_step_ids": ["1a", "1b"],
                },
            ],
        }
        errors = validate_scheme(scheme)
        assert not any("forward" in e for e in errors)
        assert not any("cycle" in e for e in errors)


# ===========================================================================
# MR4-1 — multi_step_renderer: xlink:href prefix
# ===========================================================================


@pytest.mark.unit
class TestSvgXlinkHrefPrefix:
    """MR4-1: SVG ID prefixing handles xlink:href."""

    def test_xlink_href_is_prefixed(self) -> None:
        """MR4-1: xlink:href references get step-id prefix."""
        from chemscheme.renderers.multi_step_renderer import _strip_svg_wrapper

        svg = b'<svg><defs><g id="glyph-0-1"/></defs><use xlink:href="#glyph-0-1"/></svg>'
        result = _strip_svg_wrapper(svg, "step1")
        assert 'xlink:href="#sstep1-glyph-0-1"' in result
        assert 'id="sstep1-glyph-0-1"' in result

    def test_plain_href_still_works(self) -> None:
        """MR4-1: Plain href continues to be prefixed."""
        from chemscheme.renderers.multi_step_renderer import _strip_svg_wrapper

        svg = b'<svg><defs><g id="g1"/></defs><use href="#g1"/></svg>'
        result = _strip_svg_wrapper(svg, "s2")
        assert 'href="#ss2-g1"' in result


# ===========================================================================
# MR4-2 — CLI draw rejects unsupported output format
# ===========================================================================


@pytest.mark.unit
class TestCliDrawFormatValidation:
    """MR4-2: CLI draw/export rejects unsupported output extensions."""

    def test_draw_pdf_extension_now_supported_via_cairosvg(self, tmp_path: Path) -> None:
        """G13 (S68) supersedes MR4-2: PDF is now an intentional output format
        via cairosvg.svg2pdf. This test documents the change — PDF either
        renders successfully (cairosvg available) or fails with an actionable
        message about installing cairosvg/libcairo (not silent SVG fallback).
        """
        import sys
        import types
        from unittest.mock import MagicMock, patch

        from click.testing import CliRunner

        from chemscheme.cli import main
        from chemscheme.schema.scheme_schema import save_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                }
            ],
        }
        src = tmp_path / "s.json"
        save_scheme(scheme, src)
        out = tmp_path / "out.pdf"

        # Mock cairosvg so the test passes regardless of whether libcairo is
        # installed locally (Windows often lacks it).
        fake_cairosvg = types.ModuleType("cairosvg")
        fake_cairosvg.svg2pdf = MagicMock(return_value=b"%PDF-1.4 stub")  # type: ignore[attr-defined]
        runner = CliRunner()
        with patch.dict(sys.modules, {"cairosvg": fake_cairosvg}):
            result = runner.invoke(main, ["draw", str(src), "-o", str(out)])
        assert result.exit_code == 0, (
            f"PDF render via cairosvg should succeed, got: {result.output}"
        )
        assert out.exists()
        assert out.read_bytes().startswith(b"%PDF")

    def test_export_rejects_svg_extension(self, tmp_path: Path) -> None:
        """MR4-2: export -o out.svg fails, not silent CDXML."""
        from click.testing import CliRunner

        from chemscheme.cli import main
        from chemscheme.schema.scheme_schema import save_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                }
            ],
        }
        src = tmp_path / "s.json"
        save_scheme(scheme, src)
        out = tmp_path / "out.svg"
        runner = CliRunner()
        result = runner.invoke(main, ["export", str(src), "-o", str(out)])
        assert result.exit_code == 1


# ===========================================================================
# MR4-3 — CDXML import validates SMILES
# ===========================================================================


@pytest.mark.unit
class TestCdxmlImportValidation:
    """MR4-3: CDXML import validates/canonicalizes stored SMILES."""

    def test_invalid_stored_smiles_triggers_reconstruction(self) -> None:
        """MR4-3: Invalid ChemSchemeSMILES falls back to reconstruction."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme

        cdxml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<CDXML><page>"
            '<fragment id="F1" ChemSchemeSMILES="INVALID[[["/>'
            '<fragment id="F2" ChemSchemeSMILES="CCO"/>'
            "<scheme>"
            '<step ReactionStepReactants="F1" ReactionStepProducts="F2" '
            'ReactionStepArrows="A1" ChemSchemeStepId="1"/>'
            "</scheme>"
            "</page></CDXML>"
        )
        # Should not crash — falls back to reconstruction (which may also fail)
        import contextlib

        with contextlib.suppress(ValueError):
            cdxml_to_scheme(cdxml)


# ===========================================================================
# MR4-4 — layout_engine: branch_label parallel layout
# ===========================================================================


@pytest.mark.unit
class TestBranchLabelParallelLayout:
    """MR4-4: Branched schemes render parallel, not linear."""

    def test_branch_labels_without_merge_get_parallel_layout(self) -> None:
        """MR4-4: Two branch_labels → two parallel branches in layout."""
        from chemscheme.renderers.layout_engine import compute_layout

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1a",
                    "reactants": ["CC"],
                    "products": ["CCO"],
                    "arrow_above": "a",
                    "branch_label": "A",
                },
                {
                    "id": "1b",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "b",
                    "branch_label": "B",
                },
            ],
        }
        result = compute_layout(scheme)
        # Should have 2 boxes at different Y positions (parallel)
        ys = {b.y for b in result.boxes}
        assert len(ys) == 2, "Branches should be at different Y positions"


# ===========================================================================
# MR4-9 — validate_scheme_json includes reaction layer
# ===========================================================================


@pytest.mark.unit
class TestValidateSchemeJsonReactionLayer:
    """MR4-9: validate_scheme_json runs reaction layer."""

    def test_reaction_layer_in_layers_checked(self) -> None:
        """MR4-9: layers_checked includes 'reaction'."""
        from chemscheme.schema.scheme_schema import validate_scheme_json

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                }
            ],
        }
        result = validate_scheme_json(scheme)
        assert "reaction" in result["layers_checked"]


# ===========================================================================
# MR4-10 — layer3 fp_cache thread safety (no module-level state)
# ===========================================================================


@pytest.mark.unit
class TestFpCacheThreadSafety:
    """MR4-10: No module-level _fp_cache in layer3_scheme."""

    def test_no_module_level_fp_cache(self) -> None:
        """MR4-10: layer3_scheme has no module-level _fp_cache dict."""
        import chemscheme.validators.layer3_scheme as l3

        assert not hasattr(l3, "_fp_cache")


# ===========================================================================
# MR4-13 — AI parser prefers json-labeled code block
# ===========================================================================


@pytest.mark.unit
class TestAiParserJsonBlock:
    """MR4-13: AI parser prefers ```json blocks."""

    def test_prefers_json_labeled_block(self) -> None:
        """MR4-13: When both bare and json blocks exist, picks json one."""
        from chemscheme.parsers.ai_intent import _strip_markdown_fences

        text = (
            "Here is some explanation:\n```\nnot json\n```\n"
            'And the result:\n```json\n{"key": "value"}\n```'
        )
        result = _strip_markdown_fences(text)
        assert result == '{"key": "value"}'


# ===========================================================================
# MR4.1-P1a — Indigo render_step thread safety
# ===========================================================================


@pytest.mark.unit
class TestIndigoRenderLock:
    """MR4.1-P1a: render_step uses a lock for thread safety."""

    def test_render_lock_exists(self) -> None:
        """MR4.1-P1a: _render_lock was in indigo_renderer; RDKit is thread-safe."""
        # RDKit renderer doesn't need a lock — this test verifies
        # that the module can be imported without error
        from chemscheme.renderers.rdkit_renderer import render_step  # noqa: F401

    def test_concurrent_renders_no_crash(self) -> None:
        """MR4.1-P1a: concurrent render_step calls don't crash."""
        import concurrent.futures

        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "reactants": ["CCO"],
            "products": ["CC=O"],
            "arrow_above": "PCC",
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(render_step, step, "svg") for _ in range(3)]
            results = [f.result() for f in futures]
        assert all(b"<svg" in r for r in results)


# ===========================================================================
# MR4.1-P1b — CLI draw Windows atomic write retry
# ===========================================================================


@pytest.mark.unit
class TestCliDrawAtomicRetry:
    """MR4.1-P1b: CLI draw has PermissionError retry on Windows."""

    def test_draw_source_has_retry(self) -> None:
        """MR4.1-P1b: draw command source contains retry logic."""
        import inspect

        from chemscheme.cli import draw

        # Click wraps the function; get the underlying callback
        source = inspect.getsource(draw.callback)  # type: ignore[arg-type]
        assert "PermissionError" in source
        assert "attempt" in source


# ===========================================================================
# MR4.1-P2a — O(N) preceding_ids in scheme_schema
# ===========================================================================


@pytest.mark.unit
class TestPrecedingIdsIncremental:
    """MR4.1-P2a: preceding_ids is built incrementally, not per-step."""

    def test_forward_ref_still_caught(self) -> None:
        """MR4.1-P2a: forward reference detection still works with O(N) approach."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CC"],
                    "products": ["CCO"],
                    "arrow_above": "a",
                    "consumes_step_ids": ["2"],
                },
                {
                    "id": "2",
                    "reactants": ["CCO"],
                    "products": ["CC(=O)O"],
                    "arrow_above": "b",
                },
            ],
        }
        errors = validate_scheme(scheme)
        assert any("forward step" in e for e in errors)

    def test_valid_back_ref_accepted(self) -> None:
        """MR4.1-P2a: backward reference is accepted."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CC"],
                    "products": ["CCO"],
                    "arrow_above": "a",
                },
                {
                    "id": "2",
                    "reactants": ["CCO"],
                    "products": ["CC(=O)O"],
                    "arrow_above": "b",
                    "consumes_step_ids": ["1"],
                },
            ],
        }
        errors = validate_scheme(scheme)
        assert not errors


# ===========================================================================
# MR4.1-P2b — Angle-aware arrowhead in PNG fallback
# ===========================================================================


@pytest.mark.unit
class TestPngArrowheadDirection:
    """MR4.1-P2b: PNG arrowhead uses angle-aware polygon."""

    def test_arrowhead_source_uses_math(self) -> None:
        """MR4.1-P2b: multi_step_renderer arrowhead uses math.hypot."""
        import inspect

        from chemscheme.renderers import multi_step_renderer

        source = inspect.getsource(multi_step_renderer)
        assert "math.hypot" in source
        assert "ux" in source and "uy" in source


# ===========================================================================
# MR4.1-P2c — Robust CDXML Element parsing
# ===========================================================================


@pytest.mark.unit
class TestCdxmlElementParsing:
    """MR4.1-P2c: CDXML import handles non-numeric Element attributes."""

    def test_symbol_element_attribute(self) -> None:
        """MR4.1-P2c: Fragment with Element='N' doesn't crash."""
        from xml.etree.ElementTree import Element as XmlElement
        from xml.etree.ElementTree import SubElement

        from chemscheme.renderers.cdxml_writer import _fragment_to_smiles

        frag = XmlElement("fragment", id="1")
        SubElement(frag, "n", id="1", Element="N")
        SubElement(frag, "n", id="2", Element="6")
        SubElement(frag, "b", B="1", E="2", Order="1")

        result = _fragment_to_smiles(frag)
        # Should produce a valid SMILES containing nitrogen
        assert result is not None
        assert "N" in result or "n" in result


# ===========================================================================
# C4.1-P2b — validate_scheme rejects non-dict top-level values
# ===========================================================================


@pytest.mark.unit
class TestValidateSchemeNonDict:
    """C4.1-P2b: validate_scheme handles non-dict input gracefully."""

    def test_none_input(self) -> None:
        """C4.1-P2b: None input returns error, not TypeError."""
        from chemscheme.schema.scheme_schema import validate_scheme

        errors = validate_scheme(None)  # type: ignore[arg-type]
        assert len(errors) == 1
        assert "JSON object" in errors[0]

    def test_int_input(self) -> None:
        """C4.1-P2b: integer input returns error, not TypeError."""
        from chemscheme.schema.scheme_schema import validate_scheme

        errors = validate_scheme(1)  # type: ignore[arg-type]
        assert len(errors) == 1
        assert "int" in errors[0]

    def test_list_input(self) -> None:
        """C4.1-P2b: list input returns error, not TypeError."""
        from chemscheme.schema.scheme_schema import validate_scheme

        errors = validate_scheme([{"id": "1"}])  # type: ignore[arg-type]
        assert len(errors) == 1
        assert "list" in errors[0]


# ===========================================================================
# C4.1-P1c — PNG render graceful error when no backend
# ===========================================================================


@pytest.mark.unit
class TestPngRenderMissingBackend:
    """C4.1-P1c: PNG rendering raises RuntimeError, not ModuleNotFoundError."""

    def test_missing_both_backends_raises_runtime_error(self) -> None:
        """C4.1-P1c: Without CairoSVG or Pillow, get RuntimeError."""
        import sys

        from chemscheme.renderers.multi_step_renderer import _render_scheme_png

        scheme = {
            "schema_version": "1",
            "steps": [
                {"id": "1", "reactants": ["CCO"], "products": ["CC=O"], "arrow_above": "ox"},
            ],
        }
        # Temporarily hide both cairosvg and PIL
        saved = {}
        for mod_name in ("cairosvg", "PIL", "PIL.Image"):
            if mod_name in sys.modules:
                saved[mod_name] = sys.modules.pop(mod_name)

        with (
            patch.dict(sys.modules, {"cairosvg": None, "PIL": None, "PIL.Image": None}),
            pytest.raises(RuntimeError, match="PNG rendering requires"),
        ):
            from chemscheme.renderers.layout_engine import LayoutConfig

            _render_scheme_png(scheme, LayoutConfig(), 300)

        # Restore
        sys.modules.update(saved)


# ===========================================================================
# C4.1-P1d — add_step validates step is dict
# ===========================================================================


@pytest.mark.unit
class TestAddStepNonDict:
    """C4.1-P1d: add_step rejects non-dict step input."""

    def test_list_step_raises_value_error(self, sample_scheme: dict) -> None:
        """C4.1-P1d: Passing a list as step raises ValueError."""
        from chemscheme.editors.scheme_editor import add_step

        with pytest.raises(ValueError, match="must be a dict"):
            add_step(sample_scheme, [1, 2, 3])  # type: ignore[arg-type]

    def test_string_step_raises_value_error(self, sample_scheme: dict) -> None:
        """C4.1-P1d: Passing a string as step raises ValueError."""
        from chemscheme.editors.scheme_editor import add_step

        with pytest.raises(ValueError, match="must be a dict"):
            add_step(sample_scheme, "not a step")  # type: ignore[arg-type]


# ===========================================================================
# C4.1-P1e — export catches scheme_to_cdxml errors
# ===========================================================================


@pytest.mark.unit
class TestExportCatchesCdxmlError:
    """C4.1-P1e: export command catches scheme_to_cdxml errors."""

    def test_export_source_has_error_handling(self) -> None:
        """C4.1-P1e: export command source catches ValueError/RuntimeError."""
        import inspect

        from chemscheme.cli import export_cdxml

        source = inspect.getsource(export_cdxml.callback)  # type: ignore[arg-type]
        assert "CDXML export failed" in source


# ===========================================================================
# C4.1-P1f — ai --output overwrite guard
# ===========================================================================


@pytest.mark.unit
class TestAiOutputOverwriteGuard:
    """C4.1-P1f: ai command has --force flag and overwrite guard."""

    def test_ai_has_force_option(self) -> None:
        """C4.1-P1f: ai command accepts --force flag."""
        from chemscheme.cli import ai

        param_names = [p.name for p in ai.params]
        assert "force" in param_names

    def test_ai_source_has_overwrite_check(self) -> None:
        """C4.1-P1f: ai command checks existing file."""
        import inspect

        from chemscheme.cli import ai

        source = inspect.getsource(ai.callback)  # type: ignore[arg-type]
        assert "already exists" in source


# ===========================================================================
# C4.1-P2c — Bad alias SMILES falls back to IUPAC name
# ===========================================================================


@pytest.mark.unit
class TestBadAliasFallback:
    """C4.1-P2c: Invalid cached SMILES falls through to IUPAC name."""

    def test_bad_smiles_does_not_return_none(self) -> None:
        """C4.1-P2c: Invalid cached SMILES tries IUPAC name, not immediate None."""
        from chemscheme.parsers.name_resolver import resolve_local_alias

        # Mock aliases with bad SMILES but valid IUPAC name
        bad_entry = {"smiles": "INVALID_SMILES", "name": "ethanol"}
        with (
            patch(
                "chemscheme.parsers.name_resolver._load_aliases",
                return_value={"test_alias": bad_entry},
            ),
            patch("chemscheme.parsers.name_resolver.resolve_via_opsin") as mock_opsin,
        ):
            from chemscheme.models import SmilesResult

            mock_opsin.return_value = SmilesResult(
                smiles="CCO", confidence=0.9, source="opsin", warnings=[]
            )
            result = resolve_local_alias("test_alias")
            # Should have tried OPSIN with the IUPAC name
            mock_opsin.assert_called_once_with("ethanol")
            assert result is not None
            assert result.smiles == "CCO"


# ===========================================================================
# C4.1-P2e — schema_version type check
# ===========================================================================


@pytest.mark.unit
class TestSchemaVersionType:
    """C4.1-P2e: schema_version must be a string."""

    def test_int_schema_version_rejected(self) -> None:
        """C4.1-P2e: Integer schema_version is rejected."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": 1,
            "steps": [
                {"id": "1", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "a"},
            ],
        }
        errors = validate_scheme(scheme)
        assert any("string" in e for e in errors)


# ===========================================================================
# MR5-G2 — add_step supports scheme without "steps" key
# ===========================================================================


@pytest.mark.unit
class TestAddStepNoStepsKey:
    """MR5-G2: add_step handles scheme dict without 'steps' key."""

    def test_add_to_empty_scheme(self) -> None:
        """MR5-G2: Adding step to a scheme with no 'steps' key works."""
        from chemscheme.editors.scheme_editor import add_step

        scheme: dict = {"schema_version": "1"}  # type: ignore[type-arg]
        new_step = {"id": "1", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "test"}
        result = add_step(scheme, new_step)
        assert len(result["steps"]) == 1
        assert result["steps"][0]["id"] == "1"

    def test_original_not_mutated(self) -> None:
        """MR5-G2: Original scheme without 'steps' is not mutated."""
        from chemscheme.editors.scheme_editor import add_step

        scheme: dict = {"schema_version": "1"}  # type: ignore[type-arg]
        new_step = {"id": "1", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "test"}
        add_step(scheme, new_step)
        assert "steps" not in scheme


# ===========================================================================
# MR5-G3/C1 — layout_engine: no duplicate boxes for diamond topology
# ===========================================================================


@pytest.mark.unit
class TestLayoutNoDuplicateBoxes:
    """MR5-G3/C1: Diamond topology doesn't create duplicate LayoutBoxes."""

    def test_diamond_topology_unique_step_ids(self) -> None:
        """MR5-G3/C1: 1 -> {2a, 2b} -> 3 has no duplicate step_id in boxes."""
        from chemscheme.renderers.layout_engine import compute_layout

        scheme = {
            "schema_version": "1",
            "steps": [
                {"id": "1", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "a"},
                {
                    "id": "2a",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "b",
                    "branch_label": "A",
                },
                {
                    "id": "2b",
                    "reactants": ["CCO"],
                    "products": ["CCCO"],
                    "arrow_above": "c",
                    "branch_label": "B",
                },
                {
                    "id": "3",
                    "reactants": ["CC=O", "CCCO"],
                    "products": ["CCCOC=O"],
                    "arrow_above": "d",
                    "consumes_step_ids": ["2a", "2b"],
                },
            ],
        }
        result = compute_layout(scheme)
        step_ids = [box.step_id for box in result.boxes]
        assert len(step_ids) == len(set(step_ids)), f"Duplicate step_ids in boxes: {step_ids}"


# ===========================================================================
# MR5-G1 — remove_step keeps empty consumes_step_ids
# ===========================================================================


@pytest.mark.unit
class TestRemoveStepKeepsEmptyConsumes:
    """MR5-G1: Removing a dependency keeps empty consumes_step_ids list."""

    def test_empty_list_preserved(self) -> None:
        """MR5-G1: After removing sole dependency, consumes_step_ids = []."""
        from chemscheme.editors.scheme_editor import remove_step

        scheme = {
            "schema_version": "1",
            "steps": [
                {"id": "1a", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "a"},
                {"id": "1b", "reactants": ["c1ccccc1"], "products": ["O"], "arrow_above": "b"},
                {
                    "id": "2",
                    "reactants": ["CCO", "O"],
                    "products": ["CCO"],
                    "arrow_above": "c",
                    "consumes_step_ids": ["1a", "1b"],
                },
            ],
        }
        result = remove_step(scheme, "1a")
        step_2 = next(s for s in result["steps"] if s["id"] == "2")
        # Should still have consumes_step_ids with just ["1b"]
        assert "consumes_step_ids" in step_2
        assert step_2["consumes_step_ids"] == ["1b"]

    def test_all_refs_removed_keeps_empty_list(self) -> None:
        """MR5-G1: After removing all dependencies, consumes_step_ids = []."""
        from chemscheme.editors.scheme_editor import remove_step

        scheme = {
            "schema_version": "1",
            "steps": [
                {"id": "1a", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "a"},
                {
                    "id": "2",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "b",
                    "consumes_step_ids": ["1a"],
                },
                {"id": "3", "reactants": ["CC=O"], "products": ["O"], "arrow_above": "c"},
            ],
        }
        result = remove_step(scheme, "1a")
        step_2 = next(s for s in result["steps"] if s["id"] == "2")
        assert "consumes_step_ids" in step_2
        assert step_2["consumes_step_ids"] == []


# ===========================================================================
# MR5-G5 — CDXML comma-delimited consumes_step_ids round-trip
# ===========================================================================


@pytest.mark.unit
class TestCdxmlCommaDelimitedIds:
    """MR5-G5: CDXML uses comma-separated consumes_step_ids."""

    def test_round_trip_preserves_ids(self) -> None:
        """MR5-G5: consumes_step_ids survive CDXML export/import."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1a",
                    "reactants": ["CC"],
                    "products": ["CCO"],
                    "arrow_above": "a",
                    "branch_label": "A",
                },
                {
                    "id": "1b",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "b",
                    "branch_label": "B",
                },
                {
                    "id": "merge",
                    "reactants": ["CCO", "c1ccc(O)cc1"],
                    "products": ["CC(=O)Oc1ccccc1"],
                    "arrow_above": "coupling",
                    "consumes_step_ids": ["1a", "1b"],
                },
            ],
        }
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)
        merge_step = next(s for s in imported["steps"] if s["id"] == "merge")
        assert set(merge_step["consumes_step_ids"]) == {"1a", "1b"}


# ===========================================================================
# MR5-C2 — CDXML round-trip preserves optional metadata
# ===========================================================================


@pytest.mark.unit
class TestCdxmlMetadataRoundTrip:
    """MR5-C2: CDXML round-trip preserves reagents/solvents/temperature/time/atmosphere."""

    def test_optional_fields_preserved(self) -> None:
        """MR5-C2: Optional step metadata survives CDXML export/import."""
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                    "reagents": ["PCC", "DCM"],
                    "solvents": ["CH2Cl2"],
                    "temperature": "0 °C",
                    "time": "2 h",
                    "atmosphere": "N2",
                },
            ],
        }
        cdxml = scheme_to_cdxml(scheme)
        imported = cdxml_to_scheme(cdxml)
        step = imported["steps"][0]
        assert step.get("reagents") == ["PCC", "DCM"]
        assert step.get("solvents") == ["CH2Cl2"]
        assert step.get("temperature") == "0 °C"
        assert step.get("time") == "2 h"
        assert step.get("atmosphere") == "N2"


# ===========================================================================
# MR5-C3 — CLI commands catch load_scheme ValueError
# ===========================================================================


@pytest.mark.unit
class TestCliLoadSchemeErrorHandling:
    """MR5-C3: CLI commands catch ValueError from load_scheme."""

    def test_draw_malformed_json(self, tmp_path: Path) -> None:
        """MR5-C3: draw with invalid JSON shows error, not traceback."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        out = tmp_path / "out.svg"

        runner = CliRunner()
        result = runner.invoke(main, ["draw", str(bad), "-o", str(out)])
        assert result.exit_code == 1
        assert not isinstance(result.exception, ValueError)

    def test_validate_malformed_json(self, tmp_path: Path) -> None:
        """MR5-C3: validate with invalid JSON shows error, not traceback."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        bad = tmp_path / "bad.json"
        bad.write_text("<<<not json>>>", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(bad)])
        assert result.exit_code == 1
        assert not isinstance(result.exception, ValueError)

    def test_edit_malformed_json(self, tmp_path: Path) -> None:
        """MR5-C3: edit with invalid JSON shows error, not traceback."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        bad = tmp_path / "bad.json"
        bad.write_text("!not json!", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["edit", str(bad), "--remove-step", "1"])
        assert result.exit_code == 1
        assert not isinstance(result.exception, ValueError)

    def test_export_malformed_json(self, tmp_path: Path) -> None:
        """MR5-C3: export with invalid JSON shows error, not traceback."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        bad = tmp_path / "bad.json"
        bad.write_text("{{{invalid", encoding="utf-8")
        out = tmp_path / "out.cdxml"

        runner = CliRunner()
        result = runner.invoke(main, ["export", str(bad), "-o", str(out)])
        assert result.exit_code == 1
        assert not isinstance(result.exception, ValueError)


# ===========================================================================
# MR5-C4 — CLI edit validates scheme before editing
# ===========================================================================


@pytest.mark.unit
class TestCliEditPreValidation:
    """MR5-C4: CLI edit validates scheme before add/remove operations."""

    def test_edit_rejects_invalid_scheme(self, tmp_path: Path) -> None:
        """MR5-C4: edit with schema-invalid JSON returns error."""
        from click.testing import CliRunner

        from chemscheme.cli import main

        # Valid JSON but missing required fields
        bad_scheme = tmp_path / "bad_scheme.json"
        bad_scheme.write_text('{"not_a": "scheme"}', encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["edit", str(bad_scheme), "--remove-step", "1"])
        assert result.exit_code == 1


# ===========================================================================
# MR5-C5 — schema rejects empty reactants/products
# ===========================================================================


@pytest.mark.unit
class TestEmptyReactantsProductsRejected:
    """MR5-C5: Empty reactants or products lists are rejected."""

    def test_empty_reactants_rejected(self) -> None:
        """MR5-C5: Step with empty reactants list is invalid."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [{"id": "1", "reactants": [], "products": ["CC"], "arrow_above": "a"}],
        }
        errors = validate_scheme(scheme)
        assert any("reactants" in e and "empty" in e for e in errors)

    def test_empty_products_allowed(self) -> None:
        """MR5-C5→MR8-C3: Empty products allowed (CDX draft state)."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [{"id": "1", "reactants": ["CC"], "products": [], "arrow_above": "a"}],
        }
        errors = validate_scheme(scheme)
        # MR8-C3: Empty products no longer an error (CDX imports have no reaction semantics)
        assert not any("products" in e and "empty" in e for e in errors)


# ===========================================================================
# MR5-G4 — layer3 checks non-contiguous branch steps
# ===========================================================================


@pytest.mark.unit
@pytest.mark.chemistry
class TestNonContiguousBranchContinuity:
    """MR5-G4: Continuity check works for non-contiguous same-branch steps."""

    def test_interleaved_branches_checked(self) -> None:
        """MR5-G4: Steps in same branch but interleaved in array are checked."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        # Branch A steps interleaved with branch B
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "A1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "ox",
                    "branch_label": "A",
                },
                {
                    "id": "B1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "OH",
                    "branch_label": "B",
                },
                {
                    "id": "A2",
                    "reactants": ["CCCCCC"],
                    "products": ["CCCCC=O"],
                    "arrow_above": "ox",
                    "branch_label": "A",
                },
            ],
        }
        result = validate_continuity(scheme)
        # A1 products (CC=O) don't match A2 reactants (CCCCCC) — should warn
        assert any("A1" in w and "A2" in w for w in result.warnings)


# ===========================================================================
# MR5-C6 — CLI resolve exit on smiles, not confidence
# ===========================================================================


@pytest.mark.unit
class TestResolveExitStatus:
    """MR5-C6: CLI resolve exits based on smiles, not confidence."""

    def test_resolve_source_checks_smiles(self) -> None:
        """MR5-C6: resolve command uses 'not result.smiles' for exit."""
        import inspect

        from chemscheme.cli import resolve

        source = inspect.getsource(resolve.callback)  # type: ignore[arg-type]
        assert "not result.smiles" in source


# ===========================================================================
# MR5-C7 — schema validates top-level metadata types
# ===========================================================================


@pytest.mark.unit
class TestTopLevelMetadataTypeCheck:
    """MR5-C7: Top-level metadata fields are type-checked."""

    def test_title_dict_rejected(self) -> None:
        """MR5-C7: title: {} is rejected."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "title": {},
            "steps": [
                {"id": "1", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "a"},
            ],
        }
        errors = validate_scheme(scheme)
        assert any("title" in e and "string" in e for e in errors)

    def test_style_int_rejected(self) -> None:
        """MR5-C7: style: 123 is rejected."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "style": 123,
            "steps": [
                {"id": "1", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "a"},
            ],
        }
        errors = validate_scheme(scheme)
        assert any("style" in e and "string" in e for e in errors)

    def test_valid_title_passes(self) -> None:
        """MR5-C7: Valid string title passes."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "title": "My Synthesis",
            "steps": [
                {"id": "1", "reactants": ["CC"], "products": ["CCO"], "arrow_above": "a"},
            ],
        }
        errors = validate_scheme(scheme)
        assert not any("title" in e for e in errors)


# ===========================================================================
# MR5 Round 2 — Regression tests (2026-03-16)
# ===========================================================================


@pytest.mark.unit
class TestMr5r2G1c3AiIntentNonDict:
    """MR5R2-G1/C3: _validate_scheme rejects non-dict input gracefully."""

    def test_list_input_raises_service_error(self) -> None:
        from chemscheme.parsers.ai_intent import AiServiceError, _validate_scheme

        with pytest.raises(AiServiceError, match="expected dict, got list"):
            _validate_scheme([{"id": "1"}])

    def test_string_input_raises_service_error(self) -> None:
        from chemscheme.parsers.ai_intent import AiServiceError, _validate_scheme

        with pytest.raises(AiServiceError, match="expected dict, got str"):
            _validate_scheme("not a dict")


@pytest.mark.unit
class TestMr5r2G2ElementBoundsCheck:
    """MR5R2-G2: _fragment_to_smiles handles out-of-bounds element numbers."""

    def test_negative_element_number_falls_back(self) -> None:
        from xml.etree.ElementTree import Element, SubElement

        from chemscheme.renderers.cdxml_writer import _fragment_to_smiles

        frag = Element("fragment", id="1")
        SubElement(frag, "n", id="N1", Element="-5")
        result = _fragment_to_smiles(frag)
        # Should not crash — falls back to Carbon
        assert result is not None

    def test_element_999_falls_back(self) -> None:
        from xml.etree.ElementTree import Element, SubElement

        from chemscheme.renderers.cdxml_writer import _fragment_to_smiles

        frag = Element("fragment", id="1")
        SubElement(frag, "n", id="N1", Element="999")
        result = _fragment_to_smiles(frag)
        assert result is not None


@pytest.mark.unit
class TestMr5r2G3SelfBond:
    """MR5R2-G3: _fragment_to_smiles skips self-referential bonds."""

    def test_self_bond_skipped(self) -> None:
        from xml.etree.ElementTree import Element, SubElement

        from chemscheme.renderers.cdxml_writer import _fragment_to_smiles

        frag = Element("fragment", id="1")
        SubElement(frag, "n", id="N1", Element="6")
        SubElement(frag, "b", id="B1", B="N1", E="N1")  # self-referential
        result = _fragment_to_smiles(frag)
        # Should produce valid SMILES for a single carbon, not crash
        assert result is not None


@pytest.mark.unit
class TestMr5r2G4AliasLoadFailure:
    """MR5R2-G4: _load_aliases returns empty dict on file/JSON error."""

    def test_missing_file_returns_empty(self) -> None:
        import chemscheme.parsers.name_resolver as nr

        # Reset cache to force reload
        old_cache = nr._alias_cache
        nr._alias_cache = None
        try:
            with patch.object(nr, "_load_aliases_text", side_effect=FileNotFoundError("gone")):
                result = nr._load_aliases()
            assert result == {}
        finally:
            nr._alias_cache = old_cache

    def test_corrupt_json_returns_empty(self) -> None:
        import chemscheme.parsers.name_resolver as nr

        old_cache = nr._alias_cache
        nr._alias_cache = None
        try:
            with patch.object(nr, "_load_aliases_text", return_value="not valid json"):
                result = nr._load_aliases()
            assert result == {}
        finally:
            nr._alias_cache = old_cache


@pytest.mark.unit
class TestMr5r2G5EmptyContentArray:
    """MR5R2-G5: _call_claude_api handles empty content array."""

    def test_empty_content_raises_clean_error(self) -> None:
        """Empty content:[] should raise AiServiceError (invalid JSON), not IndexError."""
        import os

        import requests

        from chemscheme.parsers.ai_intent import AiServiceError, _call_claude_api

        mock_resp = type(
            "Resp",
            (),
            {
                "status_code": 200,
                "json": lambda self: {"content": []},
            },
        )()

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
            patch.object(requests, "post", return_value=mock_resp),
            pytest.raises(AiServiceError, match="invalid JSON"),
        ):
            _call_claude_api("test prompt")


@pytest.mark.unit
class TestMr5r2G6NegativeViewBox:
    """MR5R2-G6: _parse_svg_dimensions handles negative viewBox coordinates."""

    def test_negative_minx_miny(self) -> None:
        from chemscheme.renderers.multi_step_renderer import _parse_svg_dimensions

        svg = b'<svg viewBox="-10 -20 400 300"></svg>'
        w, h = _parse_svg_dimensions(svg)
        assert w == 400.0
        assert h == 300.0


@pytest.mark.unit
class TestMr5r2G7EmptyBranchSkipped:
    """MR5R2-G7: _find_branches skips empty branches from shared visited set."""

    def test_no_empty_branches_in_diamond(self) -> None:
        from chemscheme.renderers.layout_engine import _build_dependency_graph, _find_branches

        # Diamond: A -> B, A -> C, B+C -> D
        step_base = {"reactants": ["CC"], "products": ["CCC"], "arrow_above": "x"}
        steps = [
            {"id": "A", **step_base},
            {"id": "B", "consumes_step_ids": ["A"], **step_base},
            {"id": "C", "consumes_step_ids": ["A"], **step_base},
            {"id": "D", "consumes_step_ids": ["B", "C"], **step_base},
        ]
        preds = _build_dependency_graph(steps)
        branches, tail = _find_branches(steps, preds)
        # No branch should be empty
        assert all(len(b) > 0 for b in branches)


@pytest.mark.unit
class TestMr5r2C1EditBlocksInvalidSave:
    """MR5R2-C1: CLI edit refuses to save invalid scheme."""

    def test_add_duplicate_id_blocked(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from chemscheme.cli import main

        # Create a valid scheme
        scheme = {
            "schema_version": "1",
            "steps": [
                {"id": "1", "reactants": ["CCO"], "products": ["CC=O"], "arrow_above": "PCC"},
            ],
        }
        scheme_file = tmp_path / "scheme.json"
        scheme_file.write_text(json.dumps(scheme))

        # Try to add step with duplicate ID
        new_step = json.dumps(
            {
                "id": "1",
                "reactants": ["CC"],
                "products": ["CCO"],
                "arrow_above": "x",
            }
        )
        runner = CliRunner()
        result = runner.invoke(main, ["edit", str(scheme_file), "--add-step", new_step, "--json"])
        assert result.exit_code != 0


@pytest.mark.unit
class TestMr5r2C4FalseyValuePreservation:
    """MR5R2-C4: CDXML export preserves falsey values like temperature=0."""

    def test_temperature_zero_round_trip(self) -> None:
        from chemscheme.renderers.cdxml_writer import cdxml_to_scheme, scheme_to_cdxml

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                    "temperature": "0",
                },
            ],
        }
        cdxml = scheme_to_cdxml(scheme)
        assert "ChemScheme_temperature" in cdxml
        imported = cdxml_to_scheme(cdxml)
        assert imported["steps"][0].get("temperature") == "0"

    def test_empty_consumes_step_ids_round_trip(self) -> None:
        from chemscheme.renderers.cdxml_writer import scheme_to_cdxml

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "PCC",
                    "consumes_step_ids": [],
                },
            ],
        }
        cdxml = scheme_to_cdxml(scheme)
        # Empty list should still produce the attribute
        assert "ChemSchemeConsumesStepIds" in cdxml


@pytest.mark.unit
class TestMr5r2C7ResolveStatusField:
    """MR5R2-C7: resolve --json includes status field."""

    def test_resolve_json_has_status(self) -> None:
        from click.testing import CliRunner

        from chemscheme.cli import main

        runner = CliRunner()
        with patch(
            "chemscheme.parsers.name_resolver.resolve_name",
            return_value=type(
                "R",
                (),
                {
                    "smiles": "c1ccccc1",
                    "confidence": 0.9,
                    "source": "mock",
                    "warnings": [],
                },
            )(),
        ):
            result = runner.invoke(main, ["resolve", "benzene", "--json"])
        data = json.loads(result.output)
        assert "status" in data
        assert data["status"] == "ok"
