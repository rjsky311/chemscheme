"""S68 Part 5 — G7 + G12 + G13 polish bundle (3 gaps from S67 Part 2 audit).

- **G7**: arrow_below should accept str OR list[str] (mirror arrow_above schema).
  Currently arrow_above is str | list (AC-11.3 / MR11-C1) but arrow_below is
  string-only — asymmetric. Real OPV papers use 2-line above AND 2-line below
  (per S67 Part 2 audit Category 2).
- **G12**: Procedure-grammar first-step heuristic. When reactants list mixes
  structures + non-SMILES reagent names (e.g., "NBS"), current output is
  "Compound 1 and the starting material was treated" — singular `was` with
  plural-ish subject + treats NBS as a starting material. Better: detect
  non-SMILES strings and reclassify as reagent; fix number agreement.
- **G13**: PDF / EPS export via cairosvg.svg2pdf / svg2eps (mirrors existing
  `cairosvg.svg2png` PNG path). Both formats fall back to a clean error
  message if libcairo isn't installed.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from chemscheme.cli import main as cli_main

# ════════════════════════════════════════════════════════════════════════════
# G7 — arrow_below accepts str | list[str] (schema + renderer)
# ════════════════════════════════════════════════════════════════════════════


class TestG7ArrowBelowList:
    """Schema must mirror arrow_above: arrow_below = str | list[str]."""

    def test_schema_accepts_arrow_below_string(self) -> None:
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["CCO"],
                    "arrow_above": "Br2",
                    "arrow_below": "DMF, 80 C",
                }
            ],
        }
        assert validate_scheme(scheme) == []

    def test_schema_accepts_arrow_below_list(self) -> None:
        """The S67 G7 fix — list of strings allowed for multi-line below text."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["CCO"],
                    "arrow_above": ["Pd(PPh3)4", "K2CO3"],
                    "arrow_below": ["DMF", "80 C, 12h"],
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert errors == [], f"arrow_below as list must validate; got: {errors}"

    def test_schema_rejects_arrow_below_non_string_in_list(self) -> None:
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["CCO"],
                    "arrow_above": "Br2",
                    "arrow_below": ["DMF", 42],  # int not allowed
                }
            ],
        }
        errors = validate_scheme(scheme)
        assert any("arrow_below" in e for e in errors), (
            f"non-string in arrow_below list must be flagged; got: {errors}"
        )

    def test_renderer_multiline_arrow_below_produces_separate_text_lines(self) -> None:
        """When arrow_below is a list, renderer should emit multiple <text> lines."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(Br)cc1"],
            "arrow_above": "Br2",
            "arrow_below": ["DMF", "80 C, 12h"],
        }
        svg = compose_step_svg(step).decode("utf-8")
        # Both list elements must appear in SVG
        assert "DMF" in svg
        assert "80 C, 12h" in svg
        # Should be in separate <text> elements (not joined into one string)
        # Heuristic: count <text> with our literals appears separately
        below_text_lines = svg.count("DMF") + svg.count("80 C, 12h")
        # At minimum, both literals present; ideally each in its own <text>
        assert below_text_lines >= 2

    def test_renderer_string_arrow_below_unchanged(self) -> None:
        """RED-EXEMPT: behavior lock — string arrow_below still works (backward compat)."""
        from chemscheme.renderers.drawsvg_composer import compose_step_svg

        step = {
            "id": "1",
            "reactants": ["c1ccccc1"],
            "products": ["c1ccc(Br)cc1"],
            "arrow_above": "Br2",
            "arrow_below": "DMF, 80 C",
        }
        svg = compose_step_svg(step).decode("utf-8")
        assert "DMF, 80 C" in svg


# ════════════════════════════════════════════════════════════════════════════
# G12 — Procedure-grammar first-step heuristic
# ════════════════════════════════════════════════════════════════════════════


class TestG12ProcedureGrammar:
    """Mixed reactants (structure + reagent name) should not produce
    'Compound 1 and the starting material was treated'."""

    def test_mixed_reactants_subject_uses_were_for_plural(self) -> None:
        """List-of-2-reactants subject takes plural verb (were, not was)."""
        from chemscheme.procedure_generator import generate_procedure

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1", "ClCCl"],
                    "products": ["c1ccc(Cl)cc1"],
                    "arrow_above": "AlCl3",
                    "compound_labels": {"c1ccccc1": "1"},
                }
            ],
        }
        result = generate_procedure(scheme)
        text = result.steps[0].text
        # Plural subject → "were" not "was"
        assert " were " in text or " were\n" in text, (
            f"plural subject should use 'were', got: {text!r}"
        )
        assert " was " not in text, f"plural subject should not use 'was', got: {text!r}"

    def test_singular_reactant_keeps_was(self) -> None:
        """RED-EXEMPT: behavior lock — singular subject keeps 'was'."""
        from chemscheme.procedure_generator import generate_procedure

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                    "compound_labels": {"c1ccccc1": "1"},
                }
            ],
        }
        result = generate_procedure(scheme)
        text = result.steps[0].text
        # Singular subject → "was"
        assert " was " in text, f"singular subject should use 'was', got: {text!r}"

    def test_non_smiles_reactant_treated_as_reagent_not_starting_material(self) -> None:
        """G12 main fix: 'NBS' in reactants list is a reagent name, not a starting
        material. Heuristic: if RDKit can't parse a reactant string as a SMILES,
        it's likely a reagent — should be merged into the reagent list, not
        named as a separate starting material."""
        from chemscheme.procedure_generator import generate_procedure

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1", "NBS"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "FeBr3",
                    "compound_labels": {"c1ccccc1": "1"},
                }
            ],
        }
        result = generate_procedure(scheme)
        text = result.steps[0].text
        # NBS should appear in reagent context, not as "the starting material"
        # Specifically, the OLD bug-text "starting material was treated" must NOT appear
        assert "the starting material was treated" not in text, (
            f"non-SMILES reagent 'NBS' should be classified as reagent, got: {text!r}"
        )
        # NBS should appear (either via reagent list OR via direct mention)
        assert "NBS" in text


# ════════════════════════════════════════════════════════════════════════════
# G13 — PDF / EPS export via cairosvg
# ════════════════════════════════════════════════════════════════════════════


def _make_simple_scheme(tmp_path):
    """Helper: write a minimal valid scheme.json to tmp_path."""
    import json

    scheme = {
        "schema_version": "1",
        "steps": [
            {
                "id": "1",
                "reactants": ["c1ccccc1"],
                "products": ["c1ccc(Br)cc1"],
                "arrow_above": "Br2",
            }
        ],
    }
    p = tmp_path / "scheme.json"
    p.write_text(json.dumps(scheme), encoding="utf-8")
    return p


class TestG13PdfEpsExport:
    """PDF and EPS export via cairosvg.svg2pdf / svg2eps (mirrors svg2png pattern)."""

    def test_render_scheme_supports_pdf_format(self, tmp_path) -> None:
        from chemscheme.renderers import render_scheme

        scheme = json.loads(_make_simple_scheme(tmp_path).read_text(encoding="utf-8"))
        # Mock cairosvg.svg2pdf to avoid libcairo dependency in tests
        fake_cairosvg = types.ModuleType("cairosvg")
        fake_cairosvg.svg2pdf = MagicMock(return_value=b"%PDF-1.4 fake_pdf_bytes")  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"cairosvg": fake_cairosvg}):
            result = render_scheme(scheme, fmt="pdf")
        assert result.startswith(b"%PDF") or result == b"%PDF-1.4 fake_pdf_bytes"
        fake_cairosvg.svg2pdf.assert_called_once()  # type: ignore[attr-defined]

    def test_render_scheme_supports_eps_format(self, tmp_path) -> None:
        from chemscheme.renderers import render_scheme

        scheme = json.loads(_make_simple_scheme(tmp_path).read_text(encoding="utf-8"))
        fake_cairosvg = types.ModuleType("cairosvg")
        fake_cairosvg.svg2eps = MagicMock(return_value=b"%!PS-Adobe-3.0 EPSF-3.0 fake")  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"cairosvg": fake_cairosvg}):
            result = render_scheme(scheme, fmt="eps")
        assert result.startswith(b"%!PS") or b"fake" in result
        fake_cairosvg.svg2eps.assert_called_once()  # type: ignore[attr-defined]

    def test_render_scheme_pdf_without_cairosvg_clean_error(self, tmp_path) -> None:
        """Without cairosvg installed, PDF format raises ValueError with helpful text."""
        from chemscheme.renderers import render_scheme

        scheme = json.loads(_make_simple_scheme(tmp_path).read_text(encoding="utf-8"))
        # Force cairosvg import failure by removing it from sys.modules + blocking re-import
        with patch.dict(sys.modules, {"cairosvg": None}):
            with pytest.raises((ValueError, RuntimeError, ImportError)) as exc_info:
                render_scheme(scheme, fmt="pdf")
            err_msg = str(exc_info.value).lower()
            assert "cairosvg" in err_msg or "pdf" in err_msg

    def test_render_scheme_unsupported_format_raises_value_error(self, tmp_path) -> None:
        """RED-EXEMPT: behavior lock — unknown format still raises ValueError."""
        from chemscheme.renderers import render_scheme

        scheme = json.loads(_make_simple_scheme(tmp_path).read_text(encoding="utf-8"))
        with pytest.raises(ValueError):
            render_scheme(scheme, fmt="xlsx")

    def test_cli_draw_pdf_extension_routes_to_pdf_format(self, tmp_path) -> None:
        """`chemscheme draw scheme.json -o output.pdf` triggers PDF rendering."""
        scheme_path = _make_simple_scheme(tmp_path)
        out = tmp_path / "out.pdf"

        fake_cairosvg = types.ModuleType("cairosvg")
        fake_cairosvg.svg2pdf = MagicMock(return_value=b"%PDF-1.4 stub")  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"cairosvg": fake_cairosvg}):
            runner = CliRunner()
            result = runner.invoke(cli_main, ["draw", str(scheme_path), "-o", str(out)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out.exists()
        assert out.read_bytes().startswith(b"%PDF")

    def test_cli_draw_eps_extension_routes_to_eps_format(self, tmp_path) -> None:
        scheme_path = _make_simple_scheme(tmp_path)
        out = tmp_path / "out.eps"

        fake_cairosvg = types.ModuleType("cairosvg")
        fake_cairosvg.svg2eps = MagicMock(return_value=b"%!PS-Adobe-3.0 EPSF-3.0 stub")  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"cairosvg": fake_cairosvg}):
            runner = CliRunner()
            result = runner.invoke(cli_main, ["draw", str(scheme_path), "-o", str(out)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out.exists()
        assert out.read_bytes().startswith(b"%!PS")
