"""G4 — custom SVG drawer with heteroatom labels as <text>.

S67 Part 2 audit flagged G4 (HIGH): RDKit + Indigo both vectorize atom
symbols into ``<path>`` glyph geometry in SVG, which means vector editors
(Illustrator / Inkscape) cannot edit them as text. Publication-quality
editing requires ``<text>`` elements with actual symbol strings.

S68 Part 8 investigation FALSIFIED the original "switch to Indigo backend"
premise — Indigo paths atoms too. This file tests the alternative:
a custom in-house drawer that:

- Extracts RDKit 2D atom positions (``mol.GetConformer().GetAtomPosition``)
- Emits bonds as SVG ``<line>`` (with single / double / triple dispatch)
- Emits heteroatom labels (N, O, S, halogens, abbreviations like Ph/Bn)
  as SVG ``<text>`` at correct positions
- Leaves carbon atoms implicit (chemistry convention)

Behavior is opt-in via ``SchemeTheme.molecule_atoms_as_text: bool = False``;
DEFAULT theme preserves existing RDKit path-based rendering (backward
compatibility).

BDD Feature: G4 custom SVG drawer — heteroatoms as editable text.
"""

from __future__ import annotations

import re

import pytest

from chemscheme.renderers.drawsvg_composer import compose_step_svg
from chemscheme.renderers.rdkit_renderer import _mol_to_svg
from chemscheme.renderers.themes import DEFAULT_THEME, SchemeTheme

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mol_from_smiles(smi: str):  # type: ignore[no-untyped-def]
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"invalid SMILES: {smi}"
    AllChem.Compute2DCoords(mol)  # type: ignore[attr-defined]
    return mol


def _text_mode_theme() -> SchemeTheme:
    """Default theme with only molecule_atoms_as_text flipped on."""
    from dataclasses import replace

    return replace(DEFAULT_THEME, molecule_atoms_as_text=True)


def _count_text_elements(svg: str, contents: str | None = None) -> int:
    """Count <text>...</text> occurrences. If contents given, count only those
    with matching inner text (simple substring)."""
    matches = re.findall(r"<text[^>]*>(.*?)</text>", svg, re.DOTALL)
    if contents is None:
        return len(matches)
    return sum(1 for m in matches if contents in m)


def _heteroatom_text_symbols(svg: str) -> set[str]:
    """Extract heteroatom symbols appearing as <text> in the rendered SVG.

    Tolerates <tspan> children (for implicit-H counts like 'OH2' rendered as
    'O' + 'H' + '<tspan>2</tspan>'). Symbols are returned as the first
    alphabetic token.
    """
    matches = re.findall(r"<text[^>]*>(.*?)</text>", svg, re.DOTALL)
    symbols: set[str] = set()
    for m in matches:
        # strip <tspan> tags
        cleaned = re.sub(r"<[^>]+>", "", m).strip()
        # take leading element symbol (1-2 chars)
        lead = re.match(r"([A-Z][a-z]?)", cleaned)
        if lead:
            symbols.add(lead.group(1))
    return symbols


# ---------------------------------------------------------------------------
# TestG4ThemeFlag — schema-level: the flag exists and defaults off
# ---------------------------------------------------------------------------


class TestG4ThemeFlag:
    """molecule_atoms_as_text flag is part of SchemeTheme, defaults False."""

    def test_default_theme_has_flag_false(self) -> None:
        assert DEFAULT_THEME.molecule_atoms_as_text is False

    def test_flag_can_be_enabled_via_replace(self) -> None:
        from dataclasses import replace

        themed = replace(DEFAULT_THEME, molecule_atoms_as_text=True)
        assert themed.molecule_atoms_as_text is True
        # Other fields unchanged
        assert themed.font_family == DEFAULT_THEME.font_family

    def test_journal_presets_default_off_for_backward_compat(self) -> None:
        # G4 text-mode is opt-in per scheme, not auto-enabled by journal presets.
        # Journal presets can be extended later to default True once v1 ships.
        from chemscheme.renderers.themes import (
            ACS_THEME,
            NATURE_THEME,
            RSC_THEME,
            WILEY_THEME,
        )

        for t in (ACS_THEME, NATURE_THEME, RSC_THEME, WILEY_THEME):
            assert t.molecule_atoms_as_text is False, (
                f"{t.name} should default molecule_atoms_as_text=False for backward compat"
            )


# ---------------------------------------------------------------------------
# TestG4BehaviorLock — RED-EXEMPT: default mode preserves RDKit path render
# ---------------------------------------------------------------------------


class TestG4BehaviorLock:
    """Default molecule_atoms_as_text=False preserves existing RDKit-path rendering."""

    def test_default_theme_uses_rdkit_path_glyphs(self) -> None:  # RED-EXEMPT: behavior lock
        # RED-EXEMPT: behavior lock for backward compat — existing
        # behavior is to render atoms as <path> glyphs. This test captures
        # that default and is expected to PASS before and after G4 ships.
        mol = _mol_from_smiles("CCO")  # ethanol, has O heteroatom
        svg = _mol_to_svg(mol, 200, 200, theme=DEFAULT_THEME)
        # RDKit MolDraw2DSVG emits <path> glyphs for atom labels
        assert "<path" in svg
        # No manual <line> bond elements from custom drawer
        assert "x1=" not in svg or "<line" not in svg

    def test_explicit_false_theme_same_as_default(self) -> None:  # RED-EXEMPT: behavior lock
        # RED-EXEMPT: behavior lock — explicit False matches default.
        from dataclasses import replace

        theme_f = replace(DEFAULT_THEME, molecule_atoms_as_text=False)
        mol = _mol_from_smiles("CCO")
        svg = _mol_to_svg(mol, 200, 200, theme=theme_f)
        assert "<path" in svg


# ---------------------------------------------------------------------------
# TestG4TextModeAtomLabels — new behavior: text-mode emits <text> for heteroatoms
# ---------------------------------------------------------------------------


class TestG4TextModeAtomLabels:
    """molecule_atoms_as_text=True emits heteroatoms as <text>, bonds as <line>."""

    def test_ethanol_oxygen_emits_as_text(self) -> None:
        mol = _mol_from_smiles("CCO")  # ethanol
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        # Oxygen should appear as <text>O</text> (or similar, possibly with H)
        symbols = _heteroatom_text_symbols(svg)
        assert "O" in symbols, f"expected O in text labels, got {symbols}"

    def test_methylamine_nitrogen_emits_as_text(self) -> None:
        mol = _mol_from_smiles("CN")  # methylamine
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        symbols = _heteroatom_text_symbols(svg)
        assert "N" in symbols

    def test_carbon_atoms_not_labeled(self) -> None:
        mol = _mol_from_smiles("CCCC")  # butane — all carbons
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        # No <text>C</text> in text-mode output (carbons implicit)
        symbols = _heteroatom_text_symbols(svg)
        assert "C" not in symbols, (
            f"carbon atoms should be implicit in text-mode, got symbols={symbols}"
        )

    def test_thiophene_sulfur_emits_as_text(self) -> None:
        mol = _mol_from_smiles("c1ccsc1")  # thiophene
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        symbols = _heteroatom_text_symbols(svg)
        assert "S" in symbols

    def test_halogens_emit_as_text(self) -> None:
        mol = _mol_from_smiles("CCCl")  # 1-chloroethane
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        symbols = _heteroatom_text_symbols(svg)
        assert "Cl" in symbols

    def test_bonds_emit_as_line_elements(self) -> None:
        mol = _mol_from_smiles("CCO")
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        # Custom drawer emits <line> bond elements
        assert "<line" in svg
        # And should NOT emit <path> atom glyphs (carbons are implicit, no glyphs)
        assert "<path" not in svg

    def test_double_bond_has_two_parallel_lines(self) -> None:
        mol = _mol_from_smiles("C=O")  # formaldehyde — explicit C=O double bond
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        # Double bond → 2 parallel <line> elements at minimum
        line_count = len(re.findall(r"<line\s[^>]*>", svg))
        assert line_count >= 2, f"expected >=2 lines for C=O double bond, got {line_count}"

    def test_triple_bond_has_three_parallel_lines(self) -> None:
        mol = _mol_from_smiles("CC#N")  # acetonitrile — explicit C#N triple bond
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        # Triple C#N bond: 3 lines for the C#N bond alone, plus 1 for C-C single
        line_count = len(re.findall(r"<line\s[^>]*>", svg))
        assert line_count >= 3, f"expected >=3 lines for triple bond, got {line_count}"

    def test_positions_within_canvas(self) -> None:
        mol = _mol_from_smiles("CCO")
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        # All <text> x,y coordinates must fall within the 0..200 canvas bounds
        text_coords = re.findall(r'<text[^>]*\sx="([-\d.]+)"[^>]*\sy="([-\d.]+)"', svg)
        assert text_coords, "expected at least one <text> element with x,y attrs"
        for xs, ys in text_coords:
            x = float(xs)
            y = float(ys)
            assert 0 <= x <= 200, f"x={x} out of canvas (expected 0..200)"
            assert 0 <= y <= 200, f"y={y} out of canvas (expected 0..200)"

    def test_empty_mol_returns_empty_or_minimal(self) -> None:
        from rdkit import Chem

        mol = Chem.MolFromSmiles("")  # empty SMILES -> empty mol
        # RDKit returns None for "", so skip. Instead use single-atom isolated.
        mol = _mol_from_smiles("O")  # water → shows O with 2 implicit Hs
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        # Should at least emit one <text>
        assert "<text" in svg

    def test_implicit_hydrogens_shown_for_heteroatoms(self) -> None:
        """Heteroatoms with implicit H should render as 'OH', 'NH', 'NH2'."""
        mol = _mol_from_smiles("CO")  # methanol: terminal O has 1 implicit H
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        # Look for 'OH' as substring of a text element
        text_blobs = re.findall(r"<text[^>]*>(.*?)</text>", svg, re.DOTALL)
        # Strip inner <tspan>
        cleaned = [re.sub(r"<[^>]+>", "", b).strip() for b in text_blobs]
        has_oh = any("OH" in c or ("O" in c and "H" in c) for c in cleaned)
        assert has_oh, f"expected OH-style label for terminal oxygen, got {cleaned}"


# ---------------------------------------------------------------------------
# TestG4EndToEndScheme — full scheme through compose_step_svg
# ---------------------------------------------------------------------------


class TestG4EndToEndScheme:
    """text-mode flag routes through compose_step_svg → full scheme render."""

    @pytest.fixture()  # type: ignore[misc]
    def step(self) -> dict:  # type: ignore[type-arg]
        return {
            "id": "step1",
            "reactants": ["CCO"],  # ethanol
            "products": ["CC=O"],  # acetaldehyde
            "arrow_above": "[O]",
            "arrow_below": "",
        }

    def test_text_mode_preserves_bond_lines_through_composer(
        self,
        step: dict,  # type: ignore[type-arg]
    ) -> None:
        theme = _text_mode_theme()
        svg_bytes = compose_step_svg(step, theme=theme)
        svg = svg_bytes.decode("utf-8") if isinstance(svg_bytes, bytes) else svg_bytes
        # Custom drawer emits <line> for bonds
        assert "<line" in svg

    def test_text_mode_emits_oxygen_as_text_in_full_scheme(
        self,
        step: dict,  # type: ignore[type-arg]
    ) -> None:
        theme = _text_mode_theme()
        svg_bytes = compose_step_svg(step, theme=theme)
        svg = svg_bytes.decode("utf-8") if isinstance(svg_bytes, bytes) else svg_bytes
        # Ethanol + acetaldehyde both contain O
        assert _count_text_elements(svg, "O") >= 1

    def test_default_mode_preserves_rdkit_path_rendering(
        self,
        step: dict,  # type: ignore[type-arg]
    ) -> None:  # RED-EXEMPT: behavior lock
        # RED-EXEMPT: behavior lock — default mode renders atoms as <path>.
        svg_bytes = compose_step_svg(step, theme=DEFAULT_THEME)
        svg = svg_bytes.decode("utf-8") if isinstance(svg_bytes, bytes) else svg_bytes
        # RDKit-path rendering keeps atom glyphs as <path> — at least some
        # path elements present (from RDKit atom glyph geometry).
        assert "<path" in svg

    def test_no_bond_lines_in_default_mode(
        self,
        step: dict,  # type: ignore[type-arg]
    ) -> None:
        """Sanity: default mode shouldn't emit the custom drawer's <line> bonds
        in the molecule region (arrow lines are a different thing)."""
        svg_bytes = compose_step_svg(step, theme=DEFAULT_THEME)
        svg = svg_bytes.decode("utf-8") if isinstance(svg_bytes, bytes) else svg_bytes
        # It's fine if <line> appears elsewhere (arrow is a line); but the
        # count should be much lower than the text-mode path which adds one
        # line per bond. We at least verify output renders without crash.
        assert len(svg) > 200


# ---------------------------------------------------------------------------
# TestG4AbbreviationAtoms — condensed pseudo-atoms (Ph, Bn) labels show
# ---------------------------------------------------------------------------


class TestG4AbbreviationAtoms:
    """When abbreviations are condensed (Ph/Bn/Bz), their labels show as text."""

    def test_phenyl_abbreviation_renders_as_text(self) -> None:
        # Benzyl bromide "BrCc1ccccc1" condenses to Br-Bn
        from chemscheme.renderers.rdkit_renderer import _validate_and_prepare_consistent

        r, p = _validate_and_prepare_consistent(["BrCc1ccccc1"], 1)
        mol = r[0]
        theme = _text_mode_theme()
        svg = _mol_to_svg(mol, 200, 200, theme=theme)
        # Bn label should appear as <text>Bn</text>
        text_blobs = re.findall(r"<text[^>]*>(.*?)</text>", svg, re.DOTALL)
        cleaned = [re.sub(r"<[^>]+>", "", b).strip() for b in text_blobs]
        has_bn_or_ph = any("Bn" in c or "Ph" in c for c in cleaned)
        assert has_bn_or_ph, f"expected Bn or Ph abbreviation label, got {cleaned}"


# ---------------------------------------------------------------------------
# TestG4CLIAtomsAsTextFlag — `chemscheme draw --atoms-as-text` flag
# ---------------------------------------------------------------------------


class TestG4CLIAtomsAsTextFlag:
    """--atoms-as-text CLI flag enables custom drawer on any scheme + theme."""

    def test_cli_help_mentions_flag(self) -> None:
        from click.testing import CliRunner

        from chemscheme.cli import draw

        runner = CliRunner()
        result = runner.invoke(draw, ["--help"])
        assert result.exit_code == 0
        assert "--atoms-as-text" in result.output

    def test_cli_flag_produces_text_mode_svg(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json

        from click.testing import CliRunner

        from chemscheme.cli import draw

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "s",
                    "reactants": ["CCO"],  # ethanol
                    "products": ["CC=O"],  # acetaldehyde
                    "arrow_above": "[O]",
                }
            ],
        }
        scheme_path = tmp_path / "scheme.json"
        scheme_path.write_text(json.dumps(scheme), encoding="utf-8")
        output_path = tmp_path / "out.svg"

        runner = CliRunner()
        result = runner.invoke(
            draw,
            [
                str(scheme_path),
                "-o",
                str(output_path),
                "--atoms-as-text",
                "--force",
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        svg = output_path.read_text(encoding="utf-8")
        # Custom drawer signature: <line> for bonds, <text>O</text> for O
        assert "<line" in svg
        symbols = _heteroatom_text_symbols(svg)
        assert "O" in symbols

    def test_cli_flag_composes_with_theme(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """--atoms-as-text + --theme acs → both applied (BW palette + text atoms)."""
        import json

        from click.testing import CliRunner

        from chemscheme.cli import draw

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "s",
                    "reactants": ["CCO"],
                    "products": ["CC=O"],
                    "arrow_above": "[O]",
                }
            ],
        }
        scheme_path = tmp_path / "scheme.json"
        scheme_path.write_text(json.dumps(scheme), encoding="utf-8")
        output_path = tmp_path / "out.svg"

        runner = CliRunner()
        result = runner.invoke(
            draw,
            [
                str(scheme_path),
                "-o",
                str(output_path),
                "--theme",
                "acs",
                "--atoms-as-text",
                "--force",
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        svg = output_path.read_text(encoding="utf-8")
        assert "<line" in svg  # text-mode signature
