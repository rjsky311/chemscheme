"""G3-G5 theme cluster — molecule-interior theme fidelity.

S67 Part 2 paper-grade audit surfaced:
- G3: ACS/Nature themes did NOT override RDKit default atom palette
  (sulfur rendered as olive #CCCC00, oxygen as red, nitrogen as blue,
  bromine as brown) — publication-grade journals typically require
  all-black molecule structures.
- G5: Theme font_family was applied only to annotation text, not to
  RDKit-internal atom labels (separate issue, G4, deferred).

This file tests G3 fix: journal themes with ``molecule_bw_palette=True``
MUST cause the rendered SVG to contain zero colored atom/bond strokes
inside molecule regions. DEFAULT theme preserves existing behavior
(backward compatibility).

BDD Feature: Molecule-interior theme fidelity for publication journals.
"""

from __future__ import annotations

import re

import pytest

# RDKit default atom-colour strokes seen in S67 Part 2 audit of
# paper_grade_test.svg (ACS theme): CCCC00 (sulfur olive), FF0000 (oxygen red),
# 0000FF (nitrogen blue), 7F4C19 (bromine dark-brown). We assert none of these
# appear in journal-theme-rendered SVG.
RDKIT_HETEROATOM_STROKES = {
    "CCCC00",  # sulfur (olive/dark-yellow)
    "FF0000",  # oxygen (red)
    "0000FF",  # nitrogen (blue)
    "7F4C19",  # bromine (dark brown)
    "009900",  # chlorine (green, commonly used)
    "00CCCC",  # phosphorus
}


def _count_colored_strokes(svg: str | bytes) -> dict[str, int]:
    """Return counts of non-black stroke colors per hex code.

    Accepts str or bytes (compose_step_svg may return either depending on path).
    """
    if isinstance(svg, bytes):
        svg = svg.decode("utf-8", errors="replace")
    strokes: list[str] = re.findall(r"stroke:#([0-9A-Fa-f]{6})", svg)
    counts: dict[str, int] = {}
    for s in strokes:
        code = s.upper()
        if code == "000000":
            continue
        counts[code] = counts.get(code, 0) + 1
    return counts


def _heteroatom_strokes_found(svg: str | bytes) -> dict[str, int]:
    """Return subset of colored strokes matching RDKit heteroatom palette."""
    counts = _count_colored_strokes(svg)
    return {k: v for k, v in counts.items() if k in RDKIT_HETEROATOM_STROKES}


# ---------------------------------------------------------------------------
# Fixture: paper-grade-ish molecule set (heteroatoms S/N/O/Br present)
# ---------------------------------------------------------------------------


@pytest.fixture()  # type: ignore[misc]
def paper_step() -> dict:  # type: ignore[type-arg]
    """Single-step scheme with thiophene/NBS-like SMILES containing S, N, O, Br.

    The heteroatoms will render in RDKit default palette as colored
    strokes unless theme's molecule_bw_palette forces all-black.
    """
    return {
        "id": "step1",
        "reactants": ["c1ccc(Br)cc1", "c1ccsc1"],  # benzene-Br + thiophene (has Br + S)
        "products": ["O=Cc1ccsc1"],  # thiophene-aldehyde (has S + O)
        "arrow_above": "NBS",
        "arrow_below": "CHCl3, 0 C",
        "compound_labels": {
            "c1ccc(Br)cc1": "1",
            "c1ccsc1": "2",
            "O=Cc1ccsc1": "3",
        },
    }


# ---------------------------------------------------------------------------
# G3: journal themes produce all-black molecules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "theme_name",
    ["acs", "nature", "wiley", "rsc"],
)
def test_journal_theme_molecule_has_no_heteroatom_colors(
    theme_name: str,
    paper_step: dict,  # type: ignore[type-arg]
) -> None:
    """G3 fix: ACS/Nature/Wiley/RSC themes MUST produce molecules with no
    RDKit heteroatom colored strokes.

    Before fix: SVG contained #CCCC00 (sulfur), #FF0000 (oxygen),
    #0000FF (nitrogen), #7F4C19 (bromine) strokes inside molecule paths.
    After fix: theme.molecule_bw_palette=True routes through RDKit
    useBWAtomPalette(), forcing all atom/bond strokes to #000000.
    """
    from chemscheme.renderers.drawsvg_composer import compose_step_svg
    from chemscheme.renderers.themes import get_theme

    theme = get_theme(theme_name)
    svg = compose_step_svg(paper_step, theme=theme)

    found = _heteroatom_strokes_found(svg)
    assert not found, (
        f"Theme '{theme_name}' leaked RDKit heteroatom colors into molecule "
        f"interior: {found}. Expected all-black via theme.molecule_bw_palette."
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "theme_name",
    ["acs", "nature", "wiley", "rsc"],
)
def test_journal_theme_has_molecule_bw_palette_flag(theme_name: str) -> None:
    """Each publication journal preset must declare molecule_bw_palette=True.

    This is the structural assertion: themes that PROMISE journal fidelity
    must set the flag. The fix is inoperative if a preset forgets to set it.
    """
    from chemscheme.renderers.themes import get_theme

    theme = get_theme(theme_name)
    assert hasattr(theme, "molecule_bw_palette"), (
        "SchemeTheme must have molecule_bw_palette field (G3 fix)."
    )
    assert theme.molecule_bw_palette is True, (
        f"Journal theme '{theme_name}' must have molecule_bw_palette=True."
    )


# ---------------------------------------------------------------------------
# Backward-compat: default theme preserves existing colored behavior
# ---------------------------------------------------------------------------


def test_default_theme_preserves_heteroatom_colors(paper_step: dict) -> None:  # type: ignore[type-arg]
    """Default theme must keep molecule_bw_palette=False so pre-Phase-14
    colored behavior is preserved.

    Rationale: byte-identical pre-Phase-14 behavior is a documented
    property of DEFAULT_THEME; users may depend on colored output from
    the default code path.
    """
    from chemscheme.renderers.drawsvg_composer import compose_step_svg
    from chemscheme.renderers.themes import DEFAULT_THEME, get_theme

    assert DEFAULT_THEME.molecule_bw_palette is False, (
        "DEFAULT_THEME must preserve pre-Phase-14 colored behavior (molecule_bw_palette=False)."
    )

    theme = get_theme("default")
    svg = compose_step_svg(paper_step, theme=theme)
    found = _heteroatom_strokes_found(svg)
    assert found, (
        "Default theme should still render heteroatoms in RDKit default "
        "palette (sulfur olive, oxygen red, etc.). Found none — either "
        "default behavior changed or SMILES has no heteroatoms."
    )


# ---------------------------------------------------------------------------
# Regression: adversarial — scheme with only carbons (no heteroatoms)
# ---------------------------------------------------------------------------


# RED-EXEMPT: behavior lock — all-carbon scheme has no heteroatoms to leak;
# this test is a regression guard to ensure future theme-routing refactors
# don't break all-carbon rendering. Passes from day one by design.
def test_journal_theme_allcarbon_scheme_renders_without_error() -> None:
    """Adversarial: a scheme with only carbons has no heteroatom strokes
    even in default theme. ACS theme must still render without error
    and produce no colored strokes."""
    from chemscheme.renderers.drawsvg_composer import compose_step_svg
    from chemscheme.renderers.themes import get_theme

    allcarbon = {
        "id": "step1",
        "reactants": ["CCCC"],
        "products": ["CCCCCC"],
        "arrow_above": "polymerize",
    }
    theme = get_theme("acs")
    svg = compose_step_svg(allcarbon, theme=theme)

    assert svg, "ACS theme should render all-carbon scheme without error."
    found = _heteroatom_strokes_found(svg)
    assert not found, f"All-carbon scheme should have zero heteroatom strokes; found {found}"
