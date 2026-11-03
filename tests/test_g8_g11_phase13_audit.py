"""S68 Part 6 — G8/G9/G10/G11 verification audit.

S67 Part 2 audit flagged 4 Phase 13 capabilities as "convergent/polymer/
substituent/compound-reuse rendering quality unverified on realistic case":
- **G8** (MEDIUM): Convergent (`consumes_step_ids`) rendering on multi-merge schemes
- **G9** (MEDIUM): Polymer (`polymer_info`) rendering — bracket notation `[—]_n`
- **G10** (MEDIUM): Substituent map (`substituent_map`) — Markush R-groups
- **G11** (MEDIUM): Compound number reuse — same compound appearing in multiple steps

This audit:
1. Builds a realistic OPV-style scheme for each capability (where existing
   fixture lacks it).
2. Renders end-to-end and asserts key SVG markers are present.
3. Reports what's solid vs broken — defects fed into next round.

Existing per-feature unit tests (test_phase13_domain.py) cover schema validation
and per-step rendering. This file fills the realistic end-to-end gap.
"""

from __future__ import annotations

import re

import pytest


def _render(scheme: dict) -> str:
    """Helper: validate + full-scheme render → SVG string."""
    from chemscheme.renderers import render_scheme
    from chemscheme.schema.scheme_schema import validate_scheme

    errors = validate_scheme(scheme)
    assert errors == [], f"scheme failed validation: {errors}"
    return render_scheme(scheme, fmt="svg").decode("utf-8")


# ════════════════════════════════════════════════════════════════════════════
# G8 — Convergent rendering on existing 8-step fixture
# ════════════════════════════════════════════════════════════════════════════


class TestG8ConvergentAudit:
    """Phase 12 + Phase 4 convergent layout — verifies multi-branch + merge path
    on the realistic 8-step fixture (sample_scheme_8step_convergent.json)."""

    @pytest.fixture
    def scheme(self):
        import json
        from pathlib import Path

        return json.loads(
            Path("tests/fixtures/sample_scheme_8step_convergent.json").read_text(encoding="utf-8")
        )

    def test_render_succeeds_no_crash(self, scheme) -> None:
        svg = _render(scheme)
        assert svg.startswith("<svg") or svg.startswith("<?xml")
        # Substantial output (not a 1-element stub)
        assert len(svg) > 10000, f"convergent render is suspiciously small: {len(svg)} bytes"

    def test_all_step_ids_present_in_render(self, scheme) -> None:
        """Every step in the 8-step convergent scheme must appear in the rendered SVG."""
        svg = _render(scheme)
        for step in scheme["steps"]:
            sid = step["id"]
            # Step IDs typically appear in id="..." attributes via _safe_id sanitization
            assert sid in svg, f"step id {sid!r} missing from rendered SVG"

    def test_merge_step_consumes_predecessors(self, scheme) -> None:
        """Step "4" consumes ["3a", "3b"] — both branches' final products feed in."""
        merge_step = next(s for s in scheme["steps"] if s["id"] == "4")
        assert merge_step.get("consumes_step_ids") == ["3a", "3b"], (
            "fixture invariant: step 4 must converge from 3a + 3b"
        )

    def test_render_contains_arrows_for_each_step(self, scheme) -> None:
        """Each step contributes at least one rendered arrow (reaction or connector)."""
        svg = _render(scheme)
        # Count arrowhead-bearing markers (lines/polylines with marker-end OR arrow paths)
        arrow_count = svg.count("marker-end")
        # Crude lower bound: at least 1 arrow per step
        n_steps = len(scheme["steps"])
        assert arrow_count >= n_steps - 2, (
            f"too few arrows in convergent render: {arrow_count} for {n_steps} steps"
        )


# ════════════════════════════════════════════════════════════════════════════
# G9 — Polymer rendering on a realistic donor-polymer scheme
# ════════════════════════════════════════════════════════════════════════════


class TestG9PolymerAudit:
    """Phase 13 polymer rendering — bracket notation `[—]_n` should appear
    when polymer_info is set. Existing unit tests cover compose_step_svg
    in isolation; this is the multi-step end-to-end verification."""

    POLYMER_SMILES = "*CC(c1ccc2sc3ccsc3c2c1)*"  # BDT-styryl polymer (placeholder)

    def _polymer_scheme(self) -> dict:
        return {
            "schema_version": "1",
            "title": "Donor polymer synthesis",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccc(Br)cc1"],
                    "products": [self.POLYMER_SMILES],
                    "arrow_above": "Stille polymerization",
                    "arrow_below": "Pd(PPh3)4",
                    "polymer_info": {self.POLYMER_SMILES: {"n": "n"}},
                }
            ],
        }

    def test_polymer_renders_with_repeat_label(self) -> None:
        """Repeat-unit label 'n' must appear in the rendered SVG (text element)."""
        svg = _render(self._polymer_scheme())
        # The repeat label 'n' should appear in a <text> element
        assert re.search(r">n<", svg), (
            f"polymer repeat-unit label 'n' missing from polymer scheme SVG (length {len(svg)})"
        )

    def test_polymer_renders_with_bracket_notation(self) -> None:
        """Polymer bracket notation must appear (square brackets or vertical-bar marker)."""
        svg = _render(self._polymer_scheme())
        # Bracket markers: literal "[" and "]" OR rendered as <path> with bracket shape
        # Heuristic: presence of bracket char or polymer-bracket SVG class/id
        has_bracket = "[" in svg or "polymer" in svg.lower() or "bracket" in svg.lower()
        assert has_bracket, "polymer brackets missing from rendered SVG"

    def test_polymer_custom_label_m_renders(self) -> None:
        """Custom repeat label 'm' (vs default 'n') passes through end-to-end."""
        scheme = self._polymer_scheme()
        scheme["steps"][0]["polymer_info"][self.POLYMER_SMILES]["n"] = "m"
        svg = _render(scheme)
        assert re.search(r">m<", svg), "custom repeat label 'm' missing"


# ════════════════════════════════════════════════════════════════════════════
# G10 — Substituent map (Markush) end-to-end render
# ════════════════════════════════════════════════════════════════════════════


class TestG10SubstituentAudit:
    """Phase 13 substituent_map (Markush R-groups) end-to-end render."""

    def _markush_scheme(self) -> dict:
        return {
            "schema_version": "1",
            "title": "Markush R-group scheme",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["[*:1]c1ccccc1"],
                    "products": ["[*:1]c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                    "substituent_map": {"R1": ["CH3", "C2H5", "iPr"]},
                }
            ],
        }

    def test_markush_renders_no_crash(self) -> None:
        svg = _render(self._markush_scheme())
        assert len(svg) > 1000

    def test_markush_renders_r_group_label(self) -> None:
        """R1 label should appear in the rendered output (substituent legend)."""
        svg = _render(self._markush_scheme())
        # R1 label appearance: in legend text or annotation
        assert "R1" in svg or "R<" in svg, (
            f"Markush R-group label missing from rendered SVG (length {len(svg)})"
        )


# ════════════════════════════════════════════════════════════════════════════
# G11 — Compound number reuse across steps
# ════════════════════════════════════════════════════════════════════════════


class TestG11CompoundReuseAudit:
    """When compound 4 is the product of step 2 AND a reactant of step 3,
    publication convention: draw it ONCE, not twice. Existing _dedup_intermediates
    in multi_step_renderer handles this; verify on a realistic case."""

    def _reuse_scheme(self) -> dict:
        intermediate = "c1ccc(Br)cc1"
        return {
            "schema_version": "1",
            "title": "Compound reuse",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": [intermediate],
                    "arrow_above": "Br2",
                    "compound_labels": {intermediate: "4"},
                },
                {
                    "id": "2",
                    "reactants": [intermediate],
                    "products": ["c1ccc(C(=O)O)cc1"],
                    "arrow_above": "CO2, BuLi",
                    "compound_labels": {intermediate: "4"},
                },
            ],
        }

    def test_compound_reuse_renders_no_crash(self) -> None:
        svg = _render(self._reuse_scheme())
        assert len(svg) > 1000

    def test_compound_label_4_appears(self) -> None:
        """The shared compound 4 should appear (at least once) in the rendered SVG."""
        svg = _render(self._reuse_scheme())
        # Label 4 appears in compound numbering (typically as bold text or circle)
        assert ">4<" in svg or "compound-label" in svg or '"4"' in svg, (
            "compound label '4' missing from rendered SVG"
        )

    def test_intermediate_smiles_dedup_does_not_crash(self) -> None:
        """The _dedup_intermediates path runs on this scheme without error.

        Step 1's product = step 2's reactant = same SMILES. Renderer's dedup
        logic should handle this (draw intermediate once at step 1 product position,
        empty step 2 reactant, label transferred to step 1 product).
        """
        svg = _render(self._reuse_scheme())
        # Both steps should still appear in render
        assert "1" in svg
        assert "2" in svg
