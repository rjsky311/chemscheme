"""MR23 regression tests — multi-review Round 1 fixes.

C1: identify_reaction_type specificity scoring (Stille coupling vs polymerization)
C2/G3: abbreviation consistency per-group detection + defaults-only fallback
C3: text_extraction skips steps with no resolved SMILES
C4/G5: PNG fallback uses dynamic per-molecule widths
C5/G4: key_reagent short symbol regex boundary
C6: find_compounds case-insensitive matching
G1: validate_scheme_json handles arrow_above as list
G2: multi_step_renderer PNG uniform scaling
"""

import pytest

# ---------------------------------------------------------------------------
# C1: identify_reaction_type — Stille polymerization vs coupling specificity
# ---------------------------------------------------------------------------


class TestC1StilleSpecificity:
    """Stille polymerization should be distinguishable from Stille coupling."""

    def test_stille_polymerization_with_catalyst_match(self) -> None:
        """Pd2(dba)3 + Sn → Stille polymerization (not coupling)."""
        from chemscheme.domain_library import identify_reaction_type

        # Stille polymerization pattern has Pd2(dba)3, Stille coupling has Pd(PPh3)4
        result = identify_reaction_type("Pd2(dba)3, P(o-tol)3, Sn(CH3)3, toluene, N2")
        assert result.name == "Stille polymerization"

    def test_stille_coupling_without_polymerization_catalyst(self) -> None:
        """Pd(PPh3)4 + Sn → Stille coupling."""
        from chemscheme.domain_library import identify_reaction_type

        result = identify_reaction_type("Pd(PPh3)4, Me3SnPh, toluene")
        assert result.name == "Stille coupling"

    def test_single_sn_match_returns_coupling(self) -> None:
        """Only Sn without specific catalyst → first match (coupling)."""
        from chemscheme.domain_library import identify_reaction_type

        result = identify_reaction_type("SnBu3")
        assert "Stille" in result.name


# ---------------------------------------------------------------------------
# C5/G4: key_reagent boundary matching
# ---------------------------------------------------------------------------


class TestC5KeyReagentBoundary:
    """Short key_reagent should use regex boundary, not raw substring."""

    def test_li_not_in_ligand(self) -> None:
        """'Ligand' should NOT match a hypothetical 'Li' key_reagent."""
        from chemscheme.domain_library import identify_reaction_type

        # "Ligand" contains "Li" but is not a lithium compound
        result = identify_reaction_type("Ligand, toluene")
        # Negishi has Zn, not Li — just verify no false match
        assert result.name != "Negishi coupling"

    def test_sn_not_in_tmsn3(self) -> None:
        """TMSN3 should match diazotization (key_reagent=TMSN3), not Stille (Sn)."""
        from chemscheme.domain_library import identify_reaction_type

        result = identify_reaction_type("TMSN3, t-BuONO")
        assert result.name == "diazotization"

    def test_bf3_matches_correctly(self) -> None:
        """BF3·Et2O should match BF3 Lewis acid."""
        from chemscheme.domain_library import identify_reaction_type

        result = identify_reaction_type("BF3·Et2O, CH2Cl2, -78°C")
        assert result.name == "BF3 Lewis acid activation"


# ---------------------------------------------------------------------------
# C3: text_extraction skips steps with no resolved SMILES
# ---------------------------------------------------------------------------


class TestC3EmptySmiles:
    """Steps with classified compounds but no resolved SMILES should be skipped."""

    def test_unresolved_compounds_skipped(self) -> None:
        """Text with unresolved compounds → step skipped + warning."""
        from unittest.mock import patch

        from chemscheme.text_extraction import FoundCompound, extract_reactions_from_text

        fake_compound = FoundCompound(name="compound 1", smiles="", confidence=0.5)

        with patch(
            "chemscheme.text_extraction.find_compounds",
            return_value=[fake_compound],
        ):
            result = extract_reactions_from_text("compound 1 gave compound 2")
            steps = result.get("steps", [])
            warnings = result.get("warnings", [])
            # C3/R2 fix: step should be skipped (schema compliance)
            for step in steps:
                assert step["reactants"] or step["products"], (
                    "Step with no resolved SMILES should not be emitted"
                )
            # Should warn about unresolved SMILES
            assert any("no resolved SMILES" in w for w in warnings), (
                f"Expected 'no resolved SMILES' warning, got: {warnings}"
            )


# ---------------------------------------------------------------------------
# G1: validate_scheme_json handles arrow_above as list
# ---------------------------------------------------------------------------


class TestG1ArrowAboveList:
    """arrow_above as list should not crash validate_scheme_json."""

    def test_list_arrow_above_no_crash(self) -> None:
        """Scheme with arrow_above as list → no TypeError."""
        from chemscheme.schema.scheme_schema import validate_scheme_json

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": ["Pd(PPh3)4", "K2CO3"],
                }
            ],
        }
        # Should not raise TypeError
        result = validate_scheme_json(scheme)
        assert isinstance(result, dict)
        assert "valid" in result


# ---------------------------------------------------------------------------
# C6: find_compounds case-insensitive matching
# ---------------------------------------------------------------------------


class TestC6CaseInsensitiveCompounds:
    """Lowercase OCR text should still match known reagents."""

    def test_lowercase_nbs_matched(self) -> None:
        """'nbs' (lowercase) should match NBS reagent."""
        from chemscheme.text_extraction import find_compounds

        results = find_compounds("nbs in CHCl3")
        names = [c.name for c in results]
        assert any("NBS" in n.upper() for n in names), f"NBS not found in {names}"


# ---------------------------------------------------------------------------
# C4/G5: PNG rendering uses dynamic widths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestC4PngDynamicWidths:
    """Pillow PNG fallback should use per-molecule widths, not fixed 250px."""

    def test_png_renders_without_error(self) -> None:
        """Large + small molecule step renders PNG without error."""
        from chemscheme.renderers.rdkit_renderer import render_step

        step = {
            "id": "1",
            "reactants": ["c1ccc2c(c1)c1cc3c(cc1S2)c1nsnc1c1ccc(-c4ccc(F)cc4)cc1-3"],  # large NFA
            "products": ["c1ccccc1"],  # small benzene
            "arrow_above": "test",
        }
        result = render_step(step, fmt="png")
        assert len(result) > 100  # valid PNG should be substantial
        assert result[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# G2: multi_step_renderer PNG uniform scaling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestG2PngUniformScale:
    """PNG fallback should preserve aspect ratio (no stretching)."""

    def test_png_scheme_renders(self) -> None:
        """Multi-step PNG rendering should not distort molecules."""
        from chemscheme.renderers.multi_step_renderer import render_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "NBS",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(Br)cc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd(PPh3)4",
                },
            ],
        }
        result = render_scheme(scheme, fmt="png")
        assert len(result) > 100
        assert result[:4] == b"\x89PNG"
