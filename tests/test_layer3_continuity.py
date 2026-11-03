"""Tests for Layer 3 scheme continuity (AC-6c.1 ~ AC-6c.4)."""

import pytest


# ---------------------------------------------------------------------------
# AC-6c.1  Linear step continuity (advisory)
# ---------------------------------------------------------------------------
class TestLinearContinuity:
    """AC-6c.1: Adjacent step product -> reactant similarity check."""

    @pytest.mark.unit
    @pytest.mark.chemistry
    def test_continuous_linear_scheme(self) -> None:
        """Product of step N appears as reactant of step N+1 -> no warnings."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccccc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd(PPh3)4",
                },
            ],
        }
        result = validate_continuity(scheme)
        assert result.layer == "continuity"
        assert result.errors == ()

    @pytest.mark.unit
    @pytest.mark.chemistry
    def test_discontinuous_linear_scheme_advisory(self) -> None:
        """No product-reactant overlap -> advisory warning (not error)."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(O)cc1"],
                    "arrow_above": "H2O2",
                },
                {
                    "id": "2",
                    "reactants": ["CCCC"],
                    "products": ["CCCCO"],
                    "arrow_above": "O2",
                },
            ],
        }
        result = validate_continuity(scheme)
        # Advisory warning, NOT error
        assert result.errors == ()
        assert len(result.warnings) > 0

    @pytest.mark.unit
    @pytest.mark.chemistry
    def test_single_step_no_continuity_check(self) -> None:
        """Single-step scheme has nothing to check -> no warnings."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                },
            ],
        }
        result = validate_continuity(scheme)
        assert result.errors == ()
        assert result.warnings == ()


# ---------------------------------------------------------------------------
# AC-6c.2  Branch/join dependency integrity
# ---------------------------------------------------------------------------
class TestBranchDependency:
    """AC-6c.2: consumes_step_ids dependency verification."""

    @pytest.mark.unit
    @pytest.mark.chemistry
    def test_valid_convergent_dependencies(self) -> None:
        """Valid consumes_step_ids references -> no errors."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1a",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                    "branch_label": "A",
                },
                {
                    "id": "1b",
                    "reactants": ["OB(O)c1ccccc1"],
                    "products": ["OB(O)c1ccc(C)cc1"],
                    "arrow_above": "CH3I",
                    "branch_label": "B",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccc(C)cc1"],
                    "products": ["Cc1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd(PPh3)4",
                    "consumes_step_ids": ["1a", "1b"],
                },
            ],
        }
        result = validate_continuity(scheme)
        assert result.errors == ()

    @pytest.mark.unit
    @pytest.mark.chemistry
    def test_invalid_consumes_step_ids_reference(self) -> None:
        """consumes_step_ids references nonexistent step -> error."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(Br)cc1"],
                    "products": ["c1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd",
                    "consumes_step_ids": ["1", "nonexistent"],
                },
            ],
        }
        result = validate_continuity(scheme)
        assert len(result.errors) > 0
        assert any("nonexistent" in e for e in result.errors)


# ---------------------------------------------------------------------------
# AC-6c.3  Missing dependency info -> warning, not block
# ---------------------------------------------------------------------------
class TestMissingDependency:
    """AC-6c.3: Missing dependency info produces warning, not error."""

    @pytest.mark.unit
    @pytest.mark.chemistry
    def test_multi_step_no_consumes_step_ids_warning(self) -> None:
        """Multi-step convergent-like scheme without consumes_step_ids -> warning."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1a",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                    "branch_label": "A",
                },
                {
                    "id": "1b",
                    "reactants": ["OB(O)c1ccccc1"],
                    "products": ["OB(O)c1ccc(C)cc1"],
                    "arrow_above": "CH3I",
                    "branch_label": "B",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(Br)cc1", "OB(O)c1ccc(C)cc1"],
                    "products": ["Cc1ccc(-c2ccccc2)cc1"],
                    "arrow_above": "Pd(PPh3)4",
                    # No consumes_step_ids — should warn
                },
            ],
        }
        result = validate_continuity(scheme)
        assert result.errors == ()
        # Advisory warning about missing dependency info
        assert len(result.warnings) > 0

    @pytest.mark.unit
    @pytest.mark.chemistry
    def test_post_merge_linear_step_no_false_positive(self) -> None:
        """V2-05: Linear steps after a proper merge should NOT get
        missing-dependency warnings. Only the actual merge point should warn
        if it lacks consumes_step_ids."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1a",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                    "branch_label": "A",
                },
                {
                    "id": "1b",
                    "reactants": ["c1ccncc1"],
                    "products": ["c1cc(Br)ncc1"],
                    "arrow_above": "NBS",
                    "branch_label": "B",
                },
                {
                    "id": "2",
                    "reactants": ["c1ccc(Br)cc1", "c1cc(Br)ncc1"],
                    "products": ["c1ccc(-c2ccncc2)cc1"],
                    "arrow_above": "Pd(PPh3)4",
                    "consumes_step_ids": ["1a", "1b"],
                },
                {
                    "id": "3",
                    "reactants": ["c1ccc(-c2ccncc2)cc1"],
                    "products": ["c1ccc(-c2ccncc2)c(O)c1"],
                    "arrow_above": "H2O2",
                },
                {
                    "id": "4",
                    "reactants": ["c1ccc(-c2ccncc2)c(O)c1"],
                    "products": ["c1ccc(-c2ccncc2)c(OC)c1"],
                    "arrow_above": "CH3I, K2CO3",
                },
            ],
        }
        result = validate_continuity(scheme)
        assert result.errors == ()
        # Steps 3 and 4 are post-merge linear steps — should NOT trigger
        # "merge step after branches" warning
        merge_warnings = [w for w in result.warnings if "merge step after branches" in w]
        assert len(merge_warnings) == 0, (
            f"Post-merge linear steps should not trigger merge warnings, got: {merge_warnings}"
        )


# ---------------------------------------------------------------------------
# AC-6c.4  Large NFA Tanimoto advisory
# ---------------------------------------------------------------------------
class TestLargeNFATanimoto:
    """AC-6c.4: Large NFA molecules get advisory-level Tanimoto."""

    @pytest.mark.unit
    @pytest.mark.chemistry
    def test_large_molecule_tanimoto_advisory(self) -> None:
        """Large NFA molecule continuity check is advisory only."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        # Use large NFA-style SMILES (simplified)
        large_mol = "c1ccc2c(c1)c1cc3c(cc1[nH]2)c1cc2c(cc1[nH]3)c1ccccc1[nH]2"
        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": [large_mol],
                    "products": [large_mol + "C"],  # slightly different
                    "arrow_above": "NBS",
                },
                {
                    "id": "2",
                    "reactants": ["CCCCCC"],  # completely different
                    "products": ["CCCCCCO"],
                    "arrow_above": "O2",
                },
            ],
        }
        result = validate_continuity(scheme)
        # Should be advisory (warnings), not errors
        assert result.errors == ()

    @pytest.mark.unit
    @pytest.mark.chemistry
    def test_result_has_layer_name(self) -> None:
        """Continuity result identifies its layer."""
        from chemscheme.validators.layer3_scheme import validate_continuity

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["c1ccccc1"],
                    "products": ["c1ccc(Br)cc1"],
                    "arrow_above": "Br2",
                },
            ],
        }
        result = validate_continuity(scheme)
        assert result.layer == "continuity"
