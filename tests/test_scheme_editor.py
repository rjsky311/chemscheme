"""Tests for scheme_editor: add/remove steps with re-validation.

Phase 7 — AC-7.1, AC-7.2, AC-7.3, AC-7.7.
BDD-first: all tests RED before implementation.
"""

import copy

import pytest


@pytest.mark.unit
class TestAddStep:
    """AC-7.1: Add a new step to a scheme."""

    def test_add_step_to_single_step_scheme(self, sample_scheme: dict) -> None:
        """Adding a step to a 1-step scheme yields 2 steps."""
        from chemscheme.editors.scheme_editor import add_step

        new_step = {
            "id": "2",
            "reactants": ["c1ccc(-c2ccccc2)cc1"],
            "products": ["C1CCC(C2CCCCC2)CC1"],
            "arrow_above": "H2, Pd/C",
            "arrow_below": "EtOH, rt, 6h",
        }
        result = add_step(sample_scheme, new_step)
        assert len(result["steps"]) == 2
        assert result["steps"][1]["id"] == "2"

    def test_add_step_preserves_existing_steps(self, sample_scheme: dict) -> None:
        """Original steps remain unchanged after adding."""
        from chemscheme.editors.scheme_editor import add_step

        original_step = copy.deepcopy(sample_scheme["steps"][0])
        new_step = {
            "id": "2",
            "reactants": ["c1ccc(-c2ccccc2)cc1"],
            "products": ["C1CCC(C2CCCCC2)CC1"],
            "arrow_above": "H2, Pd/C",
        }
        result = add_step(sample_scheme, new_step)
        assert result["steps"][0] == original_step

    def test_add_step_at_position(self, sample_scheme_3step: dict) -> None:
        """AC-7.1: Insert step at specific position (0-indexed)."""
        from chemscheme.editors.scheme_editor import add_step

        new_step = {
            "id": "1b",
            "reactants": ["c1ccc(-c2ccccc2)cc1"],
            "products": ["c1ccc(-c2ccc(O)cc2)cc1"],
            "arrow_above": "mCPBA",
        }
        result = add_step(sample_scheme_3step, new_step, position=1)
        assert len(result["steps"]) == 4
        assert result["steps"][1]["id"] == "1b"
        # Original step 2 should now be at position 2
        assert result["steps"][2]["id"] == "2"

    def test_add_step_validates_required_fields(self, sample_scheme: dict) -> None:
        """Adding a step missing required fields raises ValueError."""
        from chemscheme.editors.scheme_editor import add_step

        bad_step = {"id": "2", "reactants": ["C"]}  # missing products, arrow_above
        with pytest.raises(ValueError, match="required"):
            add_step(sample_scheme, bad_step)

    def test_add_step_rejects_duplicate_id(self, sample_scheme: dict) -> None:
        """Adding a step with an existing ID raises ValueError."""
        from chemscheme.editors.scheme_editor import add_step

        dup_step = {
            "id": "1",  # same as existing step
            "reactants": ["C"],
            "products": ["CC"],
            "arrow_above": "test",
        }
        with pytest.raises(ValueError, match="duplicate"):
            add_step(sample_scheme, dup_step)

    def test_add_step_preserves_metadata(self, sample_scheme: dict) -> None:
        """Top-level fields (title, style, layout) preserved after add."""
        from chemscheme.editors.scheme_editor import add_step

        new_step = {
            "id": "2",
            "reactants": ["C"],
            "products": ["CC"],
            "arrow_above": "test",
        }
        result = add_step(sample_scheme, new_step)
        assert result["title"] == sample_scheme["title"]
        assert result["style"] == sample_scheme["style"]
        assert result["schema_version"] == sample_scheme["schema_version"]


@pytest.mark.unit
class TestRemoveStep:
    """AC-7.2: Remove a step from a scheme."""

    def test_remove_step_by_id(self, sample_scheme_3step: dict) -> None:
        """Remove step 2 from a 3-step scheme yields 2 steps."""
        from chemscheme.editors.scheme_editor import remove_step

        result = remove_step(sample_scheme_3step, "2")
        assert len(result["steps"]) == 2
        ids = [s["id"] for s in result["steps"]]
        assert "2" not in ids

    def test_remove_step_preserves_other_steps(self, sample_scheme_3step: dict) -> None:
        """Remaining steps are unchanged."""
        from chemscheme.editors.scheme_editor import remove_step

        original_step1 = copy.deepcopy(sample_scheme_3step["steps"][0])
        original_step3 = copy.deepcopy(sample_scheme_3step["steps"][2])
        result = remove_step(sample_scheme_3step, "2")
        assert result["steps"][0] == original_step1
        assert result["steps"][1] == original_step3

    def test_remove_nonexistent_step_raises(self, sample_scheme: dict) -> None:
        """Removing a step that doesn't exist raises ValueError."""
        from chemscheme.editors.scheme_editor import remove_step

        with pytest.raises(ValueError, match="not found"):
            remove_step(sample_scheme, "999")

    def test_remove_last_step_raises(self, sample_scheme: dict) -> None:
        """Cannot remove the only step (scheme must have ≥1 step)."""
        from chemscheme.editors.scheme_editor import remove_step

        with pytest.raises(ValueError, match="at least one"):
            remove_step(sample_scheme, "1")

    def test_remove_step_cleans_consumes_step_ids(self, sample_scheme_convergent: dict) -> None:
        """AC-7.7: Removing a step cleans dangling consumes_step_ids refs."""
        from chemscheme.editors.scheme_editor import remove_step

        # Find a step that is referenced by consumes_step_ids
        # In convergent scheme, merge step consumes branch steps
        result = remove_step(sample_scheme_convergent, "1a")
        for step in result["steps"]:
            refs = step.get("consumes_step_ids", [])
            assert "1a" not in refs

    def test_remove_step_preserves_metadata(self, sample_scheme_3step: dict) -> None:
        """Top-level fields preserved after remove."""
        from chemscheme.editors.scheme_editor import remove_step

        result = remove_step(sample_scheme_3step, "2")
        assert result["title"] == sample_scheme_3step["title"]
        assert result["schema_version"] == sample_scheme_3step["schema_version"]


@pytest.mark.unit
class TestEditRevalidation:
    """AC-7.3: Edits automatically re-validate the scheme."""

    def test_add_step_returns_validation(self, sample_scheme: dict) -> None:
        """add_step returns (scheme, validation_result) when validate=True."""
        from chemscheme.editors.scheme_editor import add_step

        new_step = {
            "id": "2",
            "reactants": ["c1ccc(-c2ccccc2)cc1"],
            "products": ["C1CCC(C2CCCCC2)CC1"],
            "arrow_above": "H2, Pd/C",
        }
        result, validation = add_step(sample_scheme, new_step, validate=True)
        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "issues" in validation

    def test_remove_step_returns_validation(self, sample_scheme_3step: dict) -> None:
        """remove_step returns (scheme, validation_result) when validate=True."""
        from chemscheme.editors.scheme_editor import remove_step

        result, validation = remove_step(sample_scheme_3step, "2", validate=True)
        assert isinstance(validation, dict)
        assert "valid" in validation

    def test_add_invalid_smiles_flagged(self, sample_scheme: dict) -> None:
        """Adding a step with invalid SMILES is caught by re-validation."""
        from chemscheme.editors.scheme_editor import add_step

        bad_step = {
            "id": "2",
            "reactants": ["INVALID_SMILES_XXX"],
            "products": ["C"],
            "arrow_above": "test",
        }
        result, validation = add_step(sample_scheme, bad_step, validate=True)
        assert not validation["valid"]
        assert len(validation["issues"]) > 0


@pytest.mark.unit
class TestLayoutPreservation:
    """AC-7.7: Partial edits don't break other steps' layout."""

    def test_add_step_does_not_mutate_input(self, sample_scheme: dict) -> None:
        """Input scheme is not mutated (returns new dict)."""
        from chemscheme.editors.scheme_editor import add_step

        original = copy.deepcopy(sample_scheme)
        new_step = {
            "id": "2",
            "reactants": ["C"],
            "products": ["CC"],
            "arrow_above": "test",
        }
        add_step(sample_scheme, new_step)
        assert sample_scheme == original

    def test_remove_step_does_not_mutate_input(self, sample_scheme_3step: dict) -> None:
        """Input scheme is not mutated (returns new dict)."""
        from chemscheme.editors.scheme_editor import remove_step

        original = copy.deepcopy(sample_scheme_3step)
        remove_step(sample_scheme_3step, "2")
        assert sample_scheme_3step == original

    def test_add_step_extra_fields_preserved(self) -> None:
        """Extra/unknown fields in steps and top-level are preserved."""
        from chemscheme.editors.scheme_editor import add_step

        scheme = {
            "schema_version": "1",
            "custom_field": "keep_me",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "test",
                    "extra": {"custom": True},
                }
            ],
        }
        new_step = {
            "id": "2",
            "reactants": ["CC"],
            "products": ["CCC"],
            "arrow_above": "test2",
        }
        result = add_step(scheme, new_step)
        assert result["custom_field"] == "keep_me"
        assert result["steps"][0]["extra"] == {"custom": True}
