"""Tests for scheme_schema validation (AC-1.1, AC-1.2, AC-1.7, AC-1.8, AC-1.9).

BDD Feature: scheme.json validation
"""

import copy
import tempfile
from pathlib import Path

import pytest


@pytest.mark.unit
class TestSchemeSchemaValidation:
    """AC-1.1, AC-1.2: JSON Schema validates scheme.json correctly."""

    def test_valid_linear_scheme_passes(self, sample_scheme: dict) -> None:
        """Valid linear scheme passes validation with 0 errors."""
        from chemscheme.schema.scheme_schema import validate_scheme

        errors = validate_scheme(sample_scheme)
        assert errors == []

    def test_valid_convergent_scheme_passes(self, sample_scheme_convergent: dict) -> None:
        """Valid convergent scheme (with consumes_step_ids) passes validation."""
        from chemscheme.schema.scheme_schema import validate_scheme

        errors = validate_scheme(sample_scheme_convergent)
        assert errors == []

    def test_missing_products_fails(self, sample_scheme_invalid: dict) -> None:
        """Missing required field 'products' fails validation."""
        from chemscheme.schema.scheme_schema import validate_scheme

        errors = validate_scheme(sample_scheme_invalid)
        assert len(errors) > 0
        assert any("products" in e.lower() for e in errors)

    def test_missing_id_fails(self, sample_scheme: dict) -> None:
        """Missing required field 'id' fails validation."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = copy.deepcopy(sample_scheme)
        del scheme["steps"][0]["id"]
        errors = validate_scheme(scheme)
        assert len(errors) > 0
        assert any("id" in e.lower() for e in errors)

    def test_missing_reactants_fails(self, sample_scheme: dict) -> None:
        """Missing required field 'reactants' fails validation."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = copy.deepcopy(sample_scheme)
        del scheme["steps"][0]["reactants"]
        errors = validate_scheme(scheme)
        assert len(errors) > 0
        assert any("reactants" in e.lower() for e in errors)

    def test_missing_arrow_above_fails(self, sample_scheme: dict) -> None:
        """Missing required field 'arrow_above' fails validation."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = copy.deepcopy(sample_scheme)
        del scheme["steps"][0]["arrow_above"]
        errors = validate_scheme(scheme)
        assert len(errors) > 0
        assert any("arrow_above" in e.lower() for e in errors)

    def test_empty_steps_fails(self) -> None:
        """Scheme with empty steps list fails validation."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {"schema_version": "1", "title": "Empty", "steps": []}
        errors = validate_scheme(scheme)
        assert len(errors) > 0

    def test_missing_steps_fails(self) -> None:
        """Scheme without steps key fails validation."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {"schema_version": "1", "title": "No steps"}
        errors = validate_scheme(scheme)
        assert len(errors) > 0


@pytest.mark.unit
class TestSchemeSchemaGraphSemantics:
    """AC-1.1, AC-1.9: consumes_step_ids and branch_label support."""

    def test_consumes_step_ids_accepted(self, sample_scheme_convergent: dict) -> None:
        """consumes_step_ids is a valid optional field."""
        from chemscheme.schema.scheme_schema import validate_scheme

        errors = validate_scheme(sample_scheme_convergent)
        assert errors == []

    def test_branch_label_accepted(self, sample_scheme_convergent: dict) -> None:
        """branch_label is a valid optional field."""
        from chemscheme.schema.scheme_schema import validate_scheme

        # Verify the fixture has branch_label
        assert sample_scheme_convergent["steps"][0].get("branch_label") is not None
        errors = validate_scheme(sample_scheme_convergent)
        assert errors == []

    def test_consumes_step_ids_references_valid(self, sample_scheme_convergent: dict) -> None:
        """consumes_step_ids must reference existing step IDs."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = copy.deepcopy(sample_scheme_convergent)
        scheme["steps"][2]["consumes_step_ids"] = ["1a", "nonexistent"]
        errors = validate_scheme(scheme)
        assert len(errors) > 0
        assert any("nonexistent" in e.lower() for e in errors)


@pytest.mark.unit
class TestSchemeSchemaOptionalFields:
    """AC-1.1: Recommended optional fields are accepted."""

    def test_optional_fields_accepted(self, sample_scheme: dict) -> None:
        """All recommended optional fields pass validation."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = copy.deepcopy(sample_scheme)
        scheme["steps"][0].update(
            {
                "arrow_below": "DMF, 80 C",
                "yield": "78%",
                "reagents": ["Pd(PPh3)4"],
                "solvents": ["DMF"],
                "temperature": "80 C",
                "time": "12h",
                "atmosphere": "N2",
                "step_label": "i",
            }
        )
        errors = validate_scheme(scheme)
        assert errors == []

    def test_extra_field_accepted(self, sample_scheme: dict) -> None:
        """Unknown extra fields do not cause validation failure."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = copy.deepcopy(sample_scheme)
        scheme["steps"][0]["extra"] = {"custom_field": "test"}
        errors = validate_scheme(scheme)
        assert errors == []


@pytest.mark.unit
class TestSchemeSchemaVersion:
    """AC-1.8: schema_version field exists."""

    def test_schema_version_present(self, sample_scheme: dict) -> None:
        """Valid scheme has schema_version."""
        assert "schema_version" in sample_scheme

    def test_missing_schema_version_fails(self, sample_scheme: dict) -> None:
        """Missing schema_version fails validation."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = copy.deepcopy(sample_scheme)
        del scheme["schema_version"]
        errors = validate_scheme(scheme)
        assert len(errors) > 0
        assert any("schema_version" in e.lower() for e in errors)


@pytest.mark.unit
class TestSchemeRoundTrip:
    """AC-1.7: Round-trip preserves all fields including extra."""

    def test_round_trip_linear(self, sample_scheme: dict) -> None:
        """Load -> save -> reload preserves all fields."""
        from chemscheme.schema.scheme_schema import load_scheme, save_scheme

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheme.json"
            save_scheme(sample_scheme, path)
            reloaded = load_scheme(path)
            assert reloaded == sample_scheme

    def test_round_trip_convergent(self, sample_scheme_convergent: dict) -> None:
        """Round-trip preserves consumes_step_ids and branch_label."""
        from chemscheme.schema.scheme_schema import load_scheme, save_scheme

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheme.json"
            save_scheme(sample_scheme_convergent, path)
            reloaded = load_scheme(path)
            assert reloaded == sample_scheme_convergent

    def test_round_trip_extra_fields(self, sample_scheme: dict) -> None:
        """Round-trip preserves extra fields."""
        from chemscheme.schema.scheme_schema import load_scheme, save_scheme

        scheme = copy.deepcopy(sample_scheme)
        scheme["steps"][0]["extra"] = {"custom_field": "test", "nested": {"a": 1}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scheme.json"
            save_scheme(scheme, path)
            reloaded = load_scheme(path)
            assert reloaded["steps"][0]["extra"]["custom_field"] == "test"
            assert reloaded["steps"][0]["extra"]["nested"]["a"] == 1
