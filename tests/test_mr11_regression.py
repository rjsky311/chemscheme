"""Regression tests for MR11 review findings.

MR11-C1: arrow_above as list[str] should pass schema validation
MR11-C7: Mathpix non-JSON response should not crash
MR11-C9: Obstacle avoidance waypoints avoid box interior (in test_phase12_layout.py)
"""

import pytest

# ---------------------------------------------------------------------------
# MR11-C1: arrow_above accepts list[str] in schema validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMR11C1ArrowAboveList:
    """Schema should accept arrow_above as str or list[str]."""

    def test_arrow_above_string_valid(self):
        """arrow_above as string → no validation error."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": "NBS",
                },
            ],
        }
        errors = validate_scheme(scheme)
        assert not any("arrow_above" in e for e in errors)

    def test_arrow_above_list_valid(self):
        """arrow_above as list[str] → no validation error."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": ["Pd(PPh3)4", "K2CO3"],
                },
            ],
        }
        errors = validate_scheme(scheme)
        assert not any("arrow_above" in e for e in errors)

    def test_arrow_above_list_with_non_string_rejected(self):
        """arrow_above as list with non-string item → validation error."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": ["NBS", 42],
                },
            ],
        }
        errors = validate_scheme(scheme)
        assert any("arrow_above" in e for e in errors)

    def test_arrow_above_int_rejected(self):
        """arrow_above as integer → validation error."""
        from chemscheme.schema.scheme_schema import validate_scheme

        scheme = {
            "schema_version": "1",
            "steps": [
                {
                    "id": "1",
                    "reactants": ["C"],
                    "products": ["CC"],
                    "arrow_above": 42,
                },
            ],
        }
        errors = validate_scheme(scheme)
        assert any("arrow_above" in e for e in errors)


# ---------------------------------------------------------------------------
# MR11-C7: Mathpix non-JSON response graceful handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMR11C7MathpixNonJson:
    """Mathpix client should handle non-JSON error responses gracefully."""

    def test_non_json_response_returns_error(self):
        """HTML error response → MathpixResult with error, not crash."""
        from unittest.mock import MagicMock, patch

        from chemscheme.parsers.mathpix_client import MathpixClient

        client = MathpixClient(app_id="test", app_key="test")

        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.json.side_effect = ValueError("No JSON")

        with patch("chemscheme.parsers.mathpix_client.requests.post", return_value=mock_response):
            result = client.extract_text_from_base64("fake_b64")

        assert result.error is not None
        assert "non-JSON" in result.error or "502" in result.error
