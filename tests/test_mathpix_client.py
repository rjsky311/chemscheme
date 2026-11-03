"""Tests for Mathpix API client — image text/formula extraction.

BDD Scenarios:
  GIVEN an image with chemical text, WHEN extract_text(), THEN returns recognized text
  GIVEN a valid image path, WHEN extract_text(), THEN API is called with correct headers
  GIVEN no API key configured, WHEN extract_text(), THEN raises ConfigError
  GIVEN API returns error, WHEN extract_text(), THEN returns error result gracefully
  GIVEN adversarial input (non-existent file), WHEN extract_text(), THEN raises FileNotFoundError
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestMathpixClientConfig:
    """Configuration and credential handling."""

    def test_missing_api_key_raises(self) -> None:
        """GIVEN no MATHPIX env vars, WHEN create client, THEN raises ConfigError."""
        from chemscheme.parsers.mathpix_client import MathpixClient, MathpixConfigError

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(MathpixConfigError, match="MATHPIX_APP_ID"),
        ):
            MathpixClient()

    def test_client_created_with_env_vars(self) -> None:
        """GIVEN valid env vars, WHEN create client, THEN succeeds."""
        from chemscheme.parsers.mathpix_client import MathpixClient

        with patch.dict(
            "os.environ",
            {"MATHPIX_APP_ID": "test_id", "MATHPIX_APP_KEY": "test_key"},
        ):
            client = MathpixClient()
            assert client.app_id == "test_id"


@pytest.mark.unit
class TestMathpixExtractText:
    """Text extraction from images via mocked API."""

    def _make_client(self) -> object:
        from chemscheme.parsers.mathpix_client import MathpixClient

        with patch.dict(
            "os.environ",
            {"MATHPIX_APP_ID": "test_id", "MATHPIX_APP_KEY": "test_key"},
        ):
            return MathpixClient()

    def test_extract_text_from_base64(self) -> None:
        """GIVEN base64 image data, WHEN extract_text, THEN calls API correctly."""
        from chemscheme.parsers.mathpix_client import MathpixResult

        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Pd(PPh3)4, K2CO3",
            "latex_styled": "\\text{Pd(PPh}_3\\text{)}_4\\text{, K}_2\\text{CO}_3",
            "confidence": 0.95,
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = client.extract_text_from_base64("fake_b64_data")
            assert isinstance(result, MathpixResult)
            assert result.text == "Pd(PPh3)4, K2CO3"
            assert result.confidence >= 0.9
            mock_post.assert_called_once()

    def test_extract_text_from_file(self, tmp_path: object) -> None:
        """GIVEN a PNG file, WHEN extract_text_from_file, THEN reads and calls API."""
        from chemscheme.parsers.mathpix_client import MathpixResult

        # Create a minimal PNG file
        png_file = tmp_path / "test.png"
        # Minimal valid PNG (1x1 pixel)
        png_file.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "NBS, AIBN",
            "confidence": 0.88,
        }

        with patch("requests.post", return_value=mock_response):
            result = client.extract_text_from_file(png_file)
            assert isinstance(result, MathpixResult)
            assert result.text == "NBS, AIBN"

    def test_nonexistent_file_raises(self) -> None:
        """GIVEN a non-existent file path, WHEN extract_text_from_file, THEN raises."""
        from pathlib import Path

        client = self._make_client()
        with pytest.raises(FileNotFoundError):
            client.extract_text_from_file(Path("/nonexistent/file.png"))

    def test_api_error_returns_error_result(self) -> None:
        """GIVEN API returns error, WHEN extract_text, THEN returns result with error."""
        from chemscheme.parsers.mathpix_client import MathpixResult

        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error": "Content not found",
            "error_info": {"id": "image_no_content", "message": "Content not found"},
        }

        with patch("requests.post", return_value=mock_response):
            result = client.extract_text_from_base64("fake_b64")
            assert isinstance(result, MathpixResult)
            assert result.error is not None
            assert result.text == ""

    def test_api_http_error_returns_error_result(self) -> None:
        """GIVEN API returns HTTP error, WHEN extract_text, THEN returns error result."""

        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Unauthorized"}

        with patch("requests.post", return_value=mock_response):
            result = client.extract_text_from_base64("fake_b64")
            assert result.error is not None


@pytest.mark.unit
class TestSmilesExtraction:
    """SMILES and InChI extraction from Mathpix response text."""

    def test_extract_smiles_from_tags(self) -> None:
        """GIVEN text with <smiles> tags, WHEN extracted, THEN returns SMILES list."""
        from chemscheme.parsers.mathpix_client import _extract_smiles_from_text

        text = '<smiles inchi="InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H">c1ccccc1</smiles>'
        result = _extract_smiles_from_text(text)
        assert result == ["c1ccccc1"]

    def test_extract_multiple_smiles(self) -> None:
        """GIVEN text with multiple <smiles> tags, WHEN extracted, THEN returns all."""
        from chemscheme.parsers.mathpix_client import _extract_smiles_from_text

        text = "<smiles>CC</smiles> and <smiles>CCC</smiles>"
        result = _extract_smiles_from_text(text)
        assert result == ["CC", "CCC"]

    def test_extract_inchi(self) -> None:
        """GIVEN text with inchi attributes, WHEN extracted, THEN returns InChI list."""
        from chemscheme.parsers.mathpix_client import _extract_inchi_from_text

        text = '<smiles inchi="InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H">c1ccccc1</smiles>'
        result = _extract_inchi_from_text(text)
        assert len(result) == 1
        assert result[0].startswith("InChI=1S/C6H6")

    def test_no_smiles_returns_empty(self) -> None:
        """GIVEN text without smiles tags, WHEN extracted, THEN returns empty list."""
        from chemscheme.parsers.mathpix_client import _extract_smiles_from_text

        result = _extract_smiles_from_text("plain text without chemistry")
        assert result == []


@pytest.mark.integration
class TestMathpixLive:
    """Live API integration tests — skipped without credentials."""

    @pytest.fixture(autouse=True)
    def _skip_without_credentials(self) -> None:
        import os

        if not os.environ.get("MATHPIX_APP_ID"):
            pytest.skip("MATHPIX_APP_ID not set")

    def test_live_api_responds(self) -> None:
        """GIVEN valid credentials, WHEN calling API, THEN get a response."""

        from chemscheme.parsers.mathpix_client import MathpixClient

        client = MathpixClient()
        # Minimal test image — 1x1 white PNG
        result = client.extract_text_from_base64(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAB"
            "Nl7BcQAAAABJRU5ErkJggg=="
        )
        # API should respond (may be error for blank image, but no exception)
        assert result is not None
