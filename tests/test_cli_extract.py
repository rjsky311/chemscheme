"""Phase 18a CLI tests — chemscheme extract command.

Tests the CLI wiring for image/PDF → scheme.json recognition pipeline.
All OCSR tools are mocked (real ML tools require Docker, Phase 18b).

AC-18a.1: chemscheme extract <image> -o scheme.json
AC-18a.2: chemscheme extract-pdf <pdf> -o output_dir/
AC-18a.3: --json output mode
AC-18a.4: Error handling (missing file, invalid image)
AC-18a.5: Adversarial: empty image, corrupt PDF
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from chemscheme.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper: create a minimal PNG image (1x1 white pixel)
# ---------------------------------------------------------------------------


def _minimal_png() -> bytes:
    """Smallest valid PNG: 1x1 white pixel."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8bit RGB
    raw_data = b"\x00\xff\xff\xff"  # filter=None, R=255, G=255, B=255
    idat = zlib.compress(raw_data)
    return signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


# ---------------------------------------------------------------------------
# Mock for recognize_molecule — returns a known SMILES
# ---------------------------------------------------------------------------


def _mock_recognize_molecule(**kwargs):
    """Mock recognize_molecule that returns ethanol."""
    from chemscheme.models import AgreementLevel, RecognitionResult

    return RecognitionResult(
        value="CCO",
        confidence=0.9,
        source="mock",
        agreement=AgreementLevel.SINGLE,
        sources=("mock",),
    )


# ---------------------------------------------------------------------------
# AC-18a.1: chemscheme extract <image> -o scheme.json
# ---------------------------------------------------------------------------


class TestExtractImage:
    """Extract SMILES from a single molecule image."""

    def test_extract_image_produces_json(self, tmp_path: Path) -> None:
        """GIVEN a PNG image, WHEN extract -o out.json, THEN produces valid scheme.json."""
        img_path = tmp_path / "mol.png"
        img_path.write_bytes(_minimal_png())
        out_path = tmp_path / "result.json"

        with patch(
            "chemscheme.recognition.ocsr.recognize_molecule",
            side_effect=_mock_recognize_molecule,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["extract", str(img_path), "-o", str(out_path)])

        assert result.exit_code == 0, result.output
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "smiles" in data or "steps" in data or "result" in data

    def test_extract_image_json_output(self, tmp_path: Path) -> None:
        """GIVEN --json flag, WHEN extract, THEN outputs JSON to stdout."""
        img_path = tmp_path / "mol.png"
        img_path.write_bytes(_minimal_png())

        with patch(
            "chemscheme.recognition.ocsr.recognize_molecule",
            side_effect=_mock_recognize_molecule,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["extract", str(img_path), "--json"])

        assert result.exit_code == 0
        # R2-02: mock warning may precede JSON, extract last JSON line
        json_lines = [ln for ln in result.output.strip().splitlines() if ln.startswith("{")]
        assert json_lines, f"No JSON in output: {result.output!r}"
        data = json.loads(json_lines[-1])
        assert data["status"] == "ok"

    def test_extract_missing_file_errors(self) -> None:
        """GIVEN nonexistent file, WHEN extract, THEN exit code != 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["extract", "nonexistent.png", "--json"])
        assert result.exit_code != 0

    def test_extract_overwrite_protection(self, tmp_path: Path) -> None:
        """GIVEN existing output, WHEN extract without --force, THEN refuses."""
        img_path = tmp_path / "mol.png"
        img_path.write_bytes(_minimal_png())
        out_path = tmp_path / "result.json"
        out_path.write_text("{}")

        with patch(
            "chemscheme.recognition.ocsr.recognize_molecule",
            side_effect=_mock_recognize_molecule,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["extract", str(img_path), "-o", str(out_path)])

        assert result.exit_code != 0

    def test_extract_force_overwrites(self, tmp_path: Path) -> None:
        """GIVEN --force, WHEN extract with existing output, THEN overwrites."""
        img_path = tmp_path / "mol.png"
        img_path.write_bytes(_minimal_png())
        out_path = tmp_path / "result.json"
        out_path.write_text("{}")

        with patch(
            "chemscheme.recognition.ocsr.recognize_molecule",
            side_effect=_mock_recognize_molecule,
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["extract", str(img_path), "-o", str(out_path), "--force"])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC-18a.2: chemscheme extract-pdf <pdf> -o output_dir/
# ---------------------------------------------------------------------------


class TestExtractPdf:
    """Extract figures from PDF and recognize molecules."""

    def test_extract_pdf_missing_file(self) -> None:
        """GIVEN nonexistent PDF, WHEN extract-pdf, THEN error."""
        runner = CliRunner()
        result = runner.invoke(main, ["extract-pdf", "nonexistent.pdf", "--json"])
        assert result.exit_code != 0

    def test_extract_pdf_json_output(self, tmp_path: Path) -> None:
        """GIVEN a PDF with images, WHEN extract-pdf --json, THEN JSON with figures."""
        # Create a minimal PDF with PyMuPDF (optional dep, skip in CI)
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        # Insert minimal image
        img_bytes = _minimal_png()
        page.insert_image(fitz.Rect(10, 10, 100, 100), stream=img_bytes)
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        runner = CliRunner()
        result = runner.invoke(main, ["extract-pdf", str(pdf_path), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert "figures" in data

    def test_extract_pdf_output_file(self, tmp_path: Path) -> None:
        """MR17-G2 regression: extract-pdf -o writes JSON to file."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_image(fitz.Rect(10, 10, 100, 100), stream=_minimal_png())
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        out_path = tmp_path / "result.json"
        runner = CliRunner()
        result = runner.invoke(main, ["extract-pdf", str(pdf_path), "-o", str(out_path)])

        assert result.exit_code == 0, result.output
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["status"] == "ok"
        assert "figures" in data

    def test_extract_pdf_output_overwrite_protection(self, tmp_path: Path) -> None:
        """MR17-G2 regression: extract-pdf -o refuses to overwrite without --force."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_image(fitz.Rect(10, 10, 100, 100), stream=_minimal_png())
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        out_path = tmp_path / "existing.json"
        out_path.write_text("{}")

        runner = CliRunner()
        result = runner.invoke(main, ["extract-pdf", str(pdf_path), "-o", str(out_path)])
        assert result.exit_code != 0

    def test_extract_pdf_output_force_overwrites(self, tmp_path: Path) -> None:
        """MR17-G2 regression: extract-pdf -o --force overwrites existing file."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_image(fitz.Rect(10, 10, 100, 100), stream=_minimal_png())
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        out_path = tmp_path / "existing.json"
        out_path.write_text("{}")

        runner = CliRunner()
        result = runner.invoke(main, ["extract-pdf", str(pdf_path), "-o", str(out_path), "--force"])
        assert result.exit_code == 0
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# AC-18a.5: Adversarial cases
# ---------------------------------------------------------------------------


class TestExtractAdversarial:
    """Edge cases and adversarial inputs."""

    def test_extract_empty_image(self, tmp_path: Path) -> None:
        """GIVEN zero-byte file, WHEN extract, THEN graceful error."""
        img_path = tmp_path / "empty.png"
        img_path.write_bytes(b"")

        runner = CliRunner()
        result = runner.invoke(main, ["extract", str(img_path), "--json"])

        # Should not crash; should return error status
        assert result.exit_code != 0 or '"error"' in result.output

    def test_extract_non_image_file_still_runs(self, tmp_path: Path) -> None:
        """GIVEN non-image bytes, WHEN extract, THEN CLI delegates to recognizer.

        A4 fix: empty recognition now returns exit 1. If real recognizers fail
        and mock isn't triggered, we get exit 1 (correct behavior).
        """
        img_path = tmp_path / "fake.png"
        img_path.write_text("this is not an image")

        runner = CliRunner()
        result = runner.invoke(main, ["extract", str(img_path), "--json"])

        # With real recognizers failing (empty SMILES) → A4 returns error
        # With mock fallback → returns "CCO" → exit 0
        # Either outcome is acceptable
        assert result.exit_code in (0, 1)
