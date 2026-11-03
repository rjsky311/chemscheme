"""Phase 17f: PDF Figure Extraction tests.

BDD-first RED tests. PyMuPDF image extraction + figure classification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from chemscheme.recognition.pdf_extract import (  # noqa: E402
    ExtractedFigure,
    FigureClass,
    classify_figure_heuristic,
    extract_figures,
)

# ===========================================================================
# Helper: create a minimal valid PDF with an embedded image
# ===========================================================================


def _make_test_pdf(path: Path, *, width: int = 400, height: int = 300) -> None:
    """Create a minimal PDF with one embedded PNG image."""
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Create a simple image (solid color block)
    import struct
    import zlib

    # Minimal PNG: width x height, 8-bit RGB
    def _make_png(w: int, h: int) -> bytes:
        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        header = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        raw_data = b""
        for _ in range(h):
            raw_data += b"\x00" + b"\x80\x80\x80" * w  # gray pixels
        idat = _chunk(b"IDAT", zlib.compress(raw_data))
        iend = _chunk(b"IEND", b"")
        return header + ihdr + idat + iend

    img_bytes = _make_png(width, height)
    rect = fitz.Rect(50, 50, 50 + width, 50 + height)
    page.insert_image(rect, stream=img_bytes)
    doc.save(str(path))
    doc.close()


def _make_empty_pdf(path: Path) -> None:
    """Create a text-only PDF with no images."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a text-only PDF.")
    doc.save(str(path))
    doc.close()


# ===========================================================================
# AC-17f.1: PDF image extractor
# ===========================================================================


class TestExtractFigures:
    """extract_figures pulls images from PDF pages."""

    def test_extract_from_pdf_with_image(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        _make_test_pdf(pdf_path, width=400, height=300)
        figures = extract_figures(pdf_path)
        assert len(figures) >= 1
        fig = figures[0]
        assert isinstance(fig, ExtractedFigure)
        assert len(fig.image) > 0
        assert fig.page == 0

    def test_extract_respects_min_size(self, tmp_path: Path) -> None:
        """Small images below min_size are filtered out."""
        pdf_path = tmp_path / "small.pdf"
        _make_test_pdf(pdf_path, width=50, height=50)
        figures = extract_figures(pdf_path, min_size=(200, 100))
        assert len(figures) == 0

    def test_extract_from_text_only_pdf(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "text.pdf"
        _make_empty_pdf(pdf_path)
        figures = extract_figures(pdf_path)
        assert len(figures) == 0

    def test_extract_nonexistent_pdf(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_figures(Path("/nonexistent/file.pdf"))

    def test_extract_corrupt_file(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "corrupt.pdf"
        bad_path.write_bytes(b"not a pdf")
        with pytest.raises(fitz.FileDataError):
            extract_figures(bad_path)

    def test_extracted_figure_fields(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        _make_test_pdf(pdf_path)
        figures = extract_figures(pdf_path)
        assert len(figures) >= 1
        fig = figures[0]
        assert isinstance(fig.image, bytes)
        assert isinstance(fig.page, int)
        assert isinstance(fig.width, int)
        assert isinstance(fig.height, int)
        assert fig.width > 0
        assert fig.height > 0


# ===========================================================================
# AC-17f.2: Figure classifier (heuristic)
# ===========================================================================


class TestFigureClassifier:
    """Heuristic figure classification by aspect ratio and content."""

    def test_figure_class_enum(self) -> None:
        assert FigureClass.REACTION_SCHEME.value == "reaction_scheme"
        assert FigureClass.MOLECULE.value == "molecule"
        assert FigureClass.SPECTRUM.value == "spectrum"
        assert FigureClass.GRAPH.value == "graph"
        assert FigureClass.OTHER.value == "other"

    def test_wide_sparse_image_classified_as_scheme(self) -> None:
        """Wide, sparse, black-on-white → likely reaction scheme."""
        result = classify_figure_heuristic(width=800, height=300)
        assert result == FigureClass.REACTION_SCHEME

    def test_square_image_classified_as_other(self) -> None:
        """Square images are ambiguous."""
        result = classify_figure_heuristic(width=400, height=400)
        assert result in (FigureClass.OTHER, FigureClass.MOLECULE)

    def test_tall_narrow_image_classified_as_spectrum(self) -> None:
        """Tall, narrow → likely NMR spectrum or chromatogram."""
        result = classify_figure_heuristic(width=300, height=800)
        assert result == FigureClass.SPECTRUM

    def test_very_small_image_classified_as_other(self) -> None:
        result = classify_figure_heuristic(width=50, height=50)
        assert result == FigureClass.OTHER
