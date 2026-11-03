"""PDF figure extraction using PyMuPDF.

Extracts embedded images from PDF files and classifies them
by type (reaction scheme, molecule, spectrum, etc.) using heuristics.
Claude Vision classification can be added as a higher-accuracy option.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import fitz  # type: ignore[import-untyped,import-not-found]  # PyMuPDF

logger = logging.getLogger(__name__)


class FigureClass(Enum):
    """Classification of extracted figures."""

    REACTION_SCHEME = "reaction_scheme"
    MOLECULE = "molecule"
    SPECTRUM = "spectrum"
    GRAPH = "graph"
    PHOTO = "photo"
    OTHER = "other"


@dataclass(frozen=True)
class ExtractedFigure:
    """An image extracted from a PDF page."""

    image: bytes
    page: int
    width: int
    height: int
    figure_class: FigureClass = FigureClass.OTHER


def extract_figures(
    pdf_path: Path,
    min_size: tuple[int, int] = (200, 100),
) -> list[ExtractedFigure]:
    """Extract embedded images from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        min_size: Minimum (width, height) to include. Filters icons/logos.

    Returns:
        List of ExtractedFigure with image bytes and metadata.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        Exception: If PDF is corrupt or unreadable.
    """
    if not pdf_path.exists():
        msg = f"PDF not found: {pdf_path}"
        raise FileNotFoundError(msg)

    doc = fitz.open(str(pdf_path))
    figures: list[ExtractedFigure] = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    w = base_image["width"]
                    h = base_image["height"]
                except (ValueError, RuntimeError, KeyError):
                    # ValueError: invalid xref; RuntimeError: corrupt image data
                    # KeyError: missing image stream or incomplete dict
                    logger.warning(
                        "Failed to extract image xref=%d on page %d",
                        xref,
                        page_num,
                    )
                    continue

                # Filter by minimum size
                if w < min_size[0] or h < min_size[1]:
                    continue

                fig_class = classify_figure_heuristic(w, h)

                figures.append(
                    ExtractedFigure(
                        image=img_bytes,
                        page=page_num,
                        width=w,
                        height=h,
                        figure_class=fig_class,
                    )
                )
    finally:
        doc.close()

    return figures


def classify_figure_heuristic(width: int, height: int) -> FigureClass:
    """Classify figure type by aspect ratio heuristic.

    Reaction schemes tend to be wide and short.
    Spectra tend to be tall and narrow.
    Very small images are likely icons or decorations.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        FigureClass based on aspect ratio heuristic.
    """
    if width < 100 or height < 100:
        return FigureClass.OTHER

    aspect = width / height

    if aspect >= 2.0:
        # Wide, landscape → likely reaction scheme
        return FigureClass.REACTION_SCHEME
    elif aspect <= 0.5:
        # Tall, portrait → likely spectrum/chromatogram
        return FigureClass.SPECTRUM
    elif 0.8 <= aspect <= 1.2:
        # Square-ish → could be molecule or graph
        return FigureClass.MOLECULE
    else:
        return FigureClass.OTHER
