"""Mathpix API client for extracting text, formulas, and SMILES from images.

Used in the rendering pipeline to extract:
  - Reagent names and reaction conditions (text OCR)
  - Chemical structure SMILES (include_smiles=True via mpxpy SDK)
  - InChI identifiers (include_inchi=True)

Supports two backends:
  - mpxpy SDK (preferred): official Mathpix Python SDK with chemistry support
  - requests fallback: direct HTTP API calls

Credentials are read from environment variables:
  - MATHPIX_APP_ID
  - MATHPIX_APP_KEY
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests


class MathpixConfigError(Exception):
    """Raised when Mathpix API credentials are not configured."""


@dataclass
class MathpixResult:
    """Result from Mathpix text/chemistry extraction."""

    text: str = ""
    latex: str = ""
    confidence: float = 0.0
    error: str | None = None
    smiles_list: list[str] = field(default_factory=list)
    inchi_list: list[str] = field(default_factory=list)
    line_data: list[dict] = field(default_factory=list)  # type: ignore[type-arg]
    raw_response: dict = field(default_factory=dict)  # type: ignore[type-arg]


def _extract_smiles_from_text(text: str) -> list[str]:
    """Extract SMILES strings from Mathpix <smiles>...</smiles> tags."""
    return re.findall(r"<smiles[^>]*>([^<]+)</smiles>", text)


def _extract_inchi_from_text(text: str) -> list[str]:
    """Extract InChI strings from Mathpix inchi attributes."""
    return re.findall(r'inchi="([^"]+)"', text)


_API_URL = "https://api.mathpix.com/v3/text"


class MathpixClient:
    """Client for Mathpix OCR API with chemistry support."""

    def __init__(
        self,
        app_id: str | None = None,
        app_key: str | None = None,
    ) -> None:
        self.app_id = app_id or os.environ.get("MATHPIX_APP_ID", "")
        self.app_key = app_key or os.environ.get("MATHPIX_APP_KEY", "")

        if not self.app_id:
            raise MathpixConfigError(
                "MATHPIX_APP_ID not set. Set via environment variable or constructor."
            )
        if not self.app_key:
            raise MathpixConfigError(
                "MATHPIX_APP_KEY not set. Set via environment variable or constructor."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "Content-type": "application/json",
        }

    def extract_text_from_base64(
        self,
        image_b64: str,
        mime_type: str = "image/png",
        timeout: int = 30,
    ) -> MathpixResult:
        """Extract text/formulas from a base64-encoded image.

        Args:
            image_b64: Base64-encoded image data.
            mime_type: MIME type of the image.
            timeout: Request timeout in seconds.

        Returns:
            MathpixResult with extracted text, LaTeX, confidence, and any errors.
        """
        data = {
            "src": f"data:{mime_type};base64,{image_b64}",
            "formats": ["text", "latex_styled"],
        }

        try:
            resp = requests.post(
                _API_URL,
                json=data,
                headers=self._headers(),
                timeout=timeout,
            )
        except requests.RequestException as e:
            return MathpixResult(error=f"Request failed: {e}")

        try:
            body = resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            # MR11-C7: Non-JSON error response (HTML/plain text)
            return MathpixResult(error=f"HTTP {resp.status_code}: non-JSON response")

        if resp.status_code != 200:
            return MathpixResult(
                error=body.get("error", f"HTTP {resp.status_code}"),
                raw_response=body,
            )

        if "error" in body:
            return MathpixResult(
                error=body["error"],
                line_data=body.get("line_data", []),
                raw_response=body,
            )

        return MathpixResult(
            text=body.get("text", ""),
            latex=body.get("latex_styled", ""),
            confidence=body.get("confidence", 0.0),
            line_data=body.get("line_data", []),
            raw_response=body,
        )

    def extract_chemistry_from_file(
        self,
        image_path: Path,
        timeout: int = 30,
    ) -> MathpixResult:
        """Extract chemistry data (SMILES, InChI, text) from an image file.

        Uses mpxpy SDK for chemistry-specific features (include_smiles, include_inchi).
        Falls back to direct API if mpxpy is not available.

        Args:
            image_path: Path to the image file.
            timeout: Request timeout in seconds.

        Returns:
            MathpixResult with SMILES, InChI, text, and confidence.

        Raises:
            FileNotFoundError: If image file doesn't exist.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            from mpxpy import MathpixClient as MpxClient  # type: ignore[import-untyped]

            mpx = MpxClient(app_id=self.app_id, app_key=self.app_key)
            # MR15-G2: Pass timeout to SDK call (was previously dropped)
            image = mpx.image_new(
                file_path=str(image_path),
                include_smiles=True,
                include_inchi=True,
                include_line_data=True,
                formats=["text", "latex_styled"],
                timeout=timeout,
            )
            body = image.result
        except ImportError:
            # Fallback to direct API
            return self.extract_text_from_file(image_path, timeout=timeout)

        if "error" in body:
            return MathpixResult(
                error=body["error"],
                line_data=body.get("line_data", []),
                raw_response=body,
            )

        text = body.get("text", "")
        smiles_list = _extract_smiles_from_text(text)
        inchi_list = _extract_inchi_from_text(text)

        # Also extract from line_data entries
        for line in body.get("line_data", []):
            line_text = line.get("text", "")
            smiles_list.extend(_extract_smiles_from_text(line_text))
            inchi_list.extend(_extract_inchi_from_text(line_text))

        # Deduplicate
        smiles_list = list(dict.fromkeys(smiles_list))
        inchi_list = list(dict.fromkeys(inchi_list))

        return MathpixResult(
            text=text,
            latex=body.get("latex_styled", ""),
            confidence=body.get("confidence", 0.0),
            smiles_list=smiles_list,
            inchi_list=inchi_list,
            line_data=body.get("line_data", []),
            raw_response=body,
        )

    def extract_text_from_file(
        self,
        image_path: Path,
        timeout: int = 30,
    ) -> MathpixResult:
        """Extract text/formulas from an image file (basic, no chemistry).

        Args:
            image_path: Path to the image file (PNG, JPG, etc.).
            timeout: Request timeout in seconds.

        Returns:
            MathpixResult with extracted text.

        Raises:
            FileNotFoundError: If image file doesn't exist.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # MR13-G1: Guard against excessively large files (50 MB limit)
        file_size = image_path.stat().st_size
        max_image_size = 50 * 1024 * 1024  # 50 MB
        if file_size > max_image_size:
            raise ValueError(
                f"Image file too large ({file_size:,} bytes, "
                f"max {max_image_size:,}). Resize before processing."
            )

        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        suffix = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        mime_type = mime_map.get(suffix, "image/png")

        return self.extract_text_from_base64(image_b64, mime_type=mime_type, timeout=timeout)
