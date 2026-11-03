"""OCSR ensemble: MolScribe + DECIMER + Mathpix tiebreaker.

Orchestrates molecule recognition with cross-validation.
ML tools invoked via subprocess (Python 3.10 isolation) or Docker container.
FPSim2/StructureLookup short-circuit for known structures.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

from rdkit import Chem

from chemscheme.models import AgreementLevel, RecognitionResult
from chemscheme.recognition.cross_validator import cross_validate, smiles_agree
from chemscheme.recognition.lookup import StructureLookup
from chemscheme.recognition.protocols import MoleculeRecognizer
from chemscheme.recognition.runner import (
    _DOCKER_IMAGE,
    _DOCKER_IMAGE_DECIMER,
    _DOCKER_IMAGE_MOLNEXTR,
    _DOCKER_IMAGE_MOLSCRIBE,
    is_docker_ocsr_available,
    run_tool,
    run_tool_docker,
)

logger = logging.getLogger(__name__)


def _validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid via RDKit."""
    if not smiles or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles.strip()) is not None


def _make_result(value: str, confidence: float, source: str) -> RecognitionResult:
    """Create a single-source RecognitionResult."""
    return RecognitionResult(
        value=value.strip(),
        confidence=confidence,
        source=source,
        agreement=AgreementLevel.SINGLE,
        sources=(source,),
    )


def _empty_result(source: str) -> RecognitionResult:
    """Create a zero-confidence result for failures."""
    return RecognitionResult(
        value="",
        confidence=0.0,
        source=source,
        agreement=AgreementLevel.SINGLE,
        sources=(source,),
    )


def _should_use_docker() -> bool:
    """Check if Docker mode is forced via environment variable."""
    return os.environ.get("CHEMSCHEME_OCSR_DOCKER", "").strip() in ("1", "true", "yes")


class MolScribeRecognizer:
    """MolScribe OCSR adapter via subprocess or Docker.

    Expects `molscribe` CLI or a Python script wrapper in PATH
    or configured via MOLSCRIBE_CMD environment variable.
    When use_docker=True, runs inside the chemscheme-molscribe container
    (falls back to legacy chemscheme-ocsr if the dedicated image is unavailable).
    """

    def __init__(
        self,
        cmd: str = "molscribe",
        use_docker: bool | None = None,
        docker_image: str = _DOCKER_IMAGE_MOLSCRIBE,
    ) -> None:
        self._cmd = cmd
        self._use_docker = use_docker if use_docker is not None else _should_use_docker()
        self._docker_image = docker_image

    def _resolve_docker_image(self) -> str:
        """Return dedicated image if available, else fall back to legacy."""
        if is_docker_ocsr_available(self._docker_image):
            return self._docker_image
        if self._docker_image != _DOCKER_IMAGE and is_docker_ocsr_available(_DOCKER_IMAGE):
            logger.info(
                "Dedicated image %s not found, falling back to %s",
                self._docker_image,
                _DOCKER_IMAGE,
            )
            return _DOCKER_IMAGE
        return self._docker_image  # let run_tool_docker report the error

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        """Recognize molecule from image bytes."""
        if self._use_docker:
            output = run_tool_docker(
                "molscribe",
                image,
                image=self._resolve_docker_image(),
                image_format=image_format,
            )
        else:
            with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as tmp:
                tmp.write(image)
                tmp_path = tmp.name

            try:
                output = run_tool(
                    [self._cmd, "--image", tmp_path],
                    timeout=120,
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        if output.returncode != 0:
            logger.warning("MolScribe failed: %s", output.stderr)
            return _empty_result("molscribe")

        smiles = output.stdout.decode("utf-8", errors="replace").strip()
        if not _validate_smiles(smiles):
            # R2-01 fix: invalid SMILES is a failure, not low-confidence success
            logger.warning("MolScribe returned invalid SMILES: %r", smiles)
            return _empty_result("molscribe")

        return _make_result(smiles, 0.8, "molscribe")


class DecimerRecognizer:
    """DECIMER OCSR adapter via subprocess or Docker.

    Expects `decimer` CLI or configured via DECIMER_CMD env variable.
    When use_docker=True, runs inside the chemscheme-decimer container
    (falls back to legacy chemscheme-ocsr if the dedicated image is unavailable).
    """

    def __init__(
        self,
        cmd: str = "decimer",
        use_docker: bool | None = None,
        docker_image: str = _DOCKER_IMAGE_DECIMER,
    ) -> None:
        self._cmd = cmd
        self._use_docker = use_docker if use_docker is not None else _should_use_docker()
        self._docker_image = docker_image

    def _resolve_docker_image(self) -> str:
        """Return dedicated image if available, else fall back to legacy."""
        if is_docker_ocsr_available(self._docker_image):
            return self._docker_image
        if self._docker_image != _DOCKER_IMAGE and is_docker_ocsr_available(_DOCKER_IMAGE):
            logger.info(
                "Dedicated image %s not found, falling back to %s",
                self._docker_image,
                _DOCKER_IMAGE,
            )
            return _DOCKER_IMAGE
        return self._docker_image  # let run_tool_docker report the error

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        """Recognize molecule from image bytes."""
        if self._use_docker:
            output = run_tool_docker(
                "decimer",
                image,
                image=self._resolve_docker_image(),
                image_format=image_format,
            )
        else:
            with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as tmp:
                tmp.write(image)
                tmp_path = tmp.name

            try:
                output = run_tool(
                    [self._cmd, "--image", tmp_path],
                    timeout=120,
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        if output.returncode != 0:
            logger.warning("DECIMER failed: %s", output.stderr)
            return _empty_result("decimer")

        smiles = output.stdout.decode("utf-8", errors="replace").strip()
        if not _validate_smiles(smiles):
            # R2-01 fix: invalid SMILES is a failure, not low-confidence success
            logger.warning("DECIMER returned invalid SMILES: %r", smiles)
            return _empty_result("decimer")

        return _make_result(smiles, 0.75, "decimer")


class MolNexTRRecognizer:
    """MolNexTR OCSR adapter via subprocess or Docker (S68 Phase 17c MARCUS-spec parity).

    MolNexTR (Chen et al., 2024) is a ConvNext + ViT + Transformer-decoder
    OCSR model; in the MARCUS ensemble (RSC Digital Discovery, Oct 2025) it is
    the highest single-tool open-source recognizer at 67% SMILES accuracy on
    real chemistry papers — above MolScribe and DECIMER. Confidence weight 0.85
    reflects this ranking for cross_validate() weighting.

    Expects ``molnextr`` CLI or ``MOLNEXTR_CMD`` environment variable. With
    use_docker=True, runs inside the chemscheme-molnextr container (built via
    ``docker build -t chemscheme-molnextr docker/molnextr/``); falls back to
    the legacy ``chemscheme-ocsr`` image only if the dedicated one is missing.
    """

    def __init__(
        self,
        cmd: str = "molnextr",
        use_docker: bool | None = None,
        docker_image: str = _DOCKER_IMAGE_MOLNEXTR,
    ) -> None:
        self._cmd = cmd
        self._use_docker = use_docker if use_docker is not None else _should_use_docker()
        self._docker_image = docker_image

    def _resolve_docker_image(self) -> str:
        """Return dedicated image if available, else fall back to legacy."""
        if is_docker_ocsr_available(self._docker_image):
            return self._docker_image
        if self._docker_image != _DOCKER_IMAGE and is_docker_ocsr_available(_DOCKER_IMAGE):
            logger.info(
                "Dedicated image %s not found, falling back to %s",
                self._docker_image,
                _DOCKER_IMAGE,
            )
            return _DOCKER_IMAGE
        return self._docker_image

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        """Recognize molecule from image bytes."""
        if self._use_docker:
            output = run_tool_docker(
                "molnextr",
                image,
                image=self._resolve_docker_image(),
                image_format=image_format,
            )
        else:
            with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as tmp:
                tmp.write(image)
                tmp_path = tmp.name
            try:
                output = run_tool(
                    [self._cmd, "--image", tmp_path],
                    timeout=180,
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        if output.returncode != 0:
            logger.warning("MolNexTR failed: %s", output.stderr)
            return _empty_result("molnextr")

        smiles = output.stdout.decode("utf-8", errors="replace").strip()
        if not _validate_smiles(smiles):
            logger.warning("MolNexTR returned invalid SMILES: %r", smiles)
            return _empty_result("molnextr")

        # MARCUS 2025: MolNexTR = highest single-tool accuracy (67%); confidence 0.85
        # MUST be >= MolScribe (0.8) per ensemble cross_validate weighting.
        return _make_result(smiles, 0.85, "molnextr")


class MathpixRecognizer:
    """Mathpix OCSR adapter wrapping existing MathpixClient.

    Only invoked as tiebreaker when MolScribe and DECIMER disagree.
    Requires MATHPIX_APP_ID and MATHPIX_APP_KEY environment variables.
    """

    def _get_client(self):  # type: ignore[no-untyped-def]
        """Lazy-load MathpixClient (avoids import error without API keys)."""
        from chemscheme.parsers.mathpix_client import MathpixClient

        return MathpixClient()

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        """Recognize molecule from image bytes via Mathpix API."""
        try:
            client = self._get_client()
        except Exception as exc:
            logger.warning("Mathpix client unavailable: %s", exc)
            return _empty_result("mathpix")

        image_b64 = base64.b64encode(image).decode("ascii")
        # Normalize MIME type: jpg → jpeg (RFC 2045)
        fmt = "jpeg" if image_format.lower() == "jpg" else image_format.lower()
        mime = f"image/{fmt}"

        try:
            mathpix_result = client.extract_text_from_base64(image_b64, mime_type=mime)
        except Exception as exc:
            logger.warning("Mathpix API error: %s", exc)
            return _empty_result("mathpix")

        if mathpix_result.error:
            logger.warning("Mathpix error: %s", mathpix_result.error)
            return _empty_result("mathpix")

        # C2 fix: also try extracting SMILES from text if smiles_list empty
        smiles_list = mathpix_result.smiles_list
        if not smiles_list and mathpix_result.text:
            from chemscheme.parsers.mathpix_client import _extract_smiles_from_text

            smiles_list = _extract_smiles_from_text(mathpix_result.text)
        if not smiles_list:
            return _empty_result("mathpix")

        smiles = smiles_list[0]
        confidence = mathpix_result.confidence if mathpix_result.confidence > 0 else 0.85
        return _make_result(smiles, confidence, "mathpix")


def recognize_molecule(
    image: bytes = b"",
    smiles_hint: str | None = None,
    lookup: StructureLookup | None = None,
    recognizers: list[MoleculeRecognizer] | None = None,
    tiebreaker: MoleculeRecognizer | None = None,
    *,
    image_format: str = "png",
) -> RecognitionResult:
    """Ensemble OCSR with FPSim2 short-circuit and cross-validation.

    Pipeline:
    1. If smiles_hint provided + lookup available → try local lookup
    2. Run all recognizers on image bytes → cross_validate with smiles_agree
    3. If LOW agreement + tiebreaker → add tiebreaker, re-validate
    4. Return merged result

    Args:
        image: Image bytes (PNG/JPEG) for recognition.
        smiles_hint: Optional SMILES for local lookup (from prior step).
        lookup: Optional StructureLookup database (Phase 17b).
        recognizers: List of MoleculeRecognizer instances.
        tiebreaker: Optional tiebreaker recognizer (e.g., Mathpix).
        image_format: Image format hint for recognizers (default: "png").

    Raises:
        ValueError: If no recognizers provided.
    """
    if not recognizers:
        msg = "At least one recognizer is required"
        raise ValueError(msg)

    # Step 1: Local lookup short-circuit (C4 fix: only on exact canonical match)
    if smiles_hint and lookup:
        lookup_result = lookup.lookup(smiles_hint)
        if lookup_result.confidence >= 0.99:  # exact match only
            return lookup_result

    # Step 2: Run all recognizers on actual image bytes (C1/G1 fix, C3-R2 format)
    results = [r.recognize(image, image_format) for r in recognizers]

    # Filter out zero-confidence failures
    valid_results = [r for r in results if r.confidence > 0]
    if not valid_results:
        return RecognitionResult(
            value="",
            confidence=max((r.confidence for r in results), default=0.0),
            source="ensemble",
            agreement=AgreementLevel.LOW,
            sources=tuple(r.source for r in results),
            needs_human_review=True,
        )

    if len(valid_results) == 1:
        return valid_results[0]

    # Step 3: Cross-validate
    merged = cross_validate(valid_results, smiles_agree)

    # Step 4: Tiebreaker if LOW agreement
    if merged.agreement == AgreementLevel.LOW and tiebreaker:
        tie_result = tiebreaker.recognize(image, image_format)
        if tie_result.confidence > 0:
            all_results = [*valid_results, tie_result]
            merged = cross_validate(all_results, smiles_agree)

    return merged
