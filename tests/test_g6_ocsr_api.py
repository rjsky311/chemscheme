"""G6 — OCSR upload endpoint: paper screenshot → SMILES via Phase 17 ensemble.

Covers AC-G6.5 (drag-drop → recognize_molecule → editable canvas).

BDD scenarios: BDD-2 (Y6 screenshot → editable), BDD-5 (cat photo → low-conf warning),
BDD-6 (no docker → clear error), BDD-10 (API == CLI extract consistency).

Tests use mocked recognizers (matching tests/test_cli_extract.py pattern) — the
real Phase 17a-g pipeline is exercised by tests/test_e2e_pipeline.py and friends.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from chemscheme.models import AgreementLevel, RecognitionResult
from chemscheme.web.app import create_app


class _StubRecognizer:
    def __init__(self, smiles: str, confidence: float, source: str = "stub") -> None:
        self._smiles = smiles
        self._confidence = confidence
        self._source = source

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        return RecognitionResult(
            value=self._smiles,
            confidence=self._confidence,
            source=self._source,
            agreement=AgreementLevel.SINGLE,
            sources=(self._source,),
        )


class _RaisingRecognizer:
    """Simulates a recognizer that fails (e.g., docker not running)."""

    def recognize(self, image: bytes, image_format: str = "png") -> RecognitionResult:
        raise RuntimeError("docker daemon not reachable")


@pytest.fixture
def fake_png() -> bytes:
    """Minimal PNG bytes (8-byte signature + IHDR + IEND); enough to be 'an image'."""
    # 1x1 white PNG
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c63f8ffff3f0000050001a5f8a4730000000049454e"
        "44ae426082"
    )


def _client_with_recognizers(recs: list) -> TestClient:
    """Override the OCSR recognizer factory used by the /api/recognize endpoint."""
    from chemscheme.web import app as web_app

    app = create_app()

    def _factory() -> list:
        return recs

    app.dependency_overrides[web_app.get_recognizers] = _factory
    return TestClient(app)


# ──────────────────────────── AC-G6.5 / BDD-2 ────────────────────────────


def test_ocsr_endpoint_returns_smiles_high_confidence(fake_png) -> None:
    """Upload image → recognize_molecule called → SMILES + confidence returned."""
    recs = [_StubRecognizer("c1ccccc1", 0.95, "stub_high")]
    with _client_with_recognizers(recs) as client:
        r = client.post(
            "/api/recognize",
            files={"image": ("test.png", BytesIO(fake_png), "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["smiles"] == "c1ccccc1"
        assert body["confidence"] == pytest.approx(0.95)
        assert "source" in body
        assert body.get("needs_review") is False


def test_ocsr_endpoint_includes_mol_block(fake_png) -> None:
    """Endpoint also returns mol_block so client can load directly into Ketcher canvas."""
    recs = [_StubRecognizer("c1ccccc1", 0.95)]
    with _client_with_recognizers(recs) as client:
        r = client.post(
            "/api/recognize",
            files={"image": ("test.png", BytesIO(fake_png), "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "mol_block" in body
        assert "M  END" in body["mol_block"]


# ──────────────────────────── BDD-5 / low confidence ────────────────────────────


def test_ocsr_endpoint_low_confidence_flags_needs_review(fake_png) -> None:
    """Low-confidence recognition returns needs_review=True (UI shows manual fallback)."""
    recs = [_StubRecognizer("CCC", 0.20, "stub_low")]
    with _client_with_recognizers(recs) as client:
        r = client.post(
            "/api/recognize",
            files={"image": ("test.png", BytesIO(fake_png), "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["needs_review"] is True
        # UI uses this field to show a warning banner
        assert body["confidence"] < 0.5


def test_ocsr_endpoint_zero_confidence_returns_no_smiles(fake_png) -> None:
    """All recognizers fail → empty SMILES + needs_review (e.g., cat photo)."""
    recs = [_StubRecognizer("", 0.0, "stub_fail")]
    with _client_with_recognizers(recs) as client:
        r = client.post(
            "/api/recognize",
            files={"image": ("test.png", BytesIO(fake_png), "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["smiles"] == ""
        assert body["needs_review"] is True
        # Client uses LOW agreement to display "draw manually" CTA
        assert body.get("agreement") == AgreementLevel.LOW.value


# ──────────────────────────── BDD-6 / docker not available ────────────────────────────


def test_ocsr_endpoint_recognizer_raises_returns_503(fake_png) -> None:
    """If a recognizer raises (docker daemon dead), endpoint returns 503 + clear message."""
    recs = [_RaisingRecognizer()]
    with _client_with_recognizers(recs) as client:
        r = client.post(
            "/api/recognize",
            files={"image": ("test.png", BytesIO(fake_png), "image/png")},
        )
        assert r.status_code == 503
        body = r.json()
        msg = (body.get("detail") or body.get("error") or "").lower()
        # Message must give the user something actionable
        assert "ocsr" in msg or "docker" in msg or "recognizer" in msg


def test_ocsr_endpoint_no_recognizers_configured_returns_503() -> None:
    """No recognizers configured (e.g., image extra not installed) → 503 with hint."""
    fake_png = b"\x89PNG\r\n\x1a\n"  # minimal signature; never actually parsed
    with _client_with_recognizers([]) as client:
        r = client.post(
            "/api/recognize",
            files={"image": ("test.png", BytesIO(fake_png), "image/png")},
        )
        assert r.status_code == 503


# ──────────────────────────── Input validation ────────────────────────────


def test_ocsr_endpoint_missing_image_returns_422() -> None:
    """No image in multipart form → FastAPI 422, not 500."""
    recs = [_StubRecognizer("c1ccccc1", 0.9)]
    with _client_with_recognizers(recs) as client:
        r = client.post("/api/recognize", files={})
        assert r.status_code == 422


def test_ocsr_endpoint_empty_image_returns_400(fake_png) -> None:
    """Zero-byte upload → 400 (clearly user error, not server error)."""
    recs = [_StubRecognizer("c1ccccc1", 0.9)]
    with _client_with_recognizers(recs) as client:
        r = client.post(
            "/api/recognize",
            files={"image": ("empty.png", BytesIO(b""), "image/png")},
        )
        assert r.status_code == 400


# ──────────────────────────── BDD-10 / API == CLI consistency ────────────────────────────


def test_ocsr_api_uses_same_recognize_pipeline_as_cli_extract(fake_png) -> None:
    """API and CLI extract must call the same recognize_molecule() function on same inputs.

    Behavior-lock: the API MUST NOT diverge from CLI behavior — both go through
    chemscheme.recognition.ocsr.recognize_molecule, so identical input + identical
    recognizers → identical output. Without this, the GUI would silently diverge
    from the documented `chemscheme extract` pipeline.
    """
    from chemscheme.recognition.ocsr import recognize_molecule

    recs = [_StubRecognizer("c1ccccc1", 0.92)]
    cli_result = recognize_molecule(image=fake_png, recognizers=recs)

    with _client_with_recognizers(recs) as client:
        r = client.post(
            "/api/recognize",
            files={"image": ("test.png", BytesIO(fake_png), "image/png")},
        )
        assert r.status_code == 200
        api_body = r.json()

    assert api_body["smiles"] == cli_result.value
    assert api_body["confidence"] == pytest.approx(cli_result.confidence)
    assert api_body["source"] == cli_result.source
