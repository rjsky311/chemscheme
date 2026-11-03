"""C3-R2 regression tests: image_format passthrough in recognize_molecule().

MR20 deferred finding: recognize_molecule() did not forward image_format
to individual recognizers, causing JPEG images to be treated as PNG.

# RED-EXEMPT: review finding regression tests
"""

from __future__ import annotations

from unittest.mock import MagicMock

from chemscheme.models import AgreementLevel, RecognitionResult
from chemscheme.recognition.mocks import MockMoleculeRecognizer


class TestImageFormatPassthrough:
    """C3-R2: image_format flows from callers through to recognizers."""

    def test_recognize_molecule_accepts_image_format(self) -> None:
        """recognize_molecule() accepts image_format parameter."""
        from chemscheme.recognition.ocsr import recognize_molecule

        mock = MockMoleculeRecognizer(value="CCO", confidence=0.9)
        result = recognize_molecule(
            image=b"\xff\xd8\xff\xe0_fake_jpeg",
            image_format="jpeg",
            recognizers=[mock],
        )
        assert result.value == "CCO"

    def test_recognize_molecule_forwards_image_format_to_recognizers(self) -> None:
        """recognize_molecule() passes image_format to each recognizer.recognize()."""
        from chemscheme.recognition.ocsr import recognize_molecule

        mock_recognizer = MagicMock()
        mock_recognizer.recognize.return_value = RecognitionResult(
            value="CCO",
            confidence=0.9,
            source="mock",
            agreement=AgreementLevel.SINGLE,
            sources=("mock",),
        )

        recognize_molecule(
            image=b"\xff\xd8\xff\xe0_fake_jpeg",
            image_format="jpeg",
            recognizers=[mock_recognizer],
        )

        mock_recognizer.recognize.assert_called_once_with(b"\xff\xd8\xff\xe0_fake_jpeg", "jpeg")

    def test_recognize_molecule_forwards_format_to_tiebreaker(self) -> None:
        """image_format is forwarded to tiebreaker recognizer too."""
        from chemscheme.recognition.ocsr import recognize_molecule

        # Main recognizers disagree → LOW → tiebreaker invoked
        mock_a = MagicMock()
        mock_a.recognize.return_value = RecognitionResult(
            value="CCO",
            confidence=0.8,
            source="a",
            agreement=AgreementLevel.SINGLE,
            sources=("a",),
        )
        mock_b = MagicMock()
        mock_b.recognize.return_value = RecognitionResult(
            value="CCCO",
            confidence=0.8,
            source="b",
            agreement=AgreementLevel.SINGLE,
            sources=("b",),
        )
        mock_tie = MagicMock()
        mock_tie.recognize.return_value = RecognitionResult(
            value="CCO",
            confidence=0.85,
            source="tie",
            agreement=AgreementLevel.SINGLE,
            sources=("tie",),
        )

        recognize_molecule(
            image=b"img_data",
            image_format="jpeg",
            recognizers=[mock_a, mock_b],
            tiebreaker=mock_tie,
        )

        # Tiebreaker should receive image_format
        mock_tie.recognize.assert_called_once_with(b"img_data", "jpeg")

    def test_recognize_molecule_default_format_is_png(self) -> None:
        """When image_format is omitted, defaults to 'png'."""
        from chemscheme.recognition.ocsr import recognize_molecule

        mock_recognizer = MagicMock()
        mock_recognizer.recognize.return_value = RecognitionResult(
            value="CCO",
            confidence=0.9,
            source="mock",
            agreement=AgreementLevel.SINGLE,
            sources=("mock",),
        )

        recognize_molecule(
            image=b"\x89PNG_fake",
            recognizers=[mock_recognizer],
        )

        mock_recognizer.recognize.assert_called_once_with(b"\x89PNG_fake", "png")
