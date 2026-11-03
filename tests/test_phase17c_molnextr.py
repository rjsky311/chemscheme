"""Phase 17c MolNexTR add — closes S67 Part 5a gap (MARCUS-spec parity).

S67 Part 3 landscape scan + Part 5a verification: Phase 17c had
MolScribe + DECIMER + Mathpix-as-paid-tiebreaker but NOT MolNexTR
(MARCUS Oct 2025 RSC Digital Discovery ensemble includes it; 67% accuracy =
highest single-tool open-source). This adds MolNexTR following the
MolScribe/DECIMER class + docker-image pattern.

Tests follow tests/test_phase17c_ocsr_ensemble.py shape (subprocess + docker
mocked) — real container build is `docker build -t chemscheme-molnextr
docker/molnextr/`, not in pytest scope.

Citation: Chen et al. "MARCUS: Modular Architecture for Reaction Catalogue
Unification System." RSC Digital Discovery, Oct 2025.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chemscheme.recognition.protocols import MoleculeRecognizer

# ──────────────────────────── Recognizer class ────────────────────────────


class TestMolNexTRRecognizer:
    """MolNexTR subprocess + Docker adapter (parallels MolScribe/DECIMER)."""

    def test_class_importable(self) -> None:
        from chemscheme.recognition.ocsr import MolNexTRRecognizer

        assert MolNexTRRecognizer is not None

    def test_implements_protocol(self) -> None:
        from chemscheme.recognition.ocsr import MolNexTRRecognizer

        r = MolNexTRRecognizer()
        assert isinstance(r, MoleculeRecognizer)

    def test_recognize_with_mock_subprocess_returns_smiles(self) -> None:
        from chemscheme.recognition.ocsr import MolNexTRRecognizer

        r = MolNexTRRecognizer()
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"c1ccccc1\n",
                stderr="",
                returncode=0,
                elapsed=2.0,
            )
            result = r.recognize(b"fake_image_bytes")
            assert result.value == "c1ccccc1"
            assert result.source == "molnextr"
            assert result.confidence > 0

    def test_recognize_confidence_reflects_marcus_accuracy_ranking(self) -> None:
        """MolNexTR is MARCUS-best single-tool (67% accuracy); confidence MUST be
        ≥ MolScribe (0.8) AND ≥ DECIMER (0.75) to reflect that ranking in the
        ensemble's cross-validation weighting."""
        from chemscheme.recognition.ocsr import MolNexTRRecognizer

        r = MolNexTRRecognizer()
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"c1ccccc1\n", stderr="", returncode=0, elapsed=2.0
            )
            result = r.recognize(b"fake_image_bytes")
            assert result.confidence >= 0.80, (
                f"MolNexTR confidence {result.confidence} must be >= MolScribe (0.8)"
            )

    def test_recognize_tool_not_installed_returns_empty(self) -> None:
        from chemscheme.recognition.ocsr import MolNexTRRecognizer

        r = MolNexTRRecognizer()
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"",
                stderr="Command not found: molnextr",
                returncode=-1,
                elapsed=0.0,
            )
            result = r.recognize(b"fake_image")
            assert result.value == ""
            assert result.confidence == 0.0
            assert result.source == "molnextr"

    def test_recognize_invalid_smiles_returns_empty(self) -> None:
        """R2-01 pattern: invalid SMILES is failure, not low-confidence success."""
        from chemscheme.recognition.ocsr import MolNexTRRecognizer

        r = MolNexTRRecognizer()
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"C(C(C\n",  # unclosed paren
                stderr="",
                returncode=0,
                elapsed=2.0,
            )
            result = r.recognize(b"fake_image")
            assert result.value == ""
            assert result.confidence == 0.0

    def test_recognize_with_docker_mode(self) -> None:
        from chemscheme.recognition.ocsr import MolNexTRRecognizer

        r = MolNexTRRecognizer(use_docker=True)
        with patch("chemscheme.recognition.ocsr.run_tool_docker") as mock_docker:
            mock_docker.return_value = MagicMock(
                stdout=b"c1ccccc1\n", stderr="", returncode=0, elapsed=3.0
            )
            with patch(
                "chemscheme.recognition.ocsr.is_docker_ocsr_available",
                return_value=True,
            ):
                result = r.recognize(b"fake_image")
            assert result.value == "c1ccccc1"
            assert result.source == "molnextr"
            mock_docker.assert_called_once()
            # Verify tool name passed correctly
            args, kwargs = mock_docker.call_args
            assert args[0] == "molnextr"

    def test_default_docker_image_constant(self) -> None:
        """Recognizer default Docker image must use the runner-defined constant."""
        from chemscheme.recognition.ocsr import MolNexTRRecognizer
        from chemscheme.recognition.runner import _DOCKER_IMAGE_MOLNEXTR

        r = MolNexTRRecognizer()
        assert r._docker_image == _DOCKER_IMAGE_MOLNEXTR  # noqa: SLF001
        assert _DOCKER_IMAGE_MOLNEXTR == "chemscheme-molnextr"


# ──────────────────────────── runner.py constants ────────────────────────────


class TestRunnerMolNexTRSupport:
    """runner._VALID_TOOLS + _TOOL_DOCKER_IMAGES must include molnextr."""

    def test_molnextr_in_valid_tools(self) -> None:
        from chemscheme.recognition.runner import _VALID_TOOLS

        assert "molnextr" in _VALID_TOOLS

    def test_molnextr_in_tool_docker_images_map(self) -> None:
        from chemscheme.recognition.runner import _TOOL_DOCKER_IMAGES

        assert "molnextr" in _TOOL_DOCKER_IMAGES
        assert _TOOL_DOCKER_IMAGES["molnextr"] == "chemscheme-molnextr"

    def test_run_tool_docker_accepts_molnextr_tool_name(self) -> None:
        """run_tool_docker should not raise ValueError for `tool='molnextr'`."""
        from chemscheme.recognition.runner import run_tool_docker

        # Will fail at docker subprocess (no daemon assumed in tests), but
        # MUST NOT raise ValueError on the tool-name validation.
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"", stderr=b"docker daemon not reachable", returncode=1
            )
            # Should not raise — failures captured in returncode
            output = run_tool_docker("molnextr", b"fake_image")
            assert output.returncode != 0  # docker mock returns failure


# ──────────────────────────── ensemble integration ────────────────────────────


class TestEnsembleWithMolNexTR:
    """recognize_molecule should accept MolNexTR alongside MolScribe + DECIMER."""

    def test_ensemble_4_tools_agree(self) -> None:
        """When all 4 tools (MolScribe + DECIMER + MolNexTR + Mathpix) agree → HIGH agreement."""
        from chemscheme.models import AgreementLevel
        from chemscheme.recognition.ocsr import recognize_molecule

        class _Stub:
            def __init__(self, smiles: str, confidence: float, source: str):
                from chemscheme.models import RecognitionResult

                self._result = RecognitionResult(
                    value=smiles,
                    confidence=confidence,
                    source=source,
                    agreement=AgreementLevel.SINGLE,
                    sources=(source,),
                )

            def recognize(self, image: bytes, image_format: str = "png"):
                return self._result

        # All 4 agree on same SMILES
        recs = [
            _Stub("c1ccccc1", 0.80, "molscribe"),
            _Stub("c1ccccc1", 0.75, "decimer"),
            _Stub("c1ccccc1", 0.85, "molnextr"),
            _Stub("c1ccccc1", 0.90, "mathpix"),
        ]
        result = recognize_molecule(image=b"x", recognizers=recs)
        assert result.value == "c1ccccc1"
        assert result.agreement == AgreementLevel.HIGH
        # sources tuple contains all 4
        assert "molnextr" in result.sources

    def test_ensemble_molnextr_breaks_2_2_split(self) -> None:
        """When MolScribe+MolNexTR agree vs DECIMER differs → majority via MolNexTR."""
        from chemscheme.models import AgreementLevel
        from chemscheme.recognition.ocsr import recognize_molecule

        class _Stub:
            def __init__(self, smiles: str, confidence: float, source: str):
                from chemscheme.models import RecognitionResult

                self._result = RecognitionResult(
                    value=smiles,
                    confidence=confidence,
                    source=source,
                    agreement=AgreementLevel.SINGLE,
                    sources=(source,),
                )

            def recognize(self, image: bytes, image_format: str = "png"):
                return self._result

        recs = [
            _Stub("c1ccccc1", 0.80, "molscribe"),
            _Stub("CCC", 0.70, "decimer"),
            _Stub("c1ccccc1", 0.85, "molnextr"),
        ]
        result = recognize_molecule(image=b"x", recognizers=recs)
        # 2/3 agree on benzene → MEDIUM agreement
        assert result.value == "c1ccccc1"
        assert result.agreement in (AgreementLevel.MEDIUM, AgreementLevel.HIGH)


# ──────────────────────────── CLI extract integration ────────────────────────────


class TestCLIExtractIncludesMolNexTR:
    """`chemscheme extract` should auto-include MolNexTRRecognizer in its recognizer list.

    Closes S67 Part 5a gap operationally — without this wiring, adding the class
    alone wouldn't change `chemscheme extract` behavior.
    """

    def test_cli_extract_imports_molnextr_recognizer(self) -> None:
        """The CLI extract command must reference MolNexTRRecognizer in its build path."""
        from pathlib import Path

        cli_src = Path(__file__).parent.parent / "chemscheme" / "cli.py"
        assert "MolNexTRRecognizer" in cli_src.read_text(encoding="utf-8"), (
            "chemscheme/cli.py must reference MolNexTRRecognizer "
            "(extract command should auto-include it alongside MolScribe/DECIMER)"
        )

    def test_mcp_server_imports_molnextr_recognizer(self) -> None:
        """The MCP extract_image tool must reference MolNexTRRecognizer."""
        from pathlib import Path

        mcp_src = Path(__file__).parent.parent / "chemscheme" / "mcp_server.py"
        assert "MolNexTRRecognizer" in mcp_src.read_text(encoding="utf-8"), (
            "chemscheme/mcp_server.py must reference MolNexTRRecognizer "
            "(extract_image MCP tool should auto-include it)"
        )


# ──────────────────────────── Docker scaffolding present ────────────────────────────


class TestDockerScaffoldingPresent:
    """Production deployment requires docker/molnextr/ scaffolding (Dockerfile +
    entrypoint). Class + tests cover the Python side; the Dockerfile is the
    deployment artifact and must exist for `docker build -t chemscheme-molnextr
    docker/molnextr/` to work."""

    def test_dockerfile_exists(self) -> None:
        from pathlib import Path

        df = Path(__file__).parent.parent / "docker" / "molnextr" / "Dockerfile"
        assert df.exists(), f"docker/molnextr/Dockerfile missing: {df}"
        content = df.read_text(encoding="utf-8")
        assert "chemscheme-molnextr" in content
        # Must reference MolNexTR in some form
        assert "molnextr" in content.lower()

    def test_entrypoint_exists(self) -> None:
        from pathlib import Path

        ep = Path(__file__).parent.parent / "docker" / "molnextr" / "entrypoint.py"
        assert ep.exists(), f"docker/molnextr/entrypoint.py missing: {ep}"
        content = ep.read_text(encoding="utf-8")
        # Entrypoint must accept --image arg per existing pattern
        assert "--image" in content
