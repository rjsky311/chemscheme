"""Phase 18b: Docker OCSR Image tests.

BDD-first RED tests for Docker runner, adapter wiring, and
Docker availability detection.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from chemscheme.recognition.runner import ToolOutput

# ===========================================================================
# AC-18b.3: Docker runner function
# ===========================================================================


class TestRunToolDocker:
    """run_tool_docker: subprocess isolation via Docker container."""

    def test_successful_docker_run(self) -> None:
        """Docker run returns SMILES on stdout."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch("chemscheme.recognition.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"CCO\n",
                stderr=b"",
                returncode=0,
            )
            result = run_tool_docker("molscribe", b"fake_png_data")
            assert isinstance(result, ToolOutput)
            assert result.returncode == 0
            assert b"CCO" in result.stdout

    def test_docker_not_available(self) -> None:
        """Docker not installed → graceful failure."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch(
            "chemscheme.recognition.runner.subprocess.run",
            side_effect=FileNotFoundError("docker"),
        ):
            result = run_tool_docker("molscribe", b"fake_png_data")
            assert result.returncode == -1
            assert "docker" in result.stderr.lower()

    def test_docker_timeout(self) -> None:
        """Container timeout → ToolOutput with error."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch(
            "chemscheme.recognition.runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["docker"], timeout=5),
        ):
            result = run_tool_docker("molscribe", b"fake_png_data", timeout=5)
            assert result.returncode == -1
            assert "timeout" in result.stderr.lower()

    def test_container_nonzero_exit(self) -> None:
        """Container exits with error code → captured in ToolOutput."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch("chemscheme.recognition.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=b"",
                stderr=b"Error: model not found\n",
                returncode=1,
            )
            result = run_tool_docker("decimer", b"fake_png_data")
            assert result.returncode == 1
            assert "model not found" in result.stderr

    def test_docker_image_name_default(self) -> None:
        """Default image name is chemscheme-ocsr."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch("chemscheme.recognition.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=b"CCO\n", stderr=b"", returncode=0)
            run_tool_docker("molscribe", b"fake_data")
            call_args = mock_run.call_args[0][0]
            assert "chemscheme-ocsr" in " ".join(call_args)

    def test_tool_argument_passed(self) -> None:
        """Tool name passed as --tool argument to container."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch("chemscheme.recognition.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=b"CCO\n", stderr=b"", returncode=0)
            run_tool_docker("decimer", b"fake_data")
            call_args = mock_run.call_args[0][0]
            cmd_str = " ".join(call_args)
            assert "--tool" in cmd_str
            assert "decimer" in cmd_str

    def test_custom_image_name(self) -> None:
        """Custom Docker image name can be specified."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch("chemscheme.recognition.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=b"CCO\n", stderr=b"", returncode=0)
            run_tool_docker("molscribe", b"fake_data", image="my-ocsr:v2")
            call_args = mock_run.call_args[0][0]
            assert "my-ocsr:v2" in " ".join(call_args)


# ===========================================================================
# AC-18b.4: Adapter wiring with Docker mode
# ===========================================================================


class TestAdapterDockerWiring:
    """MolScribe/DECIMER adapters with use_docker flag."""

    def test_molscribe_use_docker_flag(self) -> None:
        """MolScribeRecognizer accepts use_docker parameter."""
        from chemscheme.recognition.ocsr import MolScribeRecognizer

        r = MolScribeRecognizer(use_docker=True)
        assert r._use_docker is True

    def test_decimer_use_docker_flag(self) -> None:
        """DecimerRecognizer accepts use_docker parameter."""
        from chemscheme.recognition.ocsr import DecimerRecognizer

        r = DecimerRecognizer(use_docker=True)
        assert r._use_docker is True

    def test_molscribe_docker_calls_run_tool_docker(self) -> None:
        """Docker mode uses run_tool_docker instead of run_tool."""
        from chemscheme.recognition.ocsr import MolScribeRecognizer

        r = MolScribeRecognizer(use_docker=True)
        with patch("chemscheme.recognition.ocsr.run_tool_docker") as mock_docker:
            mock_docker.return_value = ToolOutput(
                stdout=b"CCO\n", stderr="", returncode=0, elapsed=2.0
            )
            result = r.recognize(b"fake_image")
            mock_docker.assert_called_once()
            assert result.value == "CCO"

    def test_decimer_docker_calls_run_tool_docker(self) -> None:
        """Docker mode uses run_tool_docker for DECIMER."""
        from chemscheme.recognition.ocsr import DecimerRecognizer

        r = DecimerRecognizer(use_docker=True)
        with patch("chemscheme.recognition.ocsr.run_tool_docker") as mock_docker:
            mock_docker.return_value = ToolOutput(
                stdout=b"c1ccccc1\n", stderr="", returncode=0, elapsed=3.0
            )
            result = r.recognize(b"fake_image")
            mock_docker.assert_called_once()
            assert result.value == "c1ccccc1"

    def test_non_docker_mode_uses_run_tool(self) -> None:
        """Default mode still uses run_tool (backward compatible)."""
        from chemscheme.recognition.ocsr import MolScribeRecognizer

        r = MolScribeRecognizer(use_docker=False)
        with patch("chemscheme.recognition.ocsr.run_tool") as mock_run:
            mock_run.return_value = MagicMock(stdout=b"CCO\n", stderr="", returncode=0, elapsed=1.0)
            r.recognize(b"fake_image")
            mock_run.assert_called_once()


# ===========================================================================
# AC-18b.4: Docker availability detection
# ===========================================================================


class TestDockerDetection:
    """Auto-detect Docker availability."""

    def test_is_docker_available_true(self) -> None:
        """Docker available when 'docker image inspect' succeeds."""
        from chemscheme.recognition.runner import is_docker_ocsr_available

        with patch("chemscheme.recognition.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert is_docker_ocsr_available() is True

    def test_is_docker_available_false_no_image(self) -> None:
        """Docker available but image not built → False."""
        from chemscheme.recognition.runner import is_docker_ocsr_available

        with patch("chemscheme.recognition.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert is_docker_ocsr_available() is False

    def test_is_docker_available_false_no_docker(self) -> None:
        """Docker not installed → False (no exception)."""
        from chemscheme.recognition.runner import is_docker_ocsr_available

        with patch(
            "chemscheme.recognition.runner.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert is_docker_ocsr_available() is False

    def test_env_var_forces_docker(self) -> None:
        """CHEMSCHEME_OCSR_DOCKER=1 forces Docker mode."""
        from chemscheme.recognition.ocsr import MolScribeRecognizer

        with patch.dict(os.environ, {"CHEMSCHEME_OCSR_DOCKER": "1"}):
            r = MolScribeRecognizer()
            assert r._use_docker is True


# ===========================================================================
# AC-18b.2: Entrypoint script argument parsing
# ===========================================================================


class TestEntrypointParsing:
    """Entrypoint wrapper script argument validation."""

    def test_entrypoint_exists(self) -> None:
        """docker/ocsr/entrypoint.py exists."""
        from pathlib import Path

        ep = Path(__file__).parent.parent / "docker" / "ocsr" / "entrypoint.py"
        assert ep.exists()

    def test_dockerfile_exists(self) -> None:
        """docker/ocsr/Dockerfile exists."""
        from pathlib import Path

        df = Path(__file__).parent.parent / "docker" / "ocsr" / "Dockerfile"
        assert df.exists()


# ===========================================================================
# Adversarial / edge cases
# ===========================================================================


class TestDockerRunnerEdgeCases:
    """Adversarial and edge cases for Docker runner."""

    def test_empty_image_bytes(self) -> None:
        """Empty image bytes → still calls Docker (let container handle error)."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch("chemscheme.recognition.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=b"", stderr=b"Empty input", returncode=1)
            result = run_tool_docker("molscribe", b"")
            assert result.returncode == 1

    def test_invalid_tool_name(self) -> None:
        """Invalid tool name → ValueError."""
        from chemscheme.recognition.runner import run_tool_docker

        with pytest.raises(ValueError, match="tool"):
            run_tool_docker("invalid_tool", b"data")

    def test_tool_docker_images_mapping(self) -> None:
        """Per-tool Docker image mapping is defined."""
        from chemscheme.recognition.runner import _TOOL_DOCKER_IMAGES

        assert _TOOL_DOCKER_IMAGES["molscribe"] == "chemscheme-molscribe"
        assert _TOOL_DOCKER_IMAGES["decimer"] == "chemscheme-decimer"

    def test_temp_file_cleanup_on_success(self) -> None:
        """Temp files cleaned up after successful Docker run."""
        from chemscheme.recognition.runner import run_tool_docker

        with patch("chemscheme.recognition.runner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=b"CCO\n", stderr=b"", returncode=0)
            # Should not leave temp files around
            import tempfile
            from pathlib import Path

            before = set(Path(tempfile.gettempdir()).glob("chemscheme_ocsr_*"))
            run_tool_docker("molscribe", b"fake_data")
            after = set(Path(tempfile.gettempdir()).glob("chemscheme_ocsr_*"))
            assert after == before  # no leftover files


# ===========================================================================
# Docker image split: per-tool images + legacy fallback
# ===========================================================================


class TestPerToolDockerImages:
    """Dedicated Docker images per OCSR tool (split from 8.3GB combined)."""

    def test_molscribe_default_docker_image(self) -> None:
        """MolScribeRecognizer defaults to chemscheme-molscribe image."""
        from chemscheme.recognition.ocsr import MolScribeRecognizer

        r = MolScribeRecognizer(use_docker=True)
        assert r._docker_image == "chemscheme-molscribe"

    def test_decimer_default_docker_image(self) -> None:
        """DecimerRecognizer defaults to chemscheme-decimer image."""
        from chemscheme.recognition.ocsr import DecimerRecognizer

        r = DecimerRecognizer(use_docker=True)
        assert r._docker_image == "chemscheme-decimer"

    def test_molscribe_custom_docker_image(self) -> None:
        """MolScribeRecognizer accepts custom docker_image parameter."""
        from chemscheme.recognition.ocsr import MolScribeRecognizer

        r = MolScribeRecognizer(use_docker=True, docker_image="my-molscribe:v3")
        assert r._docker_image == "my-molscribe:v3"

    def test_decimer_custom_docker_image(self) -> None:
        """DecimerRecognizer accepts custom docker_image parameter."""
        from chemscheme.recognition.ocsr import DecimerRecognizer

        r = DecimerRecognizer(use_docker=True, docker_image="my-decimer:latest")
        assert r._docker_image == "my-decimer:latest"

    def test_molscribe_passes_image_to_docker(self) -> None:
        """MolScribe Docker mode passes dedicated image name to run_tool_docker."""
        from chemscheme.recognition.ocsr import MolScribeRecognizer

        r = MolScribeRecognizer(use_docker=True)
        with (
            patch("chemscheme.recognition.ocsr.is_docker_ocsr_available", return_value=True),
            patch("chemscheme.recognition.ocsr.run_tool_docker") as mock_docker,
        ):
            mock_docker.return_value = ToolOutput(
                stdout=b"CCO\n", stderr="", returncode=0, elapsed=2.0
            )
            r.recognize(b"fake_image")
            _, kwargs = mock_docker.call_args
            assert kwargs.get("image") == "chemscheme-molscribe"

    def test_decimer_passes_image_to_docker(self) -> None:
        """DECIMER Docker mode passes dedicated image name to run_tool_docker."""
        from chemscheme.recognition.ocsr import DecimerRecognizer

        r = DecimerRecognizer(use_docker=True)
        with (
            patch("chemscheme.recognition.ocsr.is_docker_ocsr_available", return_value=True),
            patch("chemscheme.recognition.ocsr.run_tool_docker") as mock_docker,
        ):
            mock_docker.return_value = ToolOutput(
                stdout=b"c1ccccc1\n", stderr="", returncode=0, elapsed=3.0
            )
            r.recognize(b"fake_image")
            _, kwargs = mock_docker.call_args
            assert kwargs.get("image") == "chemscheme-decimer"


class TestDockerImageFallback:
    """Legacy fallback: use chemscheme-ocsr if dedicated image not available."""

    def test_molscribe_fallback_to_legacy(self) -> None:
        """If chemscheme-molscribe not found, fall back to chemscheme-ocsr."""
        from chemscheme.recognition.ocsr import MolScribeRecognizer

        r = MolScribeRecognizer(use_docker=True)

        def mock_available(image: str) -> bool:
            return image == "chemscheme-ocsr"

        with (
            patch(
                "chemscheme.recognition.ocsr.is_docker_ocsr_available", side_effect=mock_available
            ),
            patch("chemscheme.recognition.ocsr.run_tool_docker") as mock_docker,
        ):
            mock_docker.return_value = ToolOutput(
                stdout=b"CCO\n", stderr="", returncode=0, elapsed=2.0
            )
            r.recognize(b"fake_image")
            _, kwargs = mock_docker.call_args
            assert kwargs.get("image") == "chemscheme-ocsr"

    def test_decimer_fallback_to_legacy(self) -> None:
        """If chemscheme-decimer not found, fall back to chemscheme-ocsr."""
        from chemscheme.recognition.ocsr import DecimerRecognizer

        r = DecimerRecognizer(use_docker=True)

        def mock_available(image: str) -> bool:
            return image == "chemscheme-ocsr"

        with (
            patch(
                "chemscheme.recognition.ocsr.is_docker_ocsr_available", side_effect=mock_available
            ),
            patch("chemscheme.recognition.ocsr.run_tool_docker") as mock_docker,
        ):
            mock_docker.return_value = ToolOutput(
                stdout=b"c1ccccc1\n", stderr="", returncode=0, elapsed=3.0
            )
            r.recognize(b"fake_image")
            _, kwargs = mock_docker.call_args
            assert kwargs.get("image") == "chemscheme-ocsr"

    def test_no_fallback_when_dedicated_available(self) -> None:
        """Uses dedicated image when available (no fallback)."""
        from chemscheme.recognition.ocsr import MolScribeRecognizer

        r = MolScribeRecognizer(use_docker=True)

        def mock_available(image: str) -> bool:
            return image == "chemscheme-molscribe"

        with (
            patch(
                "chemscheme.recognition.ocsr.is_docker_ocsr_available", side_effect=mock_available
            ),
            patch("chemscheme.recognition.ocsr.run_tool_docker") as mock_docker,
        ):
            mock_docker.return_value = ToolOutput(
                stdout=b"CCO\n", stderr="", returncode=0, elapsed=2.0
            )
            r.recognize(b"fake_image")
            _, kwargs = mock_docker.call_args
            assert kwargs.get("image") == "chemscheme-molscribe"

    def test_no_images_available_uses_default(self) -> None:
        """If neither dedicated nor legacy available, passes default (let docker report error)."""
        from chemscheme.recognition.ocsr import DecimerRecognizer

        r = DecimerRecognizer(use_docker=True)
        with (
            patch("chemscheme.recognition.ocsr.is_docker_ocsr_available", return_value=False),
            patch("chemscheme.recognition.ocsr.run_tool_docker") as mock_docker,
        ):
            mock_docker.return_value = ToolOutput(
                stdout=b"", stderr="image not found", returncode=1, elapsed=0.5
            )
            r.recognize(b"fake_image")
            _, kwargs = mock_docker.call_args
            # Falls through to default (chemscheme-decimer) even if not available
            assert kwargs.get("image") == "chemscheme-decimer"


class TestSplitEntrypoints:
    """Split Docker entrypoint and Dockerfile existence checks."""

    def test_molscribe_entrypoint_exists(self) -> None:
        """docker/molscribe/entrypoint.py exists."""
        from pathlib import Path

        ep = Path(__file__).parent.parent / "docker" / "molscribe" / "entrypoint.py"
        assert ep.exists()

    def test_molscribe_dockerfile_exists(self) -> None:
        """docker/molscribe/Dockerfile exists."""
        from pathlib import Path

        df = Path(__file__).parent.parent / "docker" / "molscribe" / "Dockerfile"
        assert df.exists()

    def test_decimer_entrypoint_exists(self) -> None:
        """docker/decimer/entrypoint.py exists."""
        from pathlib import Path

        ep = Path(__file__).parent.parent / "docker" / "decimer" / "entrypoint.py"
        assert ep.exists()

    def test_decimer_dockerfile_exists(self) -> None:
        """docker/decimer/Dockerfile exists."""
        from pathlib import Path

        df = Path(__file__).parent.parent / "docker" / "decimer" / "Dockerfile"
        assert df.exists()

    def test_molscribe_entrypoint_no_tool_arg(self) -> None:
        """MolScribe entrypoint does not accept --tool (single-tool container)."""
        from pathlib import Path

        ep = Path(__file__).parent.parent / "docker" / "molscribe" / "entrypoint.py"
        content = ep.read_text()
        assert "--tool" not in content

    def test_decimer_entrypoint_no_tool_arg(self) -> None:
        """DECIMER entrypoint does not accept --tool (single-tool container)."""
        from pathlib import Path

        ep = Path(__file__).parent.parent / "docker" / "decimer" / "entrypoint.py"
        content = ep.read_text()
        assert "--tool" not in content


class TestDockerModelCacheOwnership:
    """Regression: model caches must be accessible by non-root USER (G2/C6 fix)."""

    @pytest.mark.parametrize(
        "subdir",
        ["molscribe", "decimer", "ocsr"],
    )
    def test_no_root_cache_in_runtime_stage(self, subdir: str) -> None:
        """Dockerfiles must not COPY model caches to /root/ in runtime stage.

        When USER is non-root, /root/ is inaccessible. Model caches must
        be placed under the appuser home directory instead.
        """
        import re
        from pathlib import Path

        df = Path(__file__).parent.parent / "docker" / subdir / "Dockerfile"
        if not df.exists():
            pytest.skip(f"docker/{subdir}/Dockerfile not found")
        content = df.read_text(encoding="utf-8")

        # Find runtime stage (after last FROM ... AS runtime)
        runtime_match = list(re.finditer(r"FROM\s+\S+\s+AS\s+runtime", content))
        if not runtime_match:
            pytest.skip("No runtime stage found")
        runtime_content = content[runtime_match[-1].start() :]

        # After USER appuser, there should be no reference to /root/.cache or /root/.data
        user_match = re.search(r"^USER\s+appuser", runtime_content, re.MULTILINE)
        if not user_match:
            pytest.skip("No USER appuser directive found")

        # Check: no COPY to /root/ paths in runtime stage
        root_copies = re.findall(r"COPY\s+--from=\S+\s+/root/\.\S+\s+/root/", runtime_content)
        assert root_copies == [], (
            f"docker/{subdir}/Dockerfile copies model cache to /root/ in runtime stage "
            f"(inaccessible by appuser): {root_copies}"
        )
