"""Subprocess tool runner for external ML tools.

Provides the isolation boundary between ChemScheme (Python 3.14) and
ML tools requiring Python 3.10-3.12 (MolScribe, DECIMER, RxnIM).

Supports two modes:
- Direct subprocess: run_tool() — for locally installed tools
- Docker container: run_tool_docker() — for containerized ML tools (Phase 18b)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolOutput:
    """Result of running an external tool via subprocess."""

    stdout: bytes
    stderr: str
    returncode: int
    elapsed: float


def run_tool(
    cmd: list[str],
    input_data: bytes | None = None,
    timeout: int = 60,
) -> ToolOutput:
    """Run an external tool via subprocess with timeout and error handling.

    Args:
        cmd: Command and arguments to execute.
        input_data: Optional data to pipe to stdin.
        timeout: Maximum execution time in seconds.

    Returns:
        ToolOutput with stdout, stderr, returncode, and elapsed time.
        Never raises — errors are captured in returncode and stderr.
    """
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        return ToolOutput(
            stdout=result.stdout,
            stderr=result.stderr.decode("utf-8", errors="replace"),
            returncode=result.returncode,
            elapsed=elapsed,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.warning("Tool %s timed out after %ds", cmd[0], timeout)
        return ToolOutput(
            stdout=b"",
            stderr=f"Timeout after {timeout}s",
            returncode=-1,
            elapsed=elapsed,
        )
    except FileNotFoundError:
        elapsed = time.monotonic() - t0
        logger.warning("Tool not found: %s", cmd[0])
        return ToolOutput(
            stdout=b"",
            stderr=f"Command not found: {cmd[0]}",
            returncode=-1,
            elapsed=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.exception("Tool %s failed unexpectedly", cmd[0])
        return ToolOutput(
            stdout=b"",
            stderr=str(exc),
            returncode=-1,
            elapsed=elapsed,
        )


def is_tool_available(cmd: str) -> bool:
    """Check if a CLI tool is available in PATH (R2-02 fix)."""
    import shutil

    return shutil.which(cmd) is not None


# ---------------------------------------------------------------------------
# Docker-based tool runner (Phase 18b)
# ---------------------------------------------------------------------------

_DOCKER_IMAGE = "chemscheme-ocsr"  # Legacy combined image (deprecated)
_DOCKER_IMAGE_MOLSCRIBE = "chemscheme-molscribe"
_DOCKER_IMAGE_DECIMER = "chemscheme-decimer"
_DOCKER_IMAGE_MOLNEXTR = "chemscheme-molnextr"  # S68: MARCUS-spec parity (Phase 17c)
_VALID_TOOLS = {"molscribe", "decimer", "molnextr"}

# Map tool names to their dedicated Docker images
_TOOL_DOCKER_IMAGES: dict[str, str] = {
    "molscribe": _DOCKER_IMAGE_MOLSCRIBE,
    "decimer": _DOCKER_IMAGE_DECIMER,
    "molnextr": _DOCKER_IMAGE_MOLNEXTR,
}


def is_docker_ocsr_available(image: str = _DOCKER_IMAGE) -> bool:
    """Check if the OCSR Docker image is available locally.

    Returns True if ``docker image inspect <image>`` succeeds.
    Never raises — returns False on any error.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def run_tool_docker(
    tool: str,
    image_bytes: bytes,
    timeout: int = 180,
    image: str = _DOCKER_IMAGE,
    *,
    image_format: str = "png",
) -> ToolOutput:
    """Run an OCSR tool inside a Docker container.

    Writes image_bytes to a temp file, mounts it into the container,
    and captures SMILES output from stdout.

    Args:
        tool: Tool name ("molscribe" or "decimer").
        image_bytes: Image data (PNG/JPEG) to recognize.
        timeout: Maximum execution time in seconds.
        image: Docker image name (default: chemscheme-ocsr).
        image_format: Image format hint for file extension (default: "png").

    Returns:
        ToolOutput with SMILES on stdout (success) or error details.
        Never raises — errors are captured in returncode and stderr.

    Raises:
        ValueError: If tool name is not valid.
    """
    if tool not in _VALID_TOOLS:
        msg = f"Invalid tool: {tool!r}. Must be one of {_VALID_TOOLS}"
        raise ValueError(msg)

    # A3 fix: use correct file extension for image format
    _ext = "jpeg" if image_format.lower() == "jpg" else image_format.lower()
    _fname = f"input.{_ext}"

    # Write image bytes to a temp file for Docker volume mount
    tmp_dir = None
    t0 = time.monotonic()
    try:
        tmp_dir = tempfile.mkdtemp(prefix="chemscheme_ocsr_")
        tmp_path = Path(tmp_dir) / _fname
        tmp_path.write_bytes(image_bytes)

        # Build docker run command
        # Legacy combined image requires --tool <name> --image <path>
        # All other images (dedicated or custom) only accept --image
        # R2-03 fix: branch on legacy image, not whitelist
        if image != _DOCKER_IMAGE:
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{tmp_dir}:/data",
                image,
                "--image",
                f"/data/{_fname}",
            ]
        else:
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{tmp_dir}:/data",
                image,
                "--tool",
                tool,
                "--image",
                f"/data/{_fname}",
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        return ToolOutput(
            stdout=result.stdout,
            stderr=result.stderr.decode("utf-8", errors="replace"),
            returncode=result.returncode,
            elapsed=elapsed,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        logger.warning("Docker tool %s timed out after %ds", tool, timeout)
        return ToolOutput(
            stdout=b"",
            stderr=f"Timeout after {timeout}s",
            returncode=-1,
            elapsed=elapsed,
        )
    except FileNotFoundError:
        elapsed = time.monotonic() - t0
        logger.warning("Docker not found when running %s", tool)
        return ToolOutput(
            stdout=b"",
            stderr="Docker not found: is Docker installed and running?",
            returncode=-1,
            elapsed=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.exception("Docker tool %s failed unexpectedly", tool)
        return ToolOutput(
            stdout=b"",
            stderr=str(exc),
            returncode=-1,
            elapsed=elapsed,
        )
    finally:
        # Clean up temp directory
        if tmp_dir:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
