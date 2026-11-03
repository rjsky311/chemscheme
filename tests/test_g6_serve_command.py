"""G6 — chemscheme serve CLI command.

Covers AC-G6.1 (serve starts local web server, browser-loadable index)
and AC-G6.2 (port conflict detection with helpful error).

BDD scenarios covered: BDD-7 (port conflict).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from contextlib import closing

import httpx
import pytest
from click.testing import CliRunner

from chemscheme.cli import main as cli_main


def _free_port() -> int:
    """Bind to port 0, return the OS-assigned ephemeral port (immediately released)."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def serve_process():
    """Spawn `chemscheme serve --port N` in subprocess; tear down after."""
    procs: list[subprocess.Popen[bytes]] = []

    def _spawn(port: int, extra_args: tuple[str, ...] = ()) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(
            [sys.executable, "-m", "chemscheme.cli", "serve", "--port", str(port), *extra_args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(proc)
        return proc

    yield _spawn

    for p in procs:
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def _wait_for_server(port: int, timeout: float = 90.0) -> bool:
    """Poll the port until it responds OR timeout. Return True if up.

    Generous timeout: cold-start of `python -m chemscheme.cli` + FastAPI app
    construction + uvicorn bind takes 3-8s on Windows in isolation; under
    pre-commit-hook full-suite load (1356 tests + quality gate contention),
    observed cold-starts have crept past 60s — 90s gives headroom so the
    flake does not block commits on slow Windows runners.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=0.5) as client:
                r = client.get(f"http://127.0.0.1:{port}/api/health")
                if r.status_code == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


# ──────────────────────────── AC-G6.1 ────────────────────────────


def test_serve_command_exists() -> None:
    """`chemscheme serve --help` works (command is registered)."""
    runner = CliRunner()
    result = runner.invoke(cli_main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output.lower()
    assert "port" in result.output.lower()


def test_serve_starts_and_serves_index(serve_process) -> None:
    """`chemscheme serve --port N` starts a server; GET / returns Ketcher-loader HTML."""
    port = _free_port()
    serve_process(port)
    assert _wait_for_server(port), f"server did not respond on port {port} within timeout"

    with httpx.Client(timeout=2) as client:
        r = client.get(f"http://127.0.0.1:{port}/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        # Must mention Ketcher (either via inline reference or script src)
        assert "ketcher" in r.text.lower(), "index page must reference Ketcher editor"


def test_serve_health_endpoint(serve_process) -> None:
    """GET /api/health returns 200 + JSON status (used by smoke / monitoring)."""
    port = _free_port()
    serve_process(port)
    assert _wait_for_server(port)

    with httpx.Client(timeout=2) as client:
        r = client.get(f"http://127.0.0.1:{port}/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "ok"


# ──────────────────────────── AC-G6.2 / BDD-7 ────────────────────────────


def test_serve_port_conflict_clean_exit(serve_process) -> None:
    """Second `chemscheme serve` on same port → exits non-zero with helpful message."""
    port = _free_port()
    first = serve_process(port)
    assert _wait_for_server(port), "first server must be up before second is attempted"

    # Second invocation on same port should fail fast.
    second = subprocess.run(
        [sys.executable, "-m", "chemscheme.cli", "serve", "--port", str(port)],
        capture_output=True,
        timeout=10,
    )
    assert second.returncode != 0, "second serve on conflicting port must exit non-zero"
    err = (second.stderr + second.stdout).decode("utf-8", errors="replace").lower()
    assert "port" in err and ("in use" in err or "already" in err or "bind" in err), (
        f"error message must mention port conflict; got: {err!r}"
    )

    assert first.poll() is None, "first server must still be running after second's failure"


def test_serve_default_port_documented() -> None:
    """`chemscheme serve --help` documents a default port (so user knows what to expect)."""
    runner = CliRunner()
    result = runner.invoke(cli_main, ["serve", "--help"])
    assert result.exit_code == 0
    output = result.output.lower()
    assert "default" in output or "8" in output, "default port should be documented in --help"
