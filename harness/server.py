"""Shared watchtower serve lifecycle for scripts that need a server:
build the binary once, kill stale instances, wait for readiness."""

from __future__ import annotations

import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = "/tmp/watchtower-smoke"


class StaleServerError(RuntimeError):
    """Something other than the server we spawned answers on the port.

    Reusing it would verify every cell under whatever config and code it
    was started with; the Aug 31 artifacts were silently produced by a
    leftover `make demo` server this way."""


def server_up(endpoint: str) -> bool:
    try:
        urllib.request.urlopen(endpoint + "/v1/traces", timeout=1)
        return True
    except urllib.error.HTTPError:
        return True  # 405 means the server answered
    except (urllib.error.URLError, OSError):
        return False


def spawn_server(endpoint: str, addr: str, config: str, env: dict[str, str] | None = None) -> subprocess.Popen:
    subprocess.run(["pkill", "-f", BINARY], check=False)
    for _ in range(20):
        if not server_up(endpoint):
            break
        time.sleep(0.1)
    else:
        raise StaleServerError(
            f"{endpoint} is already served by a process this script did not start "
            f"(e.g. `make serve`/`make demo` via go run); stop it and retry: "
            f"lsof -nP -iTCP:{addr.rsplit(':', 1)[-1]} -sTCP:LISTEN"
        )
    subprocess.run(
        ["go", "build", "-o", BINARY, "./cmd/watchtower"],
        cwd=REPO_ROOT / "watchtower",
        check=True,
    )
    # A file, not a pipe: serve logs a line per trace and would block on a
    # full pipe buffer over a long matrix.
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        [BINARY, "serve", "--addr", addr, "--config", config],
        env=env,  # None = inherit; the Go side reads keys from env only
        stdout=subprocess.DEVNULL,
        stderr=log,
    )
    for _ in range(300):
        if proc.poll() is not None:
            log.seek(0)
            err = log.read().decode(errors="replace").strip()
            raise RuntimeError(f"watchtower serve exited ({proc.returncode}): {err}")
        if server_up(endpoint):
            return proc
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("watchtower serve did not come up")
