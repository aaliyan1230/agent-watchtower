"""Shared watchtower serve lifecycle for scripts that need a server:
build the binary once, kill stale instances, wait for readiness."""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = "/tmp/watchtower-smoke"


def server_up(endpoint: str) -> bool:
    try:
        urllib.request.urlopen(endpoint + "/v1/traces", timeout=1)
        return True
    except urllib.error.HTTPError:
        return True  # 405 means the server answered
    except (urllib.error.URLError, OSError):
        return False


def spawn_server(endpoint: str, addr: str, config: str, env: dict[str, str] | None = None) -> subprocess.Popen:
    # A stale instance from an earlier run would be reused by the
    # readiness check and silently serve old code.
    subprocess.run(["pkill", "-f", BINARY], check=False)
    subprocess.run(
        ["go", "build", "-o", BINARY, "./cmd/watchtower"],
        cwd=REPO_ROOT / "watchtower",
        check=True,
    )
    proc = subprocess.Popen(
        [BINARY, "serve", "--addr", addr, "--config", config],
        env=env,  # None = inherit; the Go side reads keys from env only
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(300):
        if server_up(endpoint):
            return proc
        time.sleep(0.1)
    raise RuntimeError("watchtower serve did not come up")
