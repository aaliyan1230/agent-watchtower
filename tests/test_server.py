import http.server
import socket
import threading

import pytest

from harness import server


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def no_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    return calls


def test_foreign_server_on_port_is_refused(no_subprocess):
    httpd = http.server.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with pytest.raises(server.StaleServerError):
            server.spawn_server(f"http://127.0.0.1:{port}", f"127.0.0.1:{port}", "cfg.json")
    finally:
        httpd.shutdown()
    assert not any("build" in cmd for cmd in no_subprocess), "must refuse before building"


class _ExitedProc:
    returncode = 1

    def poll(self):
        return self.returncode

    def kill(self):
        pass


def test_spawned_server_that_exits_is_reported(no_subprocess, monkeypatch):
    def fake_popen(cmd, stderr, **kwargs):
        stderr.write(b"listen tcp: address already in use")
        stderr.flush()
        return _ExitedProc()

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    port = _free_port()
    with pytest.raises(RuntimeError, match="address already in use"):
        server.spawn_server(f"http://127.0.0.1:{port}", f"127.0.0.1:{port}", "cfg.json")
