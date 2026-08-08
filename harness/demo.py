"""Offline end-to-end demo: fake providers -> OTel spans -> watchtower
serve -> verdict reports. No API keys, no network beyond localhost.

Starts `watchtower serve` itself if it is not already running, runs two
scenarios (clean and faulted), and prints the verdicts:

    clean run   -> PASS
    faulted run -> FAIL (schema verifier catches malformed JSON)

Run from the repo root: `make demo` (or `.venv/bin/python -m harness.demo`).
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from harness.faults import FaultInjector, FaultKind, FaultSpec
from harness.providers import FakeProvider, ProviderResponse
from harness.supervisor import Supervisor
from harness.telemetry import HarnessTelemetry
from harness.workers import Tool, Worker

ENDPOINT = "http://127.0.0.1:4318"
SERVE_ADDR = "127.0.0.1:4318"
# Resolve paths from this file's location so the demo works no matter
# what cwd it is launched from (the spawned server runs with its own
# cwd and needs absolute paths).
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = str(REPO_ROOT / "watchtower" / "testdata" / "demo_config.json")
GO_ROOT = str(REPO_ROOT / "watchtower")


def search(query: str) -> str:
    return f"results for {query}: ticket #42 is a login issue"


def read_file(path: str) -> str:
    return "file content: onboarding docs"


def server_up() -> bool:
    try:
        urllib.request.urlopen(ENDPOINT + "/v1/traces", timeout=1)
        return True
    except urllib.error.HTTPError:
        return True  # 405 means the server answered
    except (urllib.error.URLError, OSError):
        return False


def ensure_server() -> subprocess.Popen | None:
    if server_up():
        return None
    proc = subprocess.Popen(
        ["go", "run", "./cmd/watchtower", "serve", "--addr", SERVE_ADDR, "--config", CONFIG],
        cwd=GO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(300):  # serve compiles on first run; be patient
        if server_up():
            return proc
        time.sleep(0.1)
    raise RuntimeError("watchtower serve did not come up")


def build_workers(telemetry: HarnessTelemetry, fault_spec: FaultSpec | None):
    """Two workers: a tool-calling one and a structured-output one.
    The fault, if any, lands on the structured-output worker so the
    schema verifier is the one that catches it."""
    tools = [
        Tool("search", "search the knowledge base", {"query": {"type": "string"}}, search),
        Tool("read", "read a file", {"path": {"type": "string"}}, read_file),
    ]
    worker_a = Worker(
        "worker-a",
        "You are a triage agent. Call tools when you need data.",
        FakeProvider(
            [
                ProviderResponse(tool_calls=[{"name": "search", "args": {"query": "login"}}], input_tokens=20, output_tokens=10),
                ProviderResponse(content="Ticket #42: broken login flow, escalate to auth team.", input_tokens=10, output_tokens=15),
            ]
        ),
        tools,
        telemetry.tracer(),
        max_steps=3,
    )
    worker_b = Worker(
        "worker-b",
        "You emit tickets as JSON matching the contract.",
        FaultInjector(
            FakeProvider(
                [
                    ProviderResponse(
                        content='{"id": 3, "title": "broken login", "labels": ["auth"]}',
                        input_tokens=15,
                        output_tokens=20,
                    )
                ]
            ),
            fault_spec,
        ),
        [],
        telemetry.tracer(),
        contract="ticket",
        max_steps=2,
    )
    return [worker_a, worker_b]


def scenario(telemetry: HarnessTelemetry, fault: FaultSpec | None, label: str):
    workers = build_workers(telemetry, fault)
    supervisor = Supervisor("supervisor", telemetry.tracer())
    result = supervisor.run("Triage ticket #42: users cannot log in", workers)
    if not telemetry.flush():
        print(f"  {label:<14} ingest rejected the trace")
        return "?"
    report = telemetry.fetch_report(result.trace_id)
    verdict = report.get("verdict", "?") if report else "NO REPORT"
    print(f"  {label:<14} verdict: {verdict}")
    if report:
        for f in report.get("findings", []):
            print(f"    - {f['verifier']}: {f['message']}")
    return verdict


def main() -> None:
    proc = ensure_server()
    try:
        print(f"watchtower ingest at {ENDPOINT}")
        telemetry = HarnessTelemetry(ENDPOINT, service_name="harness-demo")
        clean = scenario(telemetry, None, "clean")
        faulted = scenario(telemetry, FaultSpec(FaultKind.MALFORMED_JSON, seed=7), "faulted")
        telemetry.shutdown()
        ok = clean == "PASS" and faulted == "FAIL"
        print("demo:", "OK" if ok else "UNEXPECTED VERDICTS")
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()
