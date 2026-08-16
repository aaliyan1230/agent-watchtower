"""Phase 2.9 acceptance: the evidence contract is producer-neutral.

Two independent producers — the Python harness (opentelemetry-python
SDK) and the Go goproducer (go.opentelemetry.io SDK) — share no
implementation code. Against one server and one claim contract they
must produce the same verdict semantics:

- a complete run gets PASS from both;
- a run whose producer omits a required evidence field gets
  INCONCLUSIVE — never a silently weakened PASS;
- every evidence finding names the claim, the missing evidence, exact
  span references, and a practical recovery action.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import pytest

from experiments.run import CONTRACT, NORMAL_SCRIPT, SYSTEM_PROMPT, TOOLS
from harness.providers import FakeProvider
from harness.server import REPO_ROOT, spawn_server
from harness.supervisor import Supervisor
from harness.telemetry import HarnessTelemetry
from harness.workers import Worker

ENDPOINT = "http://127.0.0.1:4318"
ADDR = "127.0.0.1:4318"
CONFIG = str(REPO_ROOT / "watchtower" / "testdata" / "experiment_config.json")
GOPRODUCER = "/tmp/watchtower-goproducer"
GOPRODUCER_DIR = REPO_ROOT / "producers" / "goproducer"


@pytest.fixture(scope="session")
def server():
    proc = spawn_server(ENDPOINT, ADDR, CONFIG)
    yield proc
    proc.terminate()


@pytest.fixture(scope="session")
def goproducer():
    subprocess.run(["go", "build", "-o", GOPRODUCER, "."], cwd=GOPRODUCER_DIR, check=True)
    return GOPRODUCER


def fetch_report(trace_id: str) -> dict:
    with urllib.request.urlopen(f"{ENDPOINT}/v1/reports/{trace_id}", timeout=5) as resp:
        return json.loads(resp.read())


def run_goproducer(variant: str) -> tuple[str, dict]:
    out = subprocess.run(
        [GOPRODUCER, "-endpoint", ENDPOINT, "-variant", variant],
        capture_output=True, text=True, check=True,
    ).stdout
    trace_id = ""
    for field in out.split():
        if field.startswith("traceId="):
            trace_id = field[len("traceId="):]
    assert trace_id, f"goproducer output has no traceId: {out}"
    return trace_id, fetch_report(trace_id)


def run_harness_clean() -> dict:
    telemetry = HarnessTelemetry(ENDPOINT, service_name="cross-producer-test")
    provider = FakeProvider(NORMAL_SCRIPT)
    workers = [
        Worker("worker-a", SYSTEM_PROMPT, provider, TOOLS, telemetry.tracer(), contract=CONTRACT, max_steps=3)
    ]
    supervisor = Supervisor("supervisor", telemetry.tracer(), max_steps=8, max_tokens=900, max_tool_failures=2)
    result = supervisor.run("Triage ticket #42: users cannot log in", workers)
    telemetry.flush()
    report = telemetry.fetch_report(result.trace_id)
    telemetry.shutdown()
    assert report is not None, "harness clean run produced no report"
    return report


def test_harness_clean_passes(server):
    report = run_harness_clean()
    assert report["verdict"] == "PASS"


def test_goproducer_clean_passes(server, goproducer):
    _, report = run_goproducer("clean")
    assert report["verdict"] == "PASS"


@pytest.mark.parametrize("variant,claim", [
    ("omit-genai-system", "model_correlation"),
    ("omit-tool-result", "tool_pairing"),
    ("omit-tool-id", "tool_pairing"),
    ("omit-final", "final_answer"),
])
def test_omitted_field_abstains_instead_of_passing(server, goproducer, variant, claim):
    _, report = run_goproducer(variant)
    assert report["verdict"] == "INCONCLUSIVE"
    claims = {finding.get("claim") for finding in report.get("findings", [])}
    assert claim in claims, f"expected claim {claim!r} in {claims}"


def test_evidence_finding_names_claim_evidence_and_action(server, goproducer):
    _, report = run_goproducer("omit-tool-result")
    finding = next(f for f in report["findings"] if f.get("claim") == "tool_pairing")
    assert finding["kind"] == "evidence_gap"
    assert finding["spanIds"], "finding must reference the offending span"
    assert finding["timestamps"], "finding must carry timestamps"
    assert finding["action"], "finding must name a recovery action"
