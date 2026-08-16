"""Execute the experiment grid against a watchtower server.

Offline by default: FakeProvider drives every cell deterministically —
free, reproducible, no keys. --live switches to Gemini and the LLM
judge, for small stratified samples only (the pilot budget rule).
The fault lands on a fixed response step offline (deterministic); live
cells corrupt every turn instead, because a real model's turn count is
not predictable.

Usage: python -m experiments.run [--grid artifacts/grid.json]
                                 [--out artifacts/results.json]
                                 [--live] [--limit N]
"""

from __future__ import annotations

import argparse
import subprocess
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.env import get_api_key
from harness.faults import FaultInjector, FaultKind, FaultSpec
from harness.providers import FakeProvider, GeminiProvider, Provider, ProviderResponse
from harness.server import REPO_ROOT, spawn_server
from harness.supervisor import Supervisor
from harness.telemetry import HarnessTelemetry
from harness.workers import Tool, Worker

from .suite import PROTOCOL_VERSION, ExperimentCell, build_evidence_grid, checksum

ENDPOINT = "http://127.0.0.1:4318"
ADDR = "127.0.0.1:4318"
LIVE_MODEL = "gemini-3.5-flash-lite"  # cheapest tier for the pilot


def aws_env() -> dict[str, str]:
    """AWS creds for the spawned Go server (the Bedrock judge signs
    with them). .env wins when both key+secret are set there; otherwise
    the AWS CLI session is forwarded (aws configure export-credentials
    handles profiles and session tokens)."""
    from_env = {
        "AWS_ACCESS_KEY_ID": get_api_key("AWS_ACCESS_KEY_ID") or "",
        "AWS_SECRET_ACCESS_KEY": get_api_key("AWS_SECRET_ACCESS_KEY") or "",
        "AWS_SESSION_TOKEN": get_api_key("AWS_SESSION_TOKEN") or "",
        "AWS_REGION": get_api_key("AWS_REGION") or "",
    }
    if from_env["AWS_ACCESS_KEY_ID"] and from_env["AWS_SECRET_ACCESS_KEY"]:
        return {k: v for k, v in from_env.items() if v}
    out = subprocess.run(
        ["aws", "configure", "export-credentials", "--format", "env"],
        capture_output=True, text=True, check=True,
    ).stdout
    env: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    if region := from_env["AWS_REGION"]:
        env["AWS_REGION"] = region
    return env

# Which response step each fault corrupts in the canonical script
# below ([0] tool turn, [1] final JSON answer). Live mode uses
# TOOL_FAULT_STEP for faults that inject tool calls: one corrupted turn
# is enough for the policy/loop/budget verifiers, and repeated
# corruption of every turn breaks the provider's conversation history.
# Content faults corrupt every turn instead, because a real model's
# answer turn number is not predictable.
FAULT_STEP = {
    "malformed_json": 1,
    "schema_violation": 1,
    "policy_violation": 0,
    "loop": 0,
    "budget_blowout": 0,
    "provider_timeout": 0,
}
LIVE_CONTENT_FAULTS = {"malformed_json", "schema_violation"}

SYSTEM_PROMPT = (
    "You are a triage agent. Call the search tool at most once, then answer "
    'immediately with the ticket JSON: {"id": int, "title": string, "labels": [string]}.'
)
TOOLS = [Tool("search", "search the knowledge base", {"q": {"type": "string"}}, lambda q: f"results for {q}: ticket #42")]
CONTRACT = "ticket"

# Clean run stays well under the token limit (75), a ×10 blowout on
# the first turn clears it (435) — the experiment config's
# maxTotalTokens=300 sits between the two.
NORMAL_SCRIPT = [
    ProviderResponse(tool_calls=[{"name": "search", "args": {"q": "login"}}], input_tokens=30, output_tokens=10),
    ProviderResponse(content='{"id": 1, "title": "broken login", "labels": ["auth"]}', input_tokens=20, output_tokens=15),
]


@dataclass
class CellResult:
    fault: str | None
    seed: int
    model: str
    run: int
    verdict: str
    evidence_fault: str | None = None
    judged: bool = False
    findings: list[dict] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    answer: str = ""
    trace_id: str = ""
    trace_path: str = ""
    trace_sha256: str = ""
    report_sha256: str = ""
    protocol: str = ""  # report protocolVersion, for artifact freezing
    judge_usage: dict = field(default_factory=dict)  # measured judge tokens
    judge_model: str = ""  # judging model, for cost attribution on clean runs


def build_workers(telemetry: HarnessTelemetry, cell, live: bool) -> list[Worker]:
    spec = None
    if cell.fault:
        if live and cell.fault in LIVE_CONTENT_FAULTS:
            spec = FaultSpec(FaultKind(cell.fault), seed=cell.seed, step=None)
        else:
            spec = FaultSpec(FaultKind(cell.fault), seed=cell.seed, step=FAULT_STEP[cell.fault])
    provider: Provider
    if live:
        provider = GeminiProvider(model=LIVE_MODEL, api_key=get_api_key("GEMINI_API_KEY"))
    else:
        provider = FakeProvider(NORMAL_SCRIPT)
    if spec is not None:
        provider = FaultInjector(provider, spec)
    return [
        Worker("worker-a", SYSTEM_PROMPT, provider, TOOLS, telemetry.tracer(), contract=CONTRACT, max_steps=3)
    ]


def run_cell(telemetry: HarnessTelemetry, cell, live: bool) -> CellResult:
    supervisor = Supervisor("supervisor", telemetry.tracer(), max_steps=8, max_tokens=900, max_tool_failures=2)
    result = supervisor.run("Triage ticket #42: users cannot log in", build_workers(telemetry, cell, live))
    telemetry.flush()
    report = telemetry.fetch_report(result.trace_id)
    base = dict(
        fault=cell.fault,
        seed=cell.seed,
        model=cell.model,
        run=cell.run,
        evidence_fault=getattr(cell, "evidence_fault", None),
    )
    native_outcome = "known_behavior_fault" if cell.fault else "clean_reference"
    trace_record = telemetry.finalize_trace(report, native_outcome=native_outcome)
    trace_fields = {
        "trace_path": (trace_record or {}).get("tracePath", ""),
        "trace_sha256": (trace_record or {}).get("traceSha256", ""),
        "report_sha256": (trace_record or {}).get("reportSha256", ""),
    }
    if report is None:
        return CellResult(verdict="NO_REPORT", **trace_fields, **base)
    return CellResult(
        verdict=report["verdict"],
        judged=report["judged"],
        findings=report.get("findings", []),
        budget=report.get("budget", {}),
        answer=result.answers["worker-a"],
        trace_id=result.trace_id,
        protocol=report.get("protocolVersion", ""),
        judge_usage=report.get("judgeUsage", {}),
        judge_model=report.get("judgeModel", ""),
        **trace_fields,
        **base,
    )


def file_checksum(path: str) -> str:
    """sha256 of the verifier config — pins schema contracts, policy,
    limits, and judge model into the artifact manifest."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cells_signature(cells: list[ExperimentCell]) -> str:
    """sha256 over the grid identity of executed cells — the artifact's
    self-integrity check: any truncation or tampering changes it."""
    import hashlib
    import json as _json

    identity = [
        {
            "fault": c.fault,
            "evidenceFault": getattr(c, "evidence_fault", None),
            "seed": c.seed,
            "model": c.model,
            "run": c.run,
        }
        for c in cells
    ]
    return hashlib.sha256(_json.dumps(identity, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="run the experiment grid")
    parser.add_argument("--grid", type=Path, default=Path("artifacts/grid.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/results.json"))
    parser.add_argument("--live", action="store_true", help="real Gemini + LLM judge (needs GEMINI_API_KEY)")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cells")
    parser.add_argument("--pilot", action="store_true", help="curated small spread: clean + every fault x 2 seeds")
    parser.add_argument("--evidence", action="store_true", help="run the clean-behavior telemetry-fault grid")
    parser.add_argument("--trace-dir", type=Path, default=Path("artifacts/traces"), help="directory for clean trace artifacts")
    parser.add_argument("--no-traces", action="store_true", help="disable durable trace recording")
    parser.add_argument("--judge", choices=["gemini", "bedrock", "kimi"], default="gemini", help="judge backend for live mode")
    args = parser.parse_args()

    if args.live and args.judge == "gemini" and not get_api_key("GEMINI_API_KEY"):
        raise SystemExit("--live --judge gemini needs GEMINI_API_KEY in .env")

    if args.evidence:
        cells = build_evidence_grid()
        grid_checksum = checksum(cells)
    elif args.pilot:
        cells = [
            ExperimentCell(fault=fault, seed=seed, model="flash", run=1)
            for fault in [None, *(f.value for f in FaultKind)]
            for seed in [1, 2]
        ]
        grid_checksum = "pilot"
    else:
        grid = json.loads(args.grid.read_text())
        cells = [ExperimentCell(**c) for c in grid["cells"]]
        grid_checksum = grid["checksum"]
    if args.limit:
        cells = cells[: args.limit]

    config_name = {
        ("offline", "gemini"): "experiment_config.json",
        ("live", "gemini"): "experiment_live_config.json",
        ("live", "bedrock"): "experiment_live_bedrock_config.json",
        ("live", "kimi"): "experiment_live_kimi_config.json",
    }[("live" if args.live else "offline", args.judge)]
    config = str(REPO_ROOT / "watchtower" / "testdata" / config_name)
    config_checksum = file_checksum(config)
    env: dict[str, str] | None = None
    if args.live:
        env = {"GEMINI_API_KEY": get_api_key("GEMINI_API_KEY") or ""}
        if args.judge in ("bedrock", "kimi"):
            env = aws_env()  # both ride the Bedrock transport
    proc = spawn_server(ENDPOINT, ADDR, config, env)
    try:
        results: list[CellResult] = []
        for i, cell in enumerate(cells, 1):
            trace_metadata = {
                "benchmark": "watchtower-controlled-triage",
                "taskId": "triage-ticket-42",
                "producer": "watchtower-harness",
                "framework": "python-otel-harness",
                "model": LIVE_MODEL if args.live else cell.model,
                "seed": cell.seed,
                "run": cell.run,
                "suiteProtocolVersion": PROTOCOL_VERSION,
                "configSha256": config_checksum,
                "behaviorFault": cell.fault,
                "evidenceFault": getattr(cell, "evidence_fault", None),
                "license": "project-generated controlled workload",
                "privacy": "no production data; synthetic ticket and tool result",
            }
            telemetry = HarnessTelemetry(
                ENDPOINT,
                service_name="experiment",
                evidence_fault=getattr(cell, "evidence_fault", None),
                trace_dir=None if args.no_traces else str(args.trace_dir),
                trace_metadata=trace_metadata,
            )
            res = run_cell(telemetry, cell, args.live)
            telemetry.shutdown()
            results.append(res)
            print(
                f"[{i}/{len(cells)}] fault={cell.fault} "
                f"evidence={getattr(cell, 'evidence_fault', None)} "
                f"seed={cell.seed} -> {res.verdict}"
            )
        payload = {
            "protocolVersion": PROTOCOL_VERSION,
            "gridChecksum": grid_checksum,
            "configChecksum": config_checksum,
            "cellsSignature": cells_signature(cells),
            "reportProtocolVersion": results[0].protocol,
            "mode": "live" if args.live else "offline",
            "gridKind": "evidence" if args.evidence else "behavior",
            "judgeBackend": args.judge,
            "traceDir": None if args.no_traces else str(args.trace_dir),
            "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cells": [asdict(r) for r in results],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {len(results)} results -> {args.out}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
