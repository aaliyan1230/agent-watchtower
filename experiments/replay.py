"""Replay a recorded trace through the Go verifier.

The clean artifact is the source of truth.  ``--fault`` derives a mutated
copy locally and sends that JSON envelope to the existing ingest endpoint;
the agent is never rerun.  This makes telemetry experiments cheap,
auditable, and easy to reproduce from a released trace package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import httpx

from harness.server import REPO_ROOT, spawn_server
from harness.trace_artifacts import (
    apply_evidence_fault,
    report_fingerprint,
    sha256_bytes,
    upsert_manifest,
    write_json_atomic,
)

ENDPOINT = "http://127.0.0.1:4318"
ADDR = "127.0.0.1:4318"
DEFAULT_CONFIG = REPO_ROOT / "watchtower" / "testdata" / "experiment_config.json"

EVIDENCE_FAULTS = (
    "drop_parent",
    "duplicate_span",
    "reorder_spans",
    "drop_child",
    "drop_tool_result",
    "mismatch_tool_id",
    "truncate_final",
    "late_span",
)


def load_trace(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("spans"), list) or not value["spans"]:
        raise ValueError(f"{path}: expected a JSON envelope with a spans list")
    trace_ids = {str(span.get("traceId", "")) for span in value["spans"]}
    if len(trace_ids) != 1 or "" in trace_ids:
        raise ValueError(f"{path}: expected one non-empty trace id")
    return value


def post_envelope(endpoint: str, envelope: Mapping[str, Any]) -> dict[str, Any]:
    response = httpx.post(
        endpoint.rstrip("/") + "/v1/traces",
        content=json.dumps(envelope, sort_keys=True),
        headers={"content-type": "application/json"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _derived_path(source: Path, fault: str) -> Path:
    return source.with_name(f"{source.stem}.{fault}{source.suffix}")


def _manifest_record(
    source: Path,
    derived: Path,
    envelope: Mapping[str, Any],
    trace_id: str,
    fault: str | None,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_id = derived.stem
    return {
        "schemaVersion": "watchtower.manifest.v1",
        "artifactId": artifact_id,
        "traceId": trace_id,
        "tracePath": derived.name,
        "traceSha256": sha256_bytes(envelope),
        "spanCount": len(envelope.get("spans", [])),
        "wireFormat": "canonical-json-envelope",
        "capture": "derived-from-clean",
        "sourceTrace": source.name,
        "evidenceFault": fault,
        "watchtowerVerdict": report.get("verdict"),
        "reportSha256": report_fingerprint(report),
        "reportProtocolVersion": report.get("protocolVersion"),
    }


def replay(
    trace_path: Path,
    *,
    endpoint: str = ENDPOINT,
    fault: str | None = None,
    out_trace: Path | None = None,
    out_report: Path | None = None,
    manifest: Path | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    clean = load_trace(trace_path)
    trace_id = str(clean["spans"][0]["traceId"])
    envelope = apply_evidence_fault(clean, fault)
    derived = trace_path if fault is None else (out_trace or _derived_path(trace_path, fault))
    if fault is not None:
        write_json_atomic(derived, envelope)
    report = post_envelope(endpoint, envelope)
    if out_report:
        write_json_atomic(out_report, report)
    if fault is not None:
        manifest_path = manifest or trace_path.parent / "manifest.jsonl"
        upsert_manifest(manifest_path, _manifest_record(trace_path, derived, envelope, trace_id, fault, report))
    return report, derived, envelope


def main() -> None:
    parser = argparse.ArgumentParser(description="replay a clean trace through Watchtower")
    parser.add_argument("--trace", type=Path, required=True, help="clean canonical trace JSON")
    parser.add_argument("--fault", choices=EVIDENCE_FAULTS, help="derive and apply a telemetry fault before replay")
    parser.add_argument("--endpoint", default=ENDPOINT, help="running Watchtower base URL")
    parser.add_argument("--addr", default=ADDR, help="server listen address when starting one")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="verifier config for the temporary server")
    parser.add_argument("--no-start", action="store_true", help="use an already running server")
    parser.add_argument("--out-trace", type=Path, help="path for the derived faulted trace")
    parser.add_argument("--out-report", type=Path, help="write the replay report to this file")
    parser.add_argument("--manifest", type=Path, help="manifest to update for a derived trace")
    args = parser.parse_args()

    proc = None
    if not args.no_start:
        proc = spawn_server(args.endpoint, args.addr, str(args.config))
    try:
        report, derived, envelope = replay(
            args.trace,
            endpoint=args.endpoint,
            fault=args.fault,
            out_trace=args.out_trace,
            out_report=args.out_report,
            manifest=args.manifest,
        )
        result = {
            "sourceTrace": args.trace.name,
            "trace": derived.name,
            "traceSha256": sha256_bytes(envelope),
            "fault": args.fault,
            "verdict": report.get("verdict"),
            "reportSha256": report_fingerprint(report),
            "protocolVersion": report.get("protocolVersion"),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()
