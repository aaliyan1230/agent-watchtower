"""Canonical trace artifacts and deterministic telemetry mutations.

The live exporter receives SDK ``ReadableSpan`` objects, not the raw OTLP
bytes.  This module records the same information Watchtower verifies as a
small JSON envelope before a transport fault is applied.  The envelope is
deliberately the legacy JSON wire shape, so it can be replayed through the
existing ``application/json`` ingest path without adding a second verifier
or a research-only wire contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

TRACE_SCHEMA_VERSION = "watchtower.trace.v1"
MANIFEST_SCHEMA_VERSION = "watchtower.manifest.v1"


def _value(value: Any) -> str:
    """Flatten an OTel attribute into the string form used by model.Span."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _attributes(values: Mapping[str, Any] | None) -> dict[str, str] | None:
    if not values:
        return None
    return {str(key): _value(value) for key, value in sorted(values.items()) if value is not None}


def _id(value: Any, width: int) -> str:
    """Return an OTLP id as lower-case, zero-padded hexadecimal."""
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:0{width}x}"
    text = str(value)
    if text.startswith("0x"):
        text = text[2:]
    return text.lower().zfill(width)


def _context_id(context: Any, name: str, width: int) -> str:
    if context is None:
        return ""
    value = getattr(context, name, None)
    return _id(value, width)


def _time_ns(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, datetime):
        return int(value.timestamp() * 1_000_000_000)
    return int(value)


def format_time(value: Any) -> str:
    """Format nanoseconds as RFC3339 with all nine fractional digits."""
    ns = _time_ns(value)
    seconds, remainder = divmod(ns, 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{remainder:09d}Z"


def parse_time(value: str) -> int:
    """Parse the RFC3339 form emitted by :func:`format_time`."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    return int(parsed.timestamp() * 1_000_000_000)


def _event(event: Any) -> dict[str, Any]:
    attrs = getattr(event, "attributes", None)
    if attrs is None:
        attrs = getattr(event, "_attributes", None)
    timestamp = getattr(event, "timestamp", None)
    if timestamp is None:
        timestamp = getattr(event, "_timestamp", 0)
    out: dict[str, Any] = {
        "name": str(getattr(event, "name", getattr(event, "_name", ""))),
        "time": format_time(timestamp),
    }
    encoded = _attributes(attrs)
    if encoded:
        out["attributes"] = encoded
    return out


def span_to_dict(span: Any) -> dict[str, Any]:
    """Convert one SDK ReadableSpan into the Watchtower JSON span model."""
    context = getattr(span, "context", None)
    if context is None:
        context = getattr(span, "_context", None)
    parent = getattr(span, "parent", None)
    if parent is None:
        parent = getattr(span, "_parent", None)

    attrs = getattr(span, "attributes", None)
    if attrs is None:
        attrs = getattr(span, "_attributes", None)
    events = getattr(span, "events", None)
    if events is None:
        events = getattr(span, "_events", ())
    resource = getattr(span, "resource", None)
    if resource is None:
        resource = getattr(span, "_resource", None)
    resource_attrs = getattr(resource, "attributes", None) if resource is not None else None

    status = getattr(span, "status", None)
    if status is None:
        status = getattr(span, "_status", None)
    status_code = getattr(status, "status_code", None)
    status_name = getattr(status_code, "name", str(status_code or "UNSET"))
    status_map = {"OK": "ok", "ERROR": "error", "UNSET": "unset"}

    kind = getattr(span, "kind", None)
    if kind is None:
        kind = getattr(span, "_kind", None)
    kind_name = getattr(kind, "name", str(kind or "INTERNAL")).lower()
    if kind_name == "unset":
        kind_name = "internal"

    start = getattr(span, "start_time", None)
    if start is None:
        start = getattr(span, "_start_time", 0)
    end = getattr(span, "end_time", None)
    if end is None:
        end = getattr(span, "_end_time", 0)

    out: dict[str, Any] = {
        "traceId": _context_id(context, "trace_id", 32),
        "spanId": _context_id(context, "span_id", 16),
        "name": str(getattr(span, "name", getattr(span, "_name", ""))),
        "kind": kind_name,
        "startTime": format_time(start),
        "endTime": format_time(end),
        "status": status_map.get(status_name, "unset"),
    }
    parent_id = _context_id(parent, "span_id", 16)
    if parent_id:
        out["parentSpanId"] = parent_id
    status_message = getattr(status, "description", None)
    if status_message:
        out["statusMessage"] = str(status_message)
    encoded_attrs = _attributes(attrs)
    if encoded_attrs:
        out["attributes"] = encoded_attrs
    if events:
        out["events"] = [_event(item) for item in events]
    # Resource is repeated on every SDK span.  The envelope stores it once;
    # callers use envelope_from_spans rather than this helper directly when
    # writing an artifact.
    if resource_attrs:
        out["_resource"] = _attributes(resource_attrs) or {}
    return out


def envelope_from_spans(spans: Iterable[Any]) -> dict[str, Any]:
    """Build a canonical, replayable envelope from SDK spans."""
    converted = [span_to_dict(span) for span in spans]
    resources = [item.pop("_resource", {}) for item in converted]
    resource: dict[str, str] = {}
    for candidate in resources:
        for key, value in candidate.items():
            resource.setdefault(key, value)
    converted.sort(key=lambda item: (item.get("traceId", ""), item.get("startTime", ""), item.get("spanId", "")))
    envelope: dict[str, Any] = {
        "schemaVersion": TRACE_SCHEMA_VERSION,
        "resource": resource,
        "spans": converted,
    }
    if not resource:
        envelope.pop("resource")
    return envelope


def canonical_bytes(value: Any) -> bytes:
    """Stable bytes used for all artifact checksums."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def report_fingerprint(report: Mapping[str, Any]) -> str:
    """Hash report semantics while ignoring the generation wall-clock."""
    stable = dict(report)
    stable.pop("generatedAt", None)
    return sha256_bytes(stable)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _trace_id(envelope: Mapping[str, Any]) -> str:
    ids = {str(span.get("traceId", "")) for span in envelope.get("spans", []) if span.get("traceId")}
    if len(ids) != 1:
        raise ValueError(f"expected one trace id, found {sorted(ids)}")
    return ids.pop()


def _merge_envelopes(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    spans: dict[str, dict[str, Any]] = {}
    for item in [*previous.get("spans", []), *current.get("spans", [])]:
        spans.setdefault(str(item.get("spanId", "")), copy.deepcopy(item))
    merged = {
        "schemaVersion": current.get("schemaVersion", TRACE_SCHEMA_VERSION),
        "spans": list(spans.values()),
    }
    resource = dict(previous.get("resource", {}))
    resource.update(current.get("resource", {}))
    if resource:
        merged["resource"] = resource
    merged["spans"].sort(key=lambda item: (item.get("traceId", ""), item.get("startTime", ""), item.get("spanId", "")))
    return merged


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def upsert_manifest(path: Path, record: Mapping[str, Any]) -> None:
    """Replace one artifact entry without creating duplicate manifest rows."""
    key = str(record.get("artifactId") or record.get("tracePath") or record.get("traceId"))
    records = _read_manifest(path)
    replaced = False
    output: list[dict[str, Any]] = []
    for existing in records:
        existing_key = str(existing.get("artifactId") or existing.get("tracePath") or existing.get("traceId"))
        if existing_key == key:
            output.append(dict(record))
            replaced = True
        else:
            output.append(existing)
    if not replaced:
        output.append(dict(record))
    output.sort(key=lambda item: (str(item.get("tracePath", "")), str(item.get("artifactId", ""))))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class TraceRecorder(SpanExporter):
    """Persist clean spans before an inner exporter applies transport faults."""

    def __init__(
        self,
        inner: SpanExporter,
        directory: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ):
        self._inner = inner
        self._directory = Path(directory)
        self._manifest = self._directory / "manifest.jsonl"
        self._metadata = dict(metadata or {})
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._records)

    def export(self, spans: Iterable[Any]) -> SpanExportResult:
        batch = list(spans)
        try:
            groups: dict[str, list[Any]] = {}
            for span in batch:
                converted = span_to_dict(span)
                trace_id = converted.get("traceId", "")
                if not trace_id:
                    raise ValueError("span has no trace id")
                groups.setdefault(trace_id, []).append(span)
            with self._lock:
                for trace_id, group in groups.items():
                    path = self._directory / f"trace-{trace_id}.json"
                    envelope = envelope_from_spans(group)
                    if path.exists():
                        envelope = _merge_envelopes(json.loads(path.read_text(encoding="utf-8")), envelope)
                    write_json_atomic(path, envelope)
                    record = {
                        "schemaVersion": MANIFEST_SCHEMA_VERSION,
                        "artifactId": f"trace-{trace_id}",
                        "traceId": trace_id,
                        "tracePath": path.name,
                        "traceSha256": sha256_bytes(envelope),
                        "spanCount": len(envelope.get("spans", [])),
                        "wireFormat": "canonical-json-envelope",
                        "capture": "clean-before-transport-mutation",
                        "recordedAt": format_time(time.time_ns()),
                        **self._metadata,
                    }
                    self._records[trace_id] = record
                    upsert_manifest(self._manifest, record)
        except Exception:
            return SpanExportResult.FAILURE
        return self._inner.export(batch)

    def finalize(
        self,
        report: Mapping[str, Any] | None,
        *,
        native_outcome: str | None = None,
    ) -> dict[str, Any] | None:
        """Attach the Watchtower report and outcome to the manifest row."""
        with self._lock:
            if not self._records:
                return None
            trace_id = str(report.get("traceId")) if report and report.get("traceId") else next(iter(self._records))
            record = self._records.get(trace_id)
            if record is None:
                record = next(iter(self._records.values()))
                trace_id = str(record["traceId"])
            if report is not None:
                report_path = self._directory / f"report-{trace_id}.json"
                write_json_atomic(report_path, dict(report))
                record.update({
                    "watchtowerVerdict": report.get("verdict"),
                    "reportPath": report_path.name,
                    "reportSha256": report_fingerprint(report),
                    "reportProtocolVersion": report.get("protocolVersion"),
                    "reportInput": "derived-telemetry-fault" if self._metadata.get("evidenceFault") else "clean-trace",
                })
            if native_outcome is not None:
                record["nativeOutcome"] = native_outcome
            record["finalizedAt"] = format_time(time.time_ns())
            upsert_manifest(self._manifest, record)
            return copy.deepcopy(record)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def apply_evidence_fault(envelope: Mapping[str, Any], fault: str | None) -> dict[str, Any]:
    """Return a mutated copy of a clean envelope, never changing the source."""
    out = copy.deepcopy(dict(envelope))
    spans = list(out.get("spans", []))
    if not fault:
        return out
    if fault == "drop_parent":
        spans = [s for s in spans if not (s.get("name") == "agent.run" and not s.get("parentSpanId"))]
    elif fault == "duplicate_span":
        if spans:
            spans.append(copy.deepcopy(spans[-1]))
    elif fault == "reorder_spans":
        spans.reverse()
    elif fault == "drop_child":
        spans = [s for s in spans if not (s.get("name") == "agent.run" and s.get("parentSpanId"))]
    elif fault == "drop_tool_result":
        spans = [s for s in spans if s.get("name") != "tool.call"]
    elif fault == "mismatch_tool_id":
        for span in spans:
            if span.get("name") == "tool.call":
                attrs = dict(span.get("attributes") or {})
                attrs["tool.call.id"] = "forged-tool-result-id"
                span["attributes"] = attrs
                break
    elif fault == "truncate_final":
        for span in spans:
            attrs = dict(span.get("attributes") or {})
            for key in ("watchtower.final", "watchtower.output", "watchtower.contract"):
                attrs.pop(key, None)
            if attrs:
                span["attributes"] = attrs
            else:
                span.pop("attributes", None)
    elif fault == "late_span":
        ends = [parse_time(str(s["endTime"])) for s in spans if s.get("endTime")]
        latest_end = max(ends, default=0)
        for span in spans:
            if not span.get("parentSpanId") or not span.get("startTime") or not span.get("endTime"):
                continue
            start = parse_time(str(span["startTime"]))
            end = parse_time(str(span["endTime"]))
            duration = max(1_000_000, end - start)
            new_start = latest_end + 1_000_000
            span["startTime"] = format_time(new_start)
            span["endTime"] = format_time(new_start + duration)
            break
    else:
        raise ValueError(f"unknown evidence fault: {fault}")
    out["spans"] = spans
    return out
