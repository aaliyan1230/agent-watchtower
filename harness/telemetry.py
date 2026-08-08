"""Telemetry: OTel SDK setup with a custom SpanExporter.

The OTel Python SDK exports spans through an extension point called a
SpanExporter — normally that is the OTLP exporter talking to a
collector. We implement a WatchtowerExporter instead: it serializes
ReadableSpans into the JSON envelope the Go service ingests and POSTs
them to its /v1/traces endpoint. The span *shape* on the wire mirrors
OTLP fields, so swapping this for the real OTLPSpanExporter in Phase
0.6 touches only this file.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
from opentelemetry.sdk.trace import ReadableSpan, Tracer, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode, set_tracer_provider

from . import semconv

_KIND_NAMES = {1: "internal", 2: "server", 3: "client", 4: "producer", 5: "consumer"}
_STATUS_NAMES = {StatusCode.UNSET: "unset", StatusCode.OK: "ok", StatusCode.ERROR: "error"}


def _attr_str(value: Any) -> str:
    """The Go model uses flat string attributes; coerce every value
    type the SDK can produce."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _to_rfc3339(ns: int) -> str:
    return (
        dt.datetime.fromtimestamp(ns / 1e9, tz=dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _span_to_json(span: ReadableSpan) -> dict[str, Any]:
    out: dict[str, Any] = {
        "traceId": f"{span.context.trace_id:032x}",
        "spanId": f"{span.context.span_id:016x}",
        "name": span.name,
        "kind": _KIND_NAMES.get(span.kind.value, "internal"),
        "startTime": _to_rfc3339(span.start_time),
        "endTime": _to_rfc3339(span.end_time),
        "status": _STATUS_NAMES.get(span.status.status_code, "unset"),
        "attributes": {k: _attr_str(v) for k, v in sorted(span.attributes.items())},
    }
    if span.parent and span.parent.span_id:
        out["parentSpanId"] = f"{span.parent.span_id:016x}"
    if span.status.description:
        out["statusMessage"] = span.status.description
    if span.events:
        out["events"] = [
            {
                "name": e.name,
                "time": _to_rfc3339(e.timestamp),
                "attributes": {k: _attr_str(v) for k, v in sorted(e.attributes.items())},
            }
            for e in span.events
        ]
    return out


class WatchtowerExporter(SpanExporter):
    """Serializes finished spans to the watchtower JSON envelope and
    POSTs them to the ingest endpoint. One POST per export flush keeps
    the batch simple; the Go side treats each envelope as a trace."""

    def __init__(self, endpoint: str, service_name: str = "harness"):
        self._endpoint = endpoint.rstrip("/") + "/v1/traces"
        self._service_name = service_name
        self._client = httpx.Client(timeout=5.0)

    def export(self, spans) -> SpanExportResult:
        envelope = {
            "resource": {"service.name": self._service_name},
            "spans": [_span_to_json(s) for s in spans],
        }
        try:
            resp = self._client.post(self._endpoint, json=envelope)
            if resp.status_code >= 400:
                # Surface the Go side's rejection loudly: an invalid
                # envelope means the trace was not verified.
                raise httpx.HTTPStatusError(
                    f"watchtower ingest rejected envelope: {resp.text}", request=resp.request, response=resp
                )
            self._last_report = resp.json()  # verdict report, available to callers
        except httpx.HTTPError as exc:
            print(f"[telemetry] ingest failed: {exc}")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._client.close()


class HarnessTelemetry:
    """Owns the tracer used by workers and supervisor. BatchSpanProcessor
    accumulates spans until flush() — one envelope per run, which is
    what the Go graph reconstructor expects (one trace per envelope).
    Exporting per-span would break parent/child reconstruction."""

    def __init__(self, endpoint: str, service_name: str = "harness"):
        self._exporter = WatchtowerExporter(endpoint, service_name)
        provider = TracerProvider()
        self._processor = BatchSpanProcessor(self._exporter)
        provider.add_span_processor(self._processor)
        set_tracer_provider(provider)
        self._tracer: Tracer = provider.get_tracer("watchtower.harness")

    @property
    def exporter(self) -> WatchtowerExporter:
        return self._exporter

    def tracer(self) -> Tracer:
        return self._tracer

    def flush(self) -> None:
        """Synchronously export all pending spans as one envelope;
        returns with every span of the run on the wire."""
        self._processor.force_flush()

    def shutdown(self) -> None:
        self._processor.shutdown()
        self._exporter.shutdown()
