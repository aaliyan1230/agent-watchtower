"""Telemetry: OTel SDK setup with the official OTLP/HTTP exporter.

Phase 0.6: the custom WatchtowerExporter is gone — the Go ingest now
speaks real OTLP/HTTP protobuf, so this module uses the standard
OTLPSpanExporter and the same BatchSpanProcessor + explicit flush
pattern: one export request per run, reconstructed as one trace.

The verdict report no longer rides the ingest response (OTLP clients
must receive an ExportTraceServiceResponse); it is fetched from the Go
side's report store via GET /v1/reports/{traceId}.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import Tracer, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import set_tracer_provider

# The SDK allows one global provider per process; experiments create a
# telemetry per cell, so only the first sets it.
_global_provider_set = False


class EvidenceFaultExporter(SpanExporter):
    """Apply a transport fault immediately before OTLP serialization.

    The inner exporter remains the official OTLP/HTTP exporter. This
    wrapper changes only the exported batch, which keeps evidence faults
    outside the agent and makes them reproducible in the experiment layer.
    """

    def __init__(self, inner: SpanExporter, fault: str):
        self._inner = inner
        self._fault = fault

    def export(self, spans) -> SpanExportResult:
        batch = list(spans)
        if self._fault == "drop_parent":
            batch = [
                span
                for span in batch
                if not (span.name == "agent.run" and span.parent is None)
            ]
        elif self._fault == "duplicate_span" and batch:
            batch.append(batch[-1])
        elif self._fault == "reorder_spans":
            batch.reverse()
        return self._inner.export(batch)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


class HarnessTelemetry:
    """Owns the tracer used by workers and supervisor. BatchSpanProcessor
    accumulates spans until flush() — one export per run, which is what
    the Go graph reconstructor expects (one trace per request)."""

    def __init__(self, endpoint: str, service_name: str = "harness", evidence_fault: str | None = None):
        global _global_provider_set
        self._endpoint = endpoint.rstrip("/")
        # An explicit `endpoint` is used verbatim as the export URL (the
        # /v1/traces suffix is only appended for the env-var default),
        # so the full path goes here.
        exporter: SpanExporter = OTLPSpanExporter(endpoint=self._endpoint + "/v1/traces")
        if evidence_fault:
            exporter = EvidenceFaultExporter(exporter, evidence_fault)
        self._exporter = exporter
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
        self._processor = BatchSpanProcessor(self._exporter)
        provider.add_span_processor(self._processor)
        if not _global_provider_set:
            set_tracer_provider(provider)
            _global_provider_set = True
        self._tracer: Tracer = provider.get_tracer("watchtower.harness")

    def tracer(self) -> Tracer:
        return self._tracer

    def flush(self) -> bool:
        """Synchronously export all pending spans; returns whether the
        ingest accepted them."""
        return self._processor.force_flush()

    def fetch_report(self, trace_id: str, retries: int = 5, delay: float = 0.1) -> dict[str, Any] | None:
        """Fetch the verdict for a trace from the report store. The
        ingest is synchronous on the Go side, so the report should
        already exist; retries cover slow exports without hiding the
        response entirely."""
        with httpx.Client(timeout=5.0) as client:
            for attempt in range(retries):
                resp = client.get(f"{self._endpoint}/v1/reports/{trace_id}")
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code != 404:
                    resp.raise_for_status()
                if attempt < retries - 1:
                    time.sleep(delay)
        return None

    def shutdown(self) -> None:
        # BatchSpanProcessor.shutdown owns the exporter lifecycle in
        # current SDKs — calling exporter.shutdown again would double
        # shut down and log a warning.
        self._processor.shutdown()
