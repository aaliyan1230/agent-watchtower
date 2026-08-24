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
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import Tracer, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Link, SpanContext, set_tracer_provider

from .trace_artifacts import TraceRecorder

# The SDK allows one global provider per process; experiments create a
# telemetry per cell, so only the first sets it.
_global_provider_set = False

# A span id guaranteed not to appear in any exported batch, used by the
# orphan_link_target fault to model a causally-relevant span in flight.
_FOREIGN_SPAN_ID = 0xFEEDFACECAFEF00D


class EvidenceFaultExporter(SpanExporter):
    """Apply a transport fault immediately before OTLP serialization.

    The inner exporter remains the official OTLP/HTTP exporter. This
    wrapper changes only the exported batch (drop, duplicate, reorder,
    mismatch, truncation, or late timestamps), which keeps evidence faults
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
        elif self._fault == "drop_child":
            batch = [
                span
                for span in batch
                if not (span.name == "agent.run" and span.parent is not None)
            ]
        elif self._fault == "drop_tool_result":
            batch = [span for span in batch if span.name != "tool.call"]
        elif self._fault == "mismatch_tool_id":
            batch = self._rewrite_first_tool_id(batch)
        elif self._fault == "truncate_final":
            batch = self._truncate_final_markers(batch)
        elif self._fault == "truncate_closed":
            batch = self._truncate_closed_marker(batch)
        elif self._fault == "late_span":
            batch = self._move_first_child_late(batch)
        elif self._fault == "drop_attribute":
            batch = self._drop_required_attribute(batch)
        elif self._fault == "sample_spans":
            batch = [span for i, span in enumerate(batch) if i % 2 == 0]
        elif self._fault == "drop_link":
            batch = [self._strip_links(span) for span in batch]
        elif self._fault == "orphan_link_target":
            batch = self._orphan_first_link(batch)
        return self._inner.export(batch)

    @staticmethod
    def _rewrite_first_tool_id(batch):
        for i, span in enumerate(batch):
            if span.name != "tool.call":
                continue
            attrs = dict(span.attributes)
            attrs["tool.call.id"] = "forged-tool-result-id"
            batch[i] = _clone_span(span, attributes=attrs)
            break
        return batch

    @staticmethod
    def _truncate_final_markers(batch):
        for i, span in enumerate(batch):
            attrs = dict(span.attributes)
            changed = False
            for key in ("watchtower.final", "watchtower.output", "watchtower.contract"):
                if key in attrs:
                    del attrs[key]
                    changed = True
            if changed:
                batch[i] = _clone_span(span, attributes=attrs)
        return batch

    @staticmethod
    def _truncate_closed_marker(batch):
        """Remove the trace-closure marker so the trace looks still open.
        The three-valued verifier must then answer UNKNOWN, not PASS —
        the premature-pass hazard made observable."""
        for i, span in enumerate(batch):
            attrs = dict(span.attributes)
            if "watchtower.trace.closed" in attrs:
                del attrs["watchtower.trace.closed"]
                batch[i] = _clone_span(span, attributes=attrs)
        return batch

    @staticmethod
    def _move_first_child_late(batch):
        latest_end = max((span.end_time or 0) for span in batch)
        for i, span in enumerate(batch):
            if span.parent is None or span.start_time is None or span.end_time is None:
                continue
            duration = max(1_000_000, span.end_time - span.start_time)
            start = latest_end + 1_000_000
            batch[i] = _clone_span(span, start_time=start, end_time=start + duration)
            break
        return batch

    @staticmethod
    def _drop_required_attribute(batch):
        for i, span in enumerate(batch):
            if span.name != "chat":
                continue
            attrs = dict(span.attributes)
            attrs.pop("gen_ai.system", None)
            batch[i] = _clone_span(span, attributes=attrs)
            break
        return batch

    @staticmethod
    def _strip_links(span):
        """Remove every causal link from a span, so the run graph sees
        no edges between sibling spans even though the worker produced
        them. The verifier must notice the missing causal structure."""
        if not span.links:
            return span
        return _clone_span(
            span,
            attributes=dict(span.attributes),
            links=(),
        )

    @staticmethod
    def _orphan_first_link(batch):
        """Rewrite the first link's target to an id that is not in the
        batch, modeling a causally-relevant span still in flight. The
        reconstructed run then reports a missing link target instead of
        an edge — exactly the premature-PASS hazard under late arrivals."""
        for i, span in enumerate(batch):
            if not span.links:
                continue
            original = span.links[0]
            orphan = Link(
                SpanContext(
                    trace_id=original.context.trace_id,
                    span_id=_FOREIGN_SPAN_ID,
                    is_remote=True,
                ),
                attributes=dict(original.attributes) if original.attributes else None,
            )
            new_links = (orphan,) + span.links[1:]
            imported = _clone_span(
                span, attributes=dict(span.attributes), links=new_links
            )
            batch[i] = imported
            break
        return batch

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


class HarnessTelemetry:
    """Owns the tracer used by workers and supervisor. BatchSpanProcessor
    accumulates spans until flush() — one export per run, which is what
    the Go graph reconstructor expects (one trace per request)."""

    def __init__(
        self,
        endpoint: str,
        service_name: str = "harness",
        evidence_fault: str | None = None,
        trace_dir: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ):
        global _global_provider_set
        self._endpoint = endpoint.rstrip("/")
        # An explicit `endpoint` is used verbatim as the export URL (the
        # /v1/traces suffix is only appended for the env-var default),
        # so the full path goes here.
        exporter: SpanExporter = OTLPSpanExporter(
            endpoint=self._endpoint + "/v1/traces"
        )
        if evidence_fault:
            exporter = EvidenceFaultExporter(exporter, evidence_fault)
        self._recorder: TraceRecorder | None = None
        if trace_dir:
            # This wrapper sits outside EvidenceFaultExporter, so it sees a
            # clean batch while the inner exporter sends the mutated copy.
            self._recorder = TraceRecorder(exporter, trace_dir, trace_metadata)
            exporter = self._recorder
        self._exporter = exporter
        provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: service_name})
        )
        # The scheduled export must never fire mid-run: the wire
        # contract is one export request per run (force_flush below),
        # and a 5s-scheduled partial batch would split one trace into
        # two POSTs — the server would verify only the second half and
        # could miss early spans (a natural false all-clear, observed
        # live on tau-bench). One minute is long enough for any run
        # here; only the explicit flush ships the trace.
        self._processor = BatchSpanProcessor(
            self._exporter, schedule_delay_millis=60_000
        )
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

    def finalize_trace(
        self,
        report: dict[str, Any] | None,
        *,
        native_outcome: str | None = None,
    ) -> dict[str, Any] | None:
        """Attach the report to the durable trace manifest when enabled."""
        if self._recorder is None:
            return None
        return self._recorder.finalize(report, native_outcome=native_outcome)

    def fetch_report(
        self, trace_id: str, retries: int = 5, delay: float = 0.1
    ) -> dict[str, Any] | None:
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


def _clone_span(span, *, attributes=None, start_time=None, end_time=None, links=None):
    """Clone an SDK span while changing only the evidence fault field.

    The exporter receives immutable ReadableSpan objects. Keeping the
    clone at this boundary makes faults transport-only and leaves agent
    execution untouched.
    """
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=dict(span.attributes) if attributes is None else attributes,
        events=span.events,
        links=span.links if links is None else links,
        kind=span.kind,
        instrumentation_info=getattr(span, "instrumentation_info", None),
        status=span.status,
        start_time=span.start_time if start_time is None else start_time,
        end_time=span.end_time if end_time is None else end_time,
        instrumentation_scope=getattr(span, "instrumentation_scope", None),
    )
