"""Telemetry after Phase 0.6: the harness talks real OTLP, so the unit
tests cover what we still own — the trace-id capture and report
fetching contract — while the wire format itself is the official SDK's
problem (covered end-to-end by the Go OTLP adapter tests and the demo).
"""

import re

import pytest

from harness.supervisor import Supervisor
from harness.providers import FakeProvider, ProviderResponse
from harness.workers import Worker
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult


def test_supervisor_captures_trace_id():
    tracer = TracerProvider().get_tracer("test")
    worker = Worker(
        "w",
        "short",
        FakeProvider([ProviderResponse(content="done")]),
        [],
        tracer,
        max_steps=1,
    )
    result = Supervisor("supervisor", tracer).run("task", [worker])
    assert re.fullmatch(r"[0-9a-f]{32}", result.trace_id)
    assert result.answers == {"w": "done"}


def test_fetch_report_raises_on_server_error():
    from harness.telemetry import HarnessTelemetry

    telemetry = HarnessTelemetry("http://127.0.0.1:1")  # nothing listens here
    with pytest.raises(Exception):
        telemetry.fetch_report("0" * 32, retries=1, delay=0)
    telemetry.shutdown()


class _StubExporter:
    def __init__(self):
        self.batches = []

    def export(self, spans):
        self.batches.append(list(spans))
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


class _Span:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent


def test_evidence_fault_exporter_changes_only_export_batch():
    from harness.telemetry import EvidenceFaultExporter

    root = _Span("agent.run")
    child = _Span("chat", parent=root)

    dropped_inner = _StubExporter()
    assert EvidenceFaultExporter(dropped_inner, "drop_parent").export([root, child]) == SpanExportResult.SUCCESS
    assert dropped_inner.batches[-1] == [child]

    duplicate_inner = _StubExporter()
    EvidenceFaultExporter(duplicate_inner, "duplicate_span").export([root, child])
    assert duplicate_inner.batches[-1] == [root, child, child]

    reorder_inner = _StubExporter()
    EvidenceFaultExporter(reorder_inner, "reorder_spans").export([root, child])
    assert reorder_inner.batches[-1] == [child, root]
