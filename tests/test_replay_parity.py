"""Replay must derive the same evidence fault the live exporter injects,
or a replayed artifact silently measures a different mutation."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from harness.faults import EvidenceFaultKind
from harness.providers import FakeProvider, ProviderResponse
from harness.supervisor import Supervisor
from harness.telemetry import EvidenceFaultExporter
from harness.trace_artifacts import apply_evidence_fault, envelope_from_spans
from harness.workers import Tool, Worker

# These faults pick "first"/"every other"/"last" span, so their result
# depends on batch order, which differs between SDK export order (end
# time) and the canonical envelope order (start time).
ORDER_DEPENDENT = {"duplicate_span", "reorder_spans", "late_span", "sample_spans"}


class _Capture:
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


def _harness_batch():
    capture = _Capture()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(capture))
    tracer = provider.get_tracer("parity")
    worker = Worker(
        "w",
        "triage",
        FakeProvider(
            [
                ProviderResponse(tool_calls=[{"name": "search", "args": {"q": "x"}}], input_tokens=3, output_tokens=1),
                ProviderResponse(content='{"id": 1, "title": "t"}', input_tokens=2, output_tokens=1),
            ]
        ),
        [Tool("search", "search", {"q": {"type": "string"}}, lambda q: "hit")],
        tracer,
        contract="ticket",
        max_steps=3,
    )
    Supervisor("supervisor", tracer).run("task", [worker])
    return capture.spans


def test_harness_batch_exercises_every_mutation_target():
    envelope = envelope_from_spans(_harness_batch())
    names = {s["name"] for s in envelope["spans"]}
    assert {"agent.run", "chat", "tool.call"} <= names
    assert any(s.get("links") for s in envelope["spans"])
    assert any("watchtower.trace.closed" in (s.get("attributes") or {}) for s in envelope["spans"])


@pytest.mark.parametrize("fault", [k.value for k in EvidenceFaultKind])
def test_replay_supports_every_evidence_fault(fault):
    apply_evidence_fault(envelope_from_spans(_harness_batch()), fault)


@pytest.mark.parametrize("fault", [k.value for k in EvidenceFaultKind if k.value not in ORDER_DEPENDENT])
def test_replay_matches_live_exporter(fault):
    batch = _harness_batch()
    inner = _Capture()
    EvidenceFaultExporter(inner, fault).export(list(batch))
    live = envelope_from_spans(inner.spans)
    replayed = apply_evidence_fault(envelope_from_spans(batch), fault)
    assert replayed == live
