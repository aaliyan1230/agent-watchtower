"""Telemetry: span serialization shape + envelope batching."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from harness.telemetry import _span_to_json


class RecordingExporter(SpanExporter):
    """Captures exported spans without any HTTP — the unit-test twin
    of WatchtowerExporter's serialization path."""

    def __init__(self):
        self.envelopes = []

    def export(self, spans):
        self.envelopes.append(list(spans))
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def record_spans(actions) -> list:
    exporter = RecordingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    actions(tracer)
    return exporter.envelopes


def test_span_serialization_shape():
    envelopes = record_spans(
        lambda tracer: tracer.start_as_current_span(
            "chat",
            attributes={"gen_ai.system": "fake", "gen_ai.usage.input_tokens": 7},
        ).__enter__()
    )
    spans = envelopes[0]
    assert len(spans) == 1
    doc = _span_to_json(spans[0])
    assert doc["name"] == "chat"
    assert doc["attributes"]["gen_ai.system"] == "fake"
    assert doc["attributes"]["gen_ai.usage.input_tokens"] == "7"
    assert doc["startTime"].endswith("Z")
    assert doc["status"] == "unset"
    # trace/span ids must be hex strings the Go model accepts
    assert len(doc["traceId"]) == 32
    assert len(doc["spanId"]) == 16


def test_span_parent_and_status():
    # Spans must be *ended* before the exporter sees them, and status
    # must be set while the span is still writable.
    def action(tracer):
        with tracer.start_as_current_span("root") as root:
            with tracer.start_as_current_span("child", attributes={"tool.result.ok": True}):
                pass
        return root

    spans = [s for env in record_spans(action) for s in env]
    docs = [_span_to_json(s) for s in spans]
    child_doc = next(d for d in docs if "parentSpanId" in d)
    root_doc = next(d for d in docs if "parentSpanId" not in d)
    assert child_doc["parentSpanId"] == root_doc["spanId"]
    assert child_doc["attributes"]["tool.result.ok"] == "true"


def test_error_status_maps():
    def action(tracer):
        with tracer.start_as_current_span("boom") as span:
            span.set_status(StatusCode.ERROR, "kaboom")

    doc = _span_to_json(record_spans(action)[0][0])
    assert doc["status"] == "error"
    assert doc["statusMessage"] == "kaboom"
