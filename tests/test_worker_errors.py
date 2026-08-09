"""Provider failures surface as errored spans, not crashes."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from harness.providers import FakeProvider, ProviderResponse
from harness.workers import Worker


class RecordingExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def make_recording_worker(script, **kw) -> tuple[Worker, RecordingExporter]:
    exporter = RecordingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    return Worker("w", "short", FakeProvider(script), [], tracer, **kw), exporter


def test_provider_error_marks_span_error():
    def boom(*args, **kwargs):
        raise TimeoutError("provider timed out")

    worker, exporter = make_recording_worker([], max_steps=2)
    worker._provider.chat = boom
    answer = worker.run("task")
    assert answer.startswith("(provider error:")
    error_spans = [s for s in exporter.spans if s.status.status_code == StatusCode.ERROR]
    assert len(error_spans) == 1
    assert "timed out" in error_spans[0].status.description


def test_normal_run_has_no_error_spans():
    worker, exporter = make_recording_worker([ProviderResponse(content="ok")])
    assert worker.run("task") == "ok"
    assert not any(s.status.status_code == StatusCode.ERROR for s in exporter.spans)
