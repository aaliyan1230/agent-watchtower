import copy
import json

from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext

from harness.trace_artifacts import (
    TraceRecorder,
    apply_evidence_fault,
    envelope_from_spans,
    report_fingerprint,
    sha256_bytes,
)


class _Exporter:
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
    def __init__(self, trace_id, span_id, name, *, parent=None, start=1_000_000_000, end=2_000_000_000, attrs=None):
        self.context = SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False)
        self.parent = parent
        self.name = name
        self.kind = 0
        self.start_time = start
        self.end_time = end
        self.attributes = attrs or {}
        self.events = ()
        self.resource = None
        self.status = None


def test_recorder_writes_clean_trace_and_final_manifest(tmp_path):
    trace_id = 0x01
    root = _Span(trace_id, 0x10, "agent.run", attrs={"watchtower.completed": True})
    child = _Span(trace_id, 0x11, "chat", parent=root.context, attrs={"gen_ai.usage.input_tokens": 4})
    inner = _Exporter()
    recorder = TraceRecorder(
        inner,
        tmp_path,
        {"benchmark": "test", "seed": 7, "behaviorFault": None},
    )

    assert recorder.export([child, root]) == SpanExportResult.SUCCESS
    trace_path = tmp_path / "trace-00000000000000000000000000000001.json"
    manifest_path = tmp_path / "manifest.jsonl"
    assert trace_path.exists()
    envelope = json.loads(trace_path.read_text())
    assert [span["name"] for span in envelope["spans"]] == ["agent.run", "chat"]
    assert envelope["spans"][1]["parentSpanId"] == "0000000000000010"
    assert len(inner.batches) == 1

    report = {"traceId": "00000000000000000000000000000001", "verdict": "PASS", "generatedAt": "now"}
    record = recorder.finalize(report, native_outcome="clean_reference")
    assert record["watchtowerVerdict"] == "PASS"
    assert record["reportInput"] == "clean-trace"
    assert record["traceSha256"] == sha256_bytes(envelope)
    assert record["reportSha256"] == report_fingerprint(report)
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["nativeOutcome"] == "clean_reference"
    assert (tmp_path / rows[0]["reportPath"]).exists()


def _envelope():
    return {
        "schemaVersion": "watchtower.trace.v1",
        "spans": [
            {
                "traceId": "a" * 32,
                "spanId": "1" * 16,
                "name": "agent.run",
                "startTime": "2026-08-15T00:00:00.000000000Z",
                "endTime": "2026-08-15T00:00:01.000000000Z",
                "attributes": {"watchtower.final": "true", "watchtower.output": "done"},
            },
            {
                "traceId": "a" * 32,
                "spanId": "2" * 16,
                "parentSpanId": "1" * 16,
                "name": "tool.call",
                "startTime": "2026-08-15T00:00:00.100000000Z",
                "endTime": "2026-08-15T00:00:00.200000000Z",
                "attributes": {"tool.call.id": "call-1"},
            },
        ],
    }


def test_mutation_is_copy_only_and_reorder_is_a_negative_control():
    source = _envelope()
    original = copy.deepcopy(source)
    reordered = apply_evidence_fault(source, "reorder_spans")
    assert [s["spanId"] for s in reordered["spans"]] == ["2" * 16, "1" * 16]
    assert source == original

    dropped = apply_evidence_fault(source, "drop_parent")
    assert len(dropped["spans"]) == 1
    assert dropped["spans"][0]["name"] == "tool.call"

    forged = apply_evidence_fault(source, "mismatch_tool_id")
    assert forged["spans"][1]["attributes"]["tool.call.id"] == "forged-tool-result-id"
    assert source["spans"][1]["attributes"]["tool.call.id"] == "call-1"

    truncated = apply_evidence_fault(source, "truncate_final")
    assert "watchtower.final" not in truncated["spans"][0].get("attributes", {})

    late = apply_evidence_fault(source, "late_span")
    assert late["spans"][1]["startTime"] > source["spans"][0]["endTime"]


def test_envelope_conversion_uses_stable_string_attributes():
    root = _Span(2, 3, "agent.run", attrs={"flag": True, "count": 4, "items": ["a", "b"]})
    envelope = envelope_from_spans([root])
    attrs = envelope["spans"][0]["attributes"]
    assert attrs == {"count": "4", "flag": "true", "items": '["a","b"]'}


def test_replay_derives_faulted_copy_without_rerunning_agent(tmp_path, monkeypatch):
    from experiments import replay as replay_module

    source = _envelope()
    source_path = tmp_path / "trace-clean.json"
    source_path.write_text(json.dumps(source))
    calls = []

    def fake_post(endpoint, envelope):
        calls.append(envelope)
        return {"traceId": "a" * 32, "verdict": "INCONCLUSIVE", "protocolVersion": "0.9"}

    monkeypatch.setattr(replay_module, "post_envelope", fake_post)
    report, derived, envelope = replay_module.replay(source_path, fault="drop_parent")
    assert report["verdict"] == "INCONCLUSIVE"
    assert len(calls) == 1
    assert len(calls[0]["spans"]) == 1
    assert derived.exists()
    assert json.loads(source_path.read_text()) == source
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["sourceTrace"] == source_path.name
    assert rows[0]["evidenceFault"] == "drop_parent"
