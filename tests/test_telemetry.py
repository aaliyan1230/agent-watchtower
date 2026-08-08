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
