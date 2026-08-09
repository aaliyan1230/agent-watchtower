"""Supervisor intervention: halt on limits, reroute on tool failures,
escalation when nobody is left."""

from opentelemetry.sdk.trace import TracerProvider

from harness.providers import FakeProvider, ProviderResponse
from harness.supervisor import Supervisor
from harness.workers import Tool, Worker

TRACER = TracerProvider().get_tracer("test")

TOOL_TURN = ProviderResponse(tool_calls=[{"name": "boom", "args": {}}])
GOOD_TOOLS = [
    Tool("boom", "always fails", {}, lambda: (_ for _ in ()).throw(RuntimeError("kaboom"))),
    Tool("add", "add two numbers", {"a": {"type": "int"}, "b": {"type": "int"}}, lambda a, b: a + b),
]
OK_TOOL_TURN = ProviderResponse(tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}}])


def make_worker(name, script, max_steps=5, tools=None, tokens=0):
    script = [ProviderResponse(**{**r.__dict__, "input_tokens": tokens}) if tokens else r for r in script]
    return Worker(name, "short", FakeProvider(script), tools or [], TRACER, max_steps=max_steps)


def test_halt_on_step_limit():
    script = [OK_TOOL_TURN] * 5
    worker = make_worker("a", script, max_steps=5, tools=GOOD_TOOLS)
    result = Supervisor("sup", TRACER, max_steps=3).run("task", [worker])
    assert result.halted == {"a": "step limit exceeded"}
    assert result.escalation
    assert result.answers["a"] == "(halted by supervisor)"


def test_halt_on_token_limit():
    worker = make_worker("a", [TOOL_TURN] * 3, tokens=1000)
    result = Supervisor("sup", TRACER, max_tokens=1500).run("task", [worker])
    assert result.halted["a"] == "token limit exceeded"


def test_reroute_after_repeated_tool_failures():
    failing = make_worker("a", [ProviderResponse(tool_calls=[{"name": "boom", "args": {}}])] * 5, tools=GOOD_TOOLS)
    succeeding = make_worker("b", [ProviderResponse(content="done")], tools=GOOD_TOOLS)
    result = Supervisor("sup", TRACER, max_tool_failures=2).run("task", [failing, succeeding])
    assert result.rerouted == {"a": "b"}
    assert not result.escalation
    assert result.answers["b"] == "done"
    assert failing.rerouted and not failing.halted


def test_escalation_when_no_successor():
    failing = make_worker("a", [ProviderResponse(tool_calls=[{"name": "boom", "args": {}}])] * 5, tools=GOOD_TOOLS)
    result = Supervisor("sup", TRACER, max_tool_failures=1).run("task", [failing])
    assert result.escalation
    assert result.halted["a"] == "no successor worker after reroute"


def test_clean_run_no_intervention():
    worker = make_worker("a", [ProviderResponse(content="fine")])
    result = Supervisor("sup", TRACER).run("task", [worker])
    assert not result.halted and not result.rerouted and not result.escalation
    assert result.answers["a"] == "fine"


def test_worker_monitor_stops_loop():
    worker = make_worker("a", [TOOL_TURN] * 5)
    worker.run("task", monitor=lambda info: "halt")
    assert worker.halted
