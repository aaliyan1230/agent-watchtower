"""Worker loop behavior: tool execution, final answers, step bounds."""

from opentelemetry.sdk.trace import TracerProvider

from harness.faults import FaultInjector, FaultKind, FaultSpec
from harness.providers import FakeProvider, ProviderResponse
from harness.workers import Tool, Worker

TOOLS = [Tool("add", "add two numbers", {"a": {"type": "int"}, "b": {"type": "int"}}, lambda a, b: a + b)]


def make_worker(script, **kw) -> Worker:
    provider = FakeProvider(script)
    return Worker("w", "you are a calculator", provider, TOOLS, TracerProvider().get_tracer("t"), **kw)


def test_worker_tools_then_answer():
    w = make_worker(
        [
            ProviderResponse(tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}}], output_tokens=5),
            ProviderResponse(content="the answer is 3", output_tokens=5),
        ]
    )
    assert w.run("1+2?") == "the answer is 3"


def test_worker_direct_answer():
    w = make_worker([ProviderResponse(content="hello", output_tokens=2)])
    assert w.run("say hi") == "hello"


def test_worker_unknown_tool_does_not_crash():
    w = make_worker([ProviderResponse(tool_calls=[{"name": "nope", "args": {}}]), ProviderResponse(content="done")])
    assert w.run("x") == "done"


def test_worker_step_cap():
    # Script answers in tool calls only; the step cap must stop the loop.
    tool_turn = ProviderResponse(tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}}])
    w = make_worker([tool_turn, tool_turn, tool_turn, tool_turn, tool_turn], max_steps=3)
    assert "(no final answer" in w.run("x")


def test_worker_contract_fault_deterministic():
    ok = ProviderResponse(content='{"id": 1, "title": "t"}', output_tokens=3)
    faulted = FaultInjector(FakeProvider([ok]), FaultSpec(FaultKind.SCHEMA_VIOLATION, seed=4))
    w = Worker("w", "emit json", faulted, [], TracerProvider().get_tracer("t"), contract="ticket")
    answer = w.run("make a ticket")
    assert '"id": "not-an-int"' in answer
