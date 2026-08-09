"""Supervisor: the meta-agent over workers.

Monitors worker progress via StepInfo callbacks and can halt a worker
(step/token limits), reroute a task to the next worker (repeated tool
failures), or escalate when nobody is left. Every decision is recorded
as a span event, so the Go verifiers see the supervisor's behavior —
the paper's "verification of meta-agents" angle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from opentelemetry import trace
from opentelemetry.trace import get_current_span

from . import semconv
from .workers import StepInfo, Worker


@dataclass
class SupervisorResult:
    task: str
    answers: dict[str, str] = field(default_factory=dict)
    halted: dict[str, str] = field(default_factory=dict)  # worker -> reason
    rerouted: dict[str, str] = field(default_factory=dict)  # worker -> successor
    steps_taken: dict[str, int] = field(default_factory=dict)
    escalation: bool = False
    trace_id: str = ""


@dataclass
class _WorkerState:
    steps: int = 0
    tokens: int = 0
    tool_failures: int = 0
    reason: str = ""


class Supervisor:
    def __init__(
        self,
        name: str,
        tracer: trace.Tracer,
        *,
        max_steps: int = 6,
        max_tokens: int = 4000,
        max_tool_failures: int = 2,
    ):
        self.name = name
        self._tracer = tracer
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._max_tool_failures = max_tool_failures

    def run(self, task: str, workers: list[Worker]) -> SupervisorResult:
        result = SupervisorResult(task=task)
        with self._tracer.start_as_current_span(
            "agent.run", attributes={semconv.AGENT_NAME: self.name, semconv.AGENT_ID: self.name}
        ) as root:
            result.trace_id = f"{root.get_span_context().trace_id:032x}"
            parent_ctx = trace.set_span_in_context(get_current_span())
            queue = list(workers)
            idx = 0
            while idx < len(queue):
                worker = queue[idx]
                state = _WorkerState()
                answer = worker.run(task, parent_context=parent_ctx, monitor=lambda info, w=worker, s=state: self._decide(info, w, s, queue, idx, root))
                if worker.rerouted:
                    if idx + 1 < len(queue):
                        result.rerouted[worker.name] = queue[idx + 1].name
                        idx += 1
                        continue
                    # Nobody left to take over: the reroute becomes a halt.
                    worker.halted = True
                    state.reason = "no successor worker after reroute"
                    self._record(root, worker.name, "halt", state.reason, state.steps)
                if worker.halted:
                    result.halted[worker.name] = state.reason or "halted"
                    result.escalation = True
                result.answers[worker.name] = answer
                result.steps_taken[worker.name] = state.steps
                idx += 1
        return result

    def _decide(self, info: StepInfo, worker: Worker, state: _WorkerState, queue: list[Worker], idx: int, root) -> str:
        state.steps += 1
        state.tokens += info.tokens
        if info.kind == "tool":
            state.tool_failures = state.tool_failures + 1 if not info.tool_ok else 0
            if state.tool_failures > self._max_tool_failures:
                state.reason = f"tool {info.tool} failed {state.tool_failures} times in a row"
                self._record(root, worker.name, "reroute", state.reason, info.step)
                return "reroute"
        if state.steps > self._max_steps:
            state.reason = "step limit exceeded"
            self._record(root, worker.name, "halt", state.reason, info.step)
            return "halt"
        if state.tokens > self._max_tokens:
            state.reason = "token limit exceeded"
            self._record(root, worker.name, "halt", state.reason, info.step)
            return "halt"
        return "continue"

    def _record(self, span, worker: str, action: str, reason: str, step: int) -> None:
        span.add_event(
            "supervisor.decision",
            {"worker": worker, "action": action, "reason": reason, "step": str(step)},
        )
