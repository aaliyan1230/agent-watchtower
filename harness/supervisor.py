"""Supervisor: the meta-agent over workers.

Phase 1 turns this into a real agent (monitoring worker step events,
able to halt / reroute / escalate). For the bootstrap it does what a
supervisor must do for the *verifier* story to work: own the run's root
span, run workers underneath it, and return a summary. The supervisor
is itself an agent, and its spans are verified like any other — that is
the paper's "verification of meta-agents" angle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from opentelemetry import trace
from opentelemetry.trace import get_current_span

from . import semconv
from .workers import Worker


@dataclass
class SupervisorResult:
    task: str
    answers: dict[str, str] = field(default_factory=dict)
    steps_taken: dict[str, int] = field(default_factory=dict)


class Supervisor:
    def __init__(self, name: str, tracer: trace.Tracer):
        self.name = name
        self._tracer = tracer

    def run(self, task: str, workers: list[Worker]) -> SupervisorResult:
        """Run the task with each worker, under one run-span per worker.
        The root supervisor span frames the whole run so the Go graph
        reconstructor sees one trace with the supervisor on top."""
        result = SupervisorResult(task=task)
        with self._tracer.start_as_current_span(
            "agent.run", attributes={semconv.AGENT_NAME: self.name, semconv.AGENT_ID: self.name}
        ):
            parent_ctx = trace.set_span_in_context(get_current_span())
            for worker in workers:
                answer = worker.run(task, parent_context=parent_ctx)
                result.answers[worker.name] = answer
                result.steps_taken[worker.name] = worker.max_steps
        return result
