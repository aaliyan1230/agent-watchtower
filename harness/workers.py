"""Worker agent: the minimal agent loop that the verifiers police.

Loop shape: think -> (tool call -> observe)* -> final answer, bounded by
max_steps. Every turn and every tool call becomes a span carrying the
semconv attributes the Go verifiers read; the loop is deliberately
simple because the interesting machinery is *outside* it — fault
injection, telemetry, and the verifier that catches the agent's
mistakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from opentelemetry import context, trace
from opentelemetry.trace import get_current_span

from . import semconv
from .providers import Provider, ProviderResponse, tool_schema


@dataclass
class Tool:
    """A callable the agent may invoke."""

    name: str
    description: str
    args_schema: dict[str, Any]
    func: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return tool_schema(self.name, self.description, self.args_schema)


@dataclass
class StepInfo:
    """One completed turn or tool call, handed to the supervisor's
    monitor so it can decide whether to intervene."""

    worker: str
    step: int
    kind: str  # "llm" | "tool"
    tool: str | None = None
    tool_ok: bool | None = None
    tokens: int = 0


class Worker:
    """A tool-calling worker agent. `contract` names a structured-output
    contract the final answer must honor; the harness stamps it on the
    span so the schema verifier can check the raw output."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        provider: Provider,
        tools: list[Tool],
        tracer: trace.Tracer,
        *,
        contract: str | None = None,
        max_steps: int = 5,
        agent_id: str = "",
    ):
        self.name = name
        self._system_prompt = system_prompt
        self._provider = provider
        self._tools = {t.name: t for t in tools}
        self._tracer = tracer
        self.contract = contract
        self.max_steps = max_steps
        self.agent_id = agent_id
        self.halted = False
        self.rerouted = False

    def run(self, task: str, parent_context=None, monitor: Callable[[StepInfo], str] | None = None) -> str:
        """Run one task to completion (or max_steps) and return the
        final answer. The monitor, if any, gets a StepInfo after every
        turn and tool call and may return "halt" or "reroute" to stop
        the loop."""
        self.halted = False
        self.rerouted = False
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task},
        ]
        tool_schemas = [t.schema() for t in self._tools.values()]
        token = self._tracer.start_as_current_span(
            "agent.run", context=parent_context, attributes={semconv.AGENT_NAME: self.name}
        )

        with token:
            for step in range(1, self.max_steps + 1):
                resp = self._turn(messages, tool_schemas, step)
                if resp.tool_calls:
                    for tc in resp.tool_calls:
                        ok, result = self._exec_tool(tc, step)
                        messages.append({"role": "tool", "content": result})
                        if monitor:
                            action = monitor(StepInfo(
                                worker=self.name, step=step, kind="tool",
                                tool=tc["name"], tool_ok=ok,
                                tokens=resp.input_tokens + resp.output_tokens,
                            ))
                            if action != "continue":
                                return self._stop(action)
                    continue
                if monitor:
                    action = monitor(StepInfo(worker=self.name, step=step, kind="llm", tokens=resp.input_tokens + resp.output_tokens))
                    if action != "continue":
                        return self._stop(action)
                return resp.content.strip()
        return "(no final answer within max_steps)"

    def _stop(self, action: str) -> str:
        if action == "reroute":
            self.rerouted = True
            return "(rerouted by supervisor)"
        self.halted = True
        return "(halted by supervisor)"

    def _turn(self, messages, tool_schemas, step: int) -> ProviderResponse:
        with self._tracer.start_as_current_span(
            "chat",
            attributes={
                semconv.AGENT_NAME: self.name,
                semconv.STEP_INDEX: str(step),
                semconv.GEN_AI_OPERATION_NAME: "chat",
                semconv.GEN_AI_SYSTEM: self._provider.name,
                semconv.GEN_AI_REQUEST_MODEL: getattr(self._provider, "model", "unknown"),
                semconv.WATCHTOWER_CONTRACT: self.contract or "",
            },
        ):
            resp = self._provider.chat(messages, tools=tool_schemas, contract=self.contract)
            span = get_current_span()
            span.set_attribute(semconv.GEN_AI_INPUT_TOKENS, str(resp.input_tokens))
            span.set_attribute(semconv.GEN_AI_OUTPUT_TOKENS, str(resp.output_tokens))
            span.set_attribute(semconv.GEN_AI_RESPONSE_MODEL, resp.model)
            # Stamp the raw output while this span is still active: the
            # schema verifier reads contract + output from the same span.
            if self.contract and resp.content:
                span.set_attribute(semconv.WATCHTOWER_OUTPUT, resp.content)
            return resp

    def _exec_tool(self, tool_call: dict[str, Any], step: int) -> tuple[bool, str]:
        name, args = tool_call["name"], tool_call.get("args", {})

        with self._tracer.start_as_current_span(
            "tool.call",
            attributes={
                semconv.AGENT_NAME: self.name,
                semconv.STEP_INDEX: str(step),
                semconv.TOOL_NAME: name,
                semconv.WATCHTOWER_TOOL_ARGS: json.dumps(args),
            },
        ):
            tool = self._tools.get(name)
            if tool is None:
                get_current_span().set_attribute(semconv.TOOL_RESULT_OK, "false")
                return False, f"unknown tool {name}"
            try:
                result = tool.func(**args)
                get_current_span().set_attribute(semconv.TOOL_RESULT_OK, "true")
                return True, str(result)
            except Exception as exc:  # tool errors are data, not crashes
                get_current_span().set_attribute(semconv.TOOL_RESULT_OK, "false")
                get_current_span().set_attribute(semconv.TOOL_RESULT_MSG, str(exc))
                return False, f"tool error: {exc}"
