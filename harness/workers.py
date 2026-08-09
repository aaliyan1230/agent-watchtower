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

    def run(self, task: str, parent_context=None) -> str:
        """Run one task to completion (or max_steps) and return the
        final answer. Each iteration is an LLM span; each executed tool
        call is a child tool span."""
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
                        result = self._exec_tool(tc, step)
                        messages.append({"role": "tool", "content": result})
                    continue
                return resp.content.strip()
        return "(no final answer within max_steps)"

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

    def _exec_tool(self, tool_call: dict[str, Any], step: int) -> str:
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
                msg = f"unknown tool {name}"
                get_current_span().set_attribute(semconv.TOOL_RESULT_OK, "false")
                return msg
            try:
                result = tool.func(**args)
                get_current_span().set_attribute(semconv.TOOL_RESULT_OK, "true")
                return str(result)
            except Exception as exc:  # tool errors are data, not crashes
                get_current_span().set_attribute(semconv.TOOL_RESULT_OK, "false")
                get_current_span().set_attribute(semconv.TOOL_RESULT_MSG, str(exc))
                return f"tool error: {exc}"
