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
from opentelemetry.trace import Link, Status, StatusCode, get_current_span

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


def _strip_fences(content: str) -> str:
    """Gemini often wraps JSON answers in markdown fences; the schema
    verifier needs the raw document, so the harness normalizes."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


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
        self._last_tool_ctx = None

    def run(
        self,
        task: str,
        parent_context=None,
        monitor: Callable[[StepInfo], str] | None = None,
    ) -> str:
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
        with self._tracer.start_as_current_span(
            "agent.run",
            context=parent_context,
            attributes={semconv.AGENT_NAME: self.name},
        ) as run_span:
            final = False
            answer = ""
            try:
                for step in range(1, self.max_steps + 1):
                    resp = self._turn(messages, tool_schemas, step)
                    if resp.tool_calls:
                        # The assistant turn must precede its tool results
                        # in history (Gemini's compat endpoint requires the
                        # model's own message, thought_signature included).
                        if resp.assistant_message:
                            messages.append(resp.assistant_message)
                        for tc in resp.tool_calls:
                            ok, result = self._exec_tool(tc, step)
                            messages.append(
                                {
                                    "role": "tool",
                                    "content": result,
                                    "tool_call_id": tc.get("id", ""),
                                }
                            )
                            if monitor:
                                action = monitor(
                                    StepInfo(
                                        worker=self.name,
                                        step=step,
                                        kind="tool",
                                        tool=tc["name"],
                                        tool_ok=ok,
                                        tokens=resp.input_tokens + resp.output_tokens,
                                    )
                                )
                                if action != "continue":
                                    return self._stop(action)
                        continue
                    if monitor:
                        action = monitor(
                            StepInfo(
                                worker=self.name,
                                step=step,
                                kind="llm",
                                tokens=resp.input_tokens + resp.output_tokens,
                            )
                        )
                        if action != "continue":
                            return self._stop(action)
                    answer = _strip_fences(resp.content)
                    final = bool(answer)
                    return answer
                return "(no final answer within max_steps)"
            finally:
                # The marker is the evidence obligation the Go side checks;
                # absence means the trace cannot establish that the agent
                # span completed normally.
                run_span.set_attribute(semconv.WATCHTOWER_COMPLETED, True)
                if final:
                    run_span.set_attribute(semconv.WATCHTOWER_FINAL, True)
                    run_span.set_attribute(semconv.WATCHTOWER_OUTPUT, answer)

    def _stop(self, action: str) -> str:
        if action == "reroute":
            self.rerouted = True
            return "(rerouted by supervisor)"
        self.halted = True
        return "(halted by supervisor)"

    def _data_dependency_links(self) -> list[Link] | None:
        """Return a link from the next chat turn to the tool span whose
        result it consumes, if one ran in the previous turn. A worker
        observing a tool result and then reasoning about it is a real
        causal dependency that sibling spans under the same agent.run
        parent cannot express — that edge is what order-invariant
        verification needs."""
        ctx = getattr(self, "_last_tool_ctx", None)
        if ctx is None:
            return None
        return [Link(ctx, attributes={semconv.LINK_PURPOSE: "data"})]

    def _turn(self, messages, tool_schemas, step: int) -> ProviderResponse:
        links = self._data_dependency_links()
        with self._tracer.start_as_current_span(
            "chat",
            links=links,
            attributes={
                semconv.AGENT_NAME: self.name,
                semconv.STEP_INDEX: str(step),
                semconv.GEN_AI_OPERATION_NAME: "chat",
                semconv.GEN_AI_SYSTEM: self._provider.name,
                semconv.GEN_AI_REQUEST_MODEL: getattr(
                    self._provider, "model", "unknown"
                ),
            },
        ) as span:
            try:
                # The contract is harness-level; asking the API for
                # json_object output on tool-calling turns makes Gemini
                # loop on tool calls (observed live), so the prompt
                # carries the format request instead.
                resp = self._provider.chat(messages, tools=tool_schemas)
            except Exception as exc:
                # Provider failures are evidence, not crashes: mark the
                # span errored (the status verifier catches it) and let
                # the loop terminate with a visible answer.
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                return ProviderResponse(content=f"(provider error: {exc})")
            span.set_attribute(semconv.GEN_AI_INPUT_TOKENS, str(resp.input_tokens))
            span.set_attribute(semconv.GEN_AI_OUTPUT_TOKENS, str(resp.output_tokens))
            span.set_attribute(semconv.GEN_AI_RESPONSE_MODEL, resp.model)
            if resp.tool_calls:
                span.set_attribute(
                    semconv.WATCHTOWER_TOOL_CALL_IDS,
                    json.dumps([tc.get("id", "") for tc in resp.tool_calls]),
                )
            # The contract is a promise about the final structured
            # output, so contract+output are stamped only on answer
            # turns — a tool-call turn has no output to check.
            if resp.content and not resp.tool_calls:
                output = _strip_fences(resp.content)
                span.set_attribute(semconv.WATCHTOWER_FINAL, True)
                span.set_attribute(semconv.WATCHTOWER_OUTPUT, output)
                if self.contract:
                    span.set_attribute(semconv.WATCHTOWER_CONTRACT, self.contract)
            return resp

    def _exec_tool(self, tool_call: dict[str, Any], step: int) -> tuple[bool, str]:
        name, args = tool_call["name"], tool_call.get("args", {})

        call_id = tool_call.get("id", "")
        with self._tracer.start_as_current_span(
            "tool.call",
            attributes={
                semconv.AGENT_NAME: self.name,
                semconv.STEP_INDEX: str(step),
                semconv.TOOL_NAME: name,
                semconv.TOOL_CALL_ID: call_id,
                semconv.WATCHTOWER_TOOL_ARGS: json.dumps(args),
            },
        ) as tool_span:
            # The next chat turn consumes this span's result, so it must
            # be visible as a causal predecessor: sibling spans under the
            # same agent.run parent carry no ordering, and links exist
            # exactly to record that an observation happened before the
            # reasoning it informed.
            self._last_tool_ctx = tool_span.get_span_context()
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
