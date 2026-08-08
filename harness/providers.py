"""LLM provider abstraction.

One interface, config-driven model per slot (worker / supervisor /
judge). The FakeProvider keeps the harness runnable offline and
deterministic — the experiment needs seeded reproducibility before it
ever talks to a real model. Real providers (Gemini, Bedrock) arrive in
Phase 1; GeminiProvider below is the honest stub they will replace.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

ToolCall = dict[str, Any]  # {"name": str, "args": dict}


@dataclass
class ProviderResponse:
    """A model turn: either free text, tool calls, or both."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = "unknown"
    system: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0


class Provider(ABC):
    """Abstract model client. All calls are telemetry-instrumented by
    the worker, not by the provider — the provider stays a thin wire
    client, which is what makes fault injection a pure wrapper."""

    name: str = "abstract"

    @abstractmethod
    def chat(
        self,
        messages: Sequence[dict[str, str]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        contract: str | None = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        """Complete one chat turn. `contract` names a structured-output
        contract the caller expects in the response content."""


class FakeProvider(Provider):
    """Scripted provider: replays a fixed list of responses, so a run
    is byte-for-byte reproducible. Token counts are stubs; real counts
    come from the provider's usage payload in Phase 1."""

    name = "fake"

    def __init__(self, script: Sequence[ProviderResponse], model: str = "fake-flash"):
        self._script = list(script)
        self._cursor = 0
        self.model = model

    def chat(self, messages, tools=None, *, contract=None, temperature=0.0):
        if self._cursor >= len(self._script):
            raise RuntimeError(f"FakeProvider script exhausted after {len(self._script)} turns")
        resp = self._script[self._cursor]
        self._cursor += 1
        return resp


class GeminiProvider(Provider):
    """Stub for the real Gemini client (Phase 1). Deliberately raises:
        failing loudly beats pretending to work offline.

    Phase 1 replaces this with an httpx client against the
    OpenAI-compatible endpoint, keys from the environment, retry with
    backoff, and structured-output retry loops."""

    name = "gemini"

    def chat(self, messages, tools=None, *, contract=None, temperature=0.0):
        raise NotImplementedError("GeminiProvider lands in Phase 1; use FakeProvider offline")


def tool_schema(name: str, description: str, args: dict[str, Any]) -> dict[str, Any]:
    """Build the tool definition the fake provider and real APIs both
    understand: OpenAI-style function schema."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": args}}}


def render_tool_calls(tool_calls: Sequence[ToolCall]) -> str:
    """Serialize tool calls for the model's next context turn."""
    return json.dumps([{"name": tc["name"], "args": tc.get("args", {})} for tc in tool_calls])
