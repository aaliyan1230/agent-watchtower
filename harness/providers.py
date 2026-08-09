"""LLM provider abstraction.

One interface, config-driven model per slot (worker / supervisor /
judge). FakeProvider keeps the harness runnable offline and
deterministic — the experiment needs seeded reproducibility before it
ever talks to a real model. GeminiProvider is the real client: keys via
the environment (harness/env.py), base URL overridable so tests can
point it at a mock server.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx

from .env import get_api_key

ToolCall = dict[str, Any]  # {"name": str, "args": dict}


@dataclass
class ProviderResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = "unknown"
    system: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0
    # The assistant turn as the API returned it, verbatim. Tool-calling
    # agents must echo it into history: Gemini's compat endpoint
    # requires the model's own message (incl. its internal
    # thought_signature) to precede the tool result.
    assistant_message: dict[str, Any] | None = None


class Provider(ABC):
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
        """Complete one chat turn. `contract` requests JSON output
        matching a structured-output contract."""


class FakeProvider(Provider):
    """Scripted provider: replays a fixed list of responses, so a run
    is byte-for-byte reproducible."""

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
        if resp.tool_calls:
            resp.assistant_message = {
                "role": "assistant",
                "tool_calls": [
                    {"id": tc.get("id") or f"call-{i}", "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc.get("args", {}))}}
                    for i, tc in enumerate(resp.tool_calls)
                ],
            }
        return resp


class GeminiProvider(Provider):
    """Gemini via its OpenAI-compatible endpoint. Retries transient
    failures once; the caller still sees errors for anything worse."""

    name = "gemini"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",  # best $/perf for agentic bulk; see .env.example
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.model = model
        # None means "read from env"; an explicit empty string means
        # "there is no key" and must not fall through to the env value.
        self._api_key = api_key if api_key is not None else get_api_key("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is missing: add it to .env")
        self._base = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._client = client or httpx.Client(timeout=60.0)

    def chat(self, messages, tools=None, *, contract=None, temperature=0.0) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = list(tools)
        if contract:
            payload["response_format"] = {"type": "json_object"}
        for attempt in range(2):
            try:
                return self._complete(payload)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (429, 500, 502, 503) or attempt == 1:
                    raise
                time.sleep(1)

    def _complete(self, payload: dict[str, Any]) -> ProviderResponse:
        resp = self._client.post(
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{resp.status_code}: {resp.text[:300]}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()
        msg = data["choices"][0]["message"]
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"_raw": raw_args}  # malformed args are data, not crashes
            tool_calls.append({"name": tc["function"]["name"], "args": args, "id": tc.get("id", "")})
        usage = data.get("usage", {})
        return ProviderResponse(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", self.model),
            system=self.name,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            assistant_message=msg,
        )


def tool_schema(name: str, description: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": args}}}


def render_tool_calls(tool_calls: Sequence[ToolCall]) -> str:
    return json.dumps([{"name": tc["name"], "args": tc.get("args", {})} for tc in tool_calls])
