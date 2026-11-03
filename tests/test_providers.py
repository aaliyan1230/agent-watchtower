"""OpenAI-compatible providers (Gemini direct, OpenRouter) against a mock
transport: request shape, response parsing, error handling — no live
network involved."""

import json

import httpx
import pytest

from harness.providers import GeminiProvider, OpenRouterProvider, ProviderResponse


def make_provider(handler) -> GeminiProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiProvider(api_key="test-key", base_url="http://mock", client=client)


def test_missing_key_raises():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key="", base_url="http://mock")


def test_request_shape_and_response_parsing():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = request.read().decode()
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "model": "gemini-2.5-flash",
                "choices": [{
                    "message": {
                        "content": "the answer is 3",
                        "tool_calls": [{"id": "call_1", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}],
                    }
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    p = make_provider(handler)
    resp = p.chat(
        [{"role": "user", "content": "1+2?"}],
        tools=[{"type": "function", "function": {"name": "add"}}],
        contract="calc",
    )
    assert captured["url"] == "http://mock/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    import json as _json

    body = _json.loads(captured["json"])
    assert body["model"] == "gemini-3.5-flash-lite"
    assert body["response_format"] == {"type": "json_object"}
    assert body["tools"][0]["function"]["name"] == "add"

    assert isinstance(resp, ProviderResponse)
    assert resp.content == "the answer is 3"
    assert resp.tool_calls == [{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}]
    assert resp.input_tokens == 10 and resp.output_tokens == 5
    assert resp.system == "gemini"


def test_malformed_tool_arguments_are_data():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"tool_calls": [{"function": {"name": "add", "arguments": "{oops"}}]}}]},
        )

    resp = make_provider(handler).chat([{"role": "user", "content": "x"}])
    assert resp.tool_calls[0]["args"] == {"_raw": "{oops"}


def test_retries_once_then_raises():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": "unavailable"})

    p = make_provider(handler)
    with pytest.raises(httpx.HTTPStatusError):
        p.chat([{"role": "user", "content": "x"}])
    assert calls["n"] == 2


def test_retry_succeeds_on_second_attempt():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    resp = make_provider(handler).chat([{"role": "user", "content": "x"}])
    assert resp.content == "ok"
    assert calls["n"] == 2


def test_openrouter_provider_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    p = OpenRouterProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    resp = p.chat([{"role": "user", "content": "hi"}])
    assert seen == {"url": "https://openrouter.ai/api/v1/chat/completions", "auth": "Bearer or-key"}
    assert p.model == "deepseek/deepseek-v4.1-flash-1017"
    assert resp.system == "openrouter"


def test_openrouter_missing_key_names_its_variable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider()


def test_transport_error_retried_once():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.RemoteProtocolError("peer closed connection", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    resp = make_provider(handler).chat([{"role": "user", "content": "hi"}])
    assert resp.content == "ok" and calls["n"] == 2


@pytest.mark.parametrize(
    "cls, level, fields",
    [
        (GeminiProvider, "low", {"reasoning_effort": "low"}),
        (GeminiProvider, "off", {"reasoning_effort": "none"}),
        (OpenRouterProvider, "off", {"reasoning": {"enabled": False}}),
        (OpenRouterProvider, "high", {"reasoning": {"effort": "high"}}),
        (OpenRouterProvider, None, {}),
    ],
)
def test_reasoning_level_maps_to_each_wire_format(cls, level, fields):
    seen = {}

    def handler(request):
        body = json.loads(request.read())
        seen.update({k: body[k] for k in ("reasoning", "reasoning_effort") if k in body})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    p = cls(api_key="k", base_url="http://mock", client=httpx.Client(transport=httpx.MockTransport(handler)), reasoning=level)
    p.chat([{"role": "user", "content": "x"}])
    assert seen == fields
