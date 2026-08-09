"""GeminiProvider against a mock transport: request shape, response
parsing, error handling — no live network involved."""

import httpx
import pytest

from harness.providers import GeminiProvider, ProviderResponse


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
