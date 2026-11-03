"""CausalTrace live judge: verdict parsing, the driver loop, and the
provider wiring — all offline via mocked HTTP transports."""

import json

import httpx
import pytest

from experiments import causal_judge as cj
from experiments import causal_run as cr
from experiments.causal_judge import (
    FakeCausalJudge,
    make_judge,
    parse_verdict,
    run_judge,
)


def _cells(n=4):
    cells = []
    for idx in range(1, n + 1):
        cells += cr.build_cells(f"causal-{idx}-flash", seed=idx, variant="flash")
    return cells


def test_parse_verdict_tokens():
    assert parse_verdict('{"verdict": "PASS"}') == "PASS"
    assert parse_verdict("The log is INCONCLUSIVE: incomplete.") == "INCONCLUSIVE"
    assert parse_verdict('"FAIL": true') == "FAIL"
    assert parse_verdict("uncertain") == ""
    assert parse_verdict("") == ""


def test_fake_judge_records_calls():
    judge = FakeCausalJudge(
        script=[("PASS", {"inputTokens": 5, "outputTokens": 1, "durationMs": 4.0})]
    )
    verdict, usage = judge.judge("render")
    assert verdict == "PASS"
    assert usage["inputTokens"] == 5
    assert len(judge.calls) == 1
    # Exhausted script falls back to a neutral abstention.
    assert judge.judge("render")[0] == "INCONCLUSIVE"


def test_run_judge_three_renderings_per_cell():
    judge = FakeCausalJudge(
        script=[("PASS", {"inputTokens": 5, "outputTokens": 1, "durationMs": 4.0})] * 30
    )
    out = run_judge(_cells(3), judge)
    assert len(out) == 12  # 3 executions x 4 conditions
    assert all(
        set(row["renderings"]) == {"text", "time_sorted", "canonical"} for row in out
    )
    assert all(row["evidenceFault"] is not None or row["traceId"] for row in out)


def test_run_judge_does_not_touch_cells_without_renderings():
    judge = FakeCausalJudge()
    out = run_judge([{"traceId": "old-grid-cell"}], judge)
    assert out == [
        {
            "traceId": "old-grid-cell",
            "evidenceFault": None,
            "checksum": "",
            "renderings": {},
            "usage": {},
        }
    ]


def test_run_judge_limit():
    judge = FakeCausalJudge(
        script=[("PASS", {"inputTokens": 1, "outputTokens": 1, "durationMs": 1.0})] * 30
    )
    out = run_judge(_cells(4), judge, limit=4)
    assert len(out) == 4  # first 4 cells only



class _FlakyJudge:
    """Fails the second call with a transport error, answers PASS otherwise."""

    def __init__(self):
        self.calls = 0

    def judge(self, render):
        self.calls += 1
        if self.calls == 2:
            raise httpx.ConnectError("connection reset by peer")
        return "PASS", {"inputTokens": 1, "outputTokens": 1, "durationMs": 1.0}


def test_run_judge_records_outage_without_aborting():
    judge = _FlakyJudge()
    out = run_judge(_cells(1), judge, limit=2)
    assert len(out) == 2
    assert set(out[0]["renderings"]) == {"text", "canonical"}
    assert out[0]["errors"]["time_sorted"].startswith("ConnectError")
    assert "errors" not in out[1]
    assert set(out[1]["renderings"]) == {"text", "time_sorted", "canonical"}

def _completion(content):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        },
    )


@pytest.mark.parametrize(
    "family, key_env, reasoning",
    [
        ("gemini", "GEMINI_API_KEY", {"reasoning_effort": "low"}),
        ("deepseek", "OPENROUTER_API_KEY", {"reasoning": {"enabled": False}}),
        ("kimi", "OPENROUTER_API_KEY", {"reasoning": {"enabled": False}}),
    ],
)
def test_make_judge_family_route_and_reasoning(monkeypatch, family, key_env, reasoning):
    monkeypatch.setenv(key_env, "test-key")
    seen = {}

    def handler(request):
        body = json.loads(request.read())
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["model"] = body["model"]
        seen["reasoning"] = {k: body[k] for k in ("reasoning", "reasoning_effort") if k in body}
        return _completion('{"verdict": "INCONCLUSIVE"}')

    http = httpx.Client(transport=httpx.MockTransport(handler))
    judge = make_judge(family, base_url="http://mock", http=http)
    verdict, usage = judge.judge("a rendering")
    model = cj.FAMILIES[family][1]
    assert judge.name == family and judge.model == model
    assert seen == {
        "url": "http://mock/chat/completions",
        "auth": "Bearer test-key",
        "model": model,
        "reasoning": reasoning,
    }
    assert verdict == "INCONCLUSIVE"
    assert usage["inputTokens"] == 7
    assert usage["outputTokens"] == 2


def test_make_judge_model_override(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    http = httpx.Client(transport=httpx.MockTransport(lambda r: _completion("PASS")))
    assert make_judge("gemini", model="test-model", base_url="http://mock", http=http).model == "test-model"


def test_make_judge_missing_key_fails(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        make_judge("deepseek", base_url="http://mock", http=httpx.Client())


def test_make_judge_unknown_family():
    with pytest.raises(ValueError, match="unknown judge"):
        make_judge("claude")
