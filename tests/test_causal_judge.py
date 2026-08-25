"""CausalTrace live judge: verdict parsing, the driver loop, and the
provider wiring — all offline via mocked HTTP transports."""

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


def _gemini_completion(content):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        },
    )


def test_make_judge_gemini_mock_transport():
    def handler(request):
        assert request.headers["authorization"].startswith("Bearer ")
        return _gemini_completion('{"verdict": "INCONCLUSIVE"}')

    http = httpx.Client(transport=httpx.MockTransport(handler))
    judge = make_judge("gemini", model="test-model", base_url="http://mock", http=http)
    assert judge.model == "test-model"
    verdict, usage = judge.judge("a rendering")
    assert verdict == "INCONCLUSIVE"
    assert usage["inputTokens"] == 7
    assert usage["outputTokens"] == 2


def test_make_judge_gemini_missing_key_fails(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        make_judge("gemini", model="x", base_url="http://mock", http=httpx.Client())


def test_make_judge_kimi_mock_transport(monkeypatch):
    monkeypatch.setattr(
        cj,
        "aws_env",
        lambda: {
            "AWS_ACCESS_KEY_ID": "AK",
            "AWS_SECRET_ACCESS_KEY": "SK",
            "AWS_REGION": "us-east-1",
        },
    )
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "PASS"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            },
        )

    judge = make_judge(
        "kimi",
        base_url="https://mock.example.com",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert judge.name == "kimi"
    assert judge.model == "moonshotai.kimi-k2.5"
    verdict, usage = judge.judge("rendering")
    assert verdict == "PASS"
    assert usage["inputTokens"] == 9
    assert usage["durationMs"] >= 0
    assert seen["url"].startswith("https://mock.example.com/model/")
    assert seen["url"].endswith("/invoke")


def test_make_judge_bedrock_missing_creds(monkeypatch):
    monkeypatch.setattr(cj, "aws_env", lambda: {})
    with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
        make_judge("bedrock", http=httpx.Client())


def test_make_judge_unknown_family():
    with pytest.raises(ValueError, match="unknown judge"):
        make_judge("claude")
