"""The end-to-end decision replay: cost accounting and client-side
handling of server decisions, without a server."""

import pytest

from experiments import decide_replay as dr

COSTS = {"miss": 100, "falseHalt": 10, "human": 5, "replay": 1, "judge": 0.5}


@pytest.mark.parametrize(
    "action, faulty, want",
    [("accept", True, 100), ("accept", False, 0), ("halt", False, 10), ("halt", True, 0), ("escalate", True, 5)],
)
def test_outcome_cost(action, faulty, want):
    assert dr.outcome_cost(action, faulty, COSTS) == want


def test_outcome_cost_rejects_probes():
    with pytest.raises(ValueError):
        dr.outcome_cost("replay", True, COSTS)


def test_follow_replays_then_falls_back(monkeypatch):
    responses = iter([
        {"verdict": "INCONCLUSIVE", "decision": {"first": "replay", "action": "replay", "fallback": "halt"}},
        {"verdict": "INCONCLUSIVE", "decision": {"first": "judge", "action": "replay", "fallback": "escalate"}},
    ])
    posted = []
    monkeypatch.setattr(dr, "post_envelope", lambda endpoint, env: posted.append(env) or next(responses))
    row = dr.follow({"clean": True}, {"clean": False}, "http://x")
    assert posted == [{"clean": False}, {"clean": True}]
    assert row == {"first": "replay", "final": "escalate", "probes": ["replay", "judge"], "verdict": "INCONCLUSIVE"}


def test_summarize_charges_probes_and_outcomes():
    rows = [
        {"final": "accept", "faulty": False, "probes": ["judge"]},
        {"final": "halt", "faulty": True, "probes": ["replay"]},
        {"final": "accept", "faulty": True, "probes": []},
        {"final": "escalate", "faulty": False, "probes": []},
    ]
    s = dr.summarize(rows, COSTS)
    assert s["costPerRun"] == pytest.approx((0.5 + 1 + 100 + 5) / 4)
    assert s["acceptedFaulty"] == 0.25
    assert s["judgeCallsPer100"] == 25 and s["replaysPer100"] == 25
