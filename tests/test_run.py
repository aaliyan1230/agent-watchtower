"""Phase 2 analysis + runner wiring tests."""

import pytest

from experiments.analyze import (
    fault_coverage,
    judge_vs_deterministic,
    overhead_summary,
    render_table,
)
from experiments.run import FAULT_STEP, NORMAL_SCRIPT, build_workers
from harness.faults import FaultKind
from harness.telemetry import HarnessTelemetry


def cell(fault, verdict="FAIL", judged=False, findings=None, budget=None, seed=1):
    return {
        "fault": fault, "seed": seed, "model": "flash", "run": 1,
        "verdict": verdict, "judged": judged,
        "findings": findings or [], "budget": budget or {"totalTokens": 100, "durationMs": 50},
    }


def test_fault_step_covers_all_kinds():
    assert set(FAULT_STEP) == {f.value for f in FaultKind}


def test_fault_coverage_matrix():
    results = [
        cell("malformed_json", findings=[{"verifier": "schema"}]),
        cell("malformed_json", findings=[{"verifier": "schema"}, {"verifier": "status"}]),
        cell("policy_violation", findings=[{"verifier": "policy"}]),
        cell(None, verdict="PASS"),
    ]
    cov = fault_coverage(results)
    assert cov["malformed_json"]["schema"] == 1.0
    assert cov["malformed_json"]["status"] == 0.5
    assert cov["policy_violation"]["policy"] == 1.0
    assert "clean" not in cov


def test_overhead_summary_groups_by_fault():
    results = [
        cell(None, verdict="PASS", budget={"totalTokens": 75, "durationMs": 50}),
        cell(None, verdict="PASS", budget={"totalTokens": 95, "durationMs": 70}),
        cell("loop", budget={"totalTokens": 300, "durationMs": 120}),
    ]
    over = overhead_summary(results)
    assert over["clean"]["meanTokens"] == pytest.approx(85)
    assert over["loop"]["meanTokens"] == 300


def test_judge_vs_deterministic_kappa():
    results = [
        cell(None, verdict="FLAGGED", judged=True, findings=[{"verifier": "judge"}]),
        cell(None, verdict="PASS", judged=True, findings=[]),
        cell("schema_violation", verdict="FAIL", judged=True, findings=[{"verifier": "schema"}, {"verifier": "judge"}]),
        cell("schema_violation", verdict="FAIL", judged=True, findings=[{"verifier": "schema"}]),
    ]
    kappa, n = judge_vs_deterministic(results)
    assert n == 4
    # det [F,F,T,T] vs judge [T,F,T,F]: po=pe=0.5 -> kappa 0
    assert kappa == pytest.approx(0.0)


def test_judge_vs_deterministic_no_judged_runs():
    kappa, n = judge_vs_deterministic([cell(None, verdict="PASS")])
    assert kappa is None and n == 0


def test_render_table_is_well_formed():
    results = [
        cell("malformed_json", findings=[{"verifier": "schema"}]),
        cell(None, verdict="PASS"),
    ]
    table = render_table(results)
    assert "overall detection" in table
    assert "malformed_json" in table
    assert "schema" in table


def test_build_workers_offline_script_shape():
    # The canonical script: [tool turn, final JSON answer]; the clean
    # run must satisfy the ticket contract and stay within budget.
    from opentelemetry.sdk.trace import TracerProvider

    class FakeCell:
        fault = None
        seed = 1
        model = "flash"
        run = 1

    telemetry = HarnessTelemetry("http://127.0.0.1:9")
    try:
        workers = build_workers(telemetry, FakeCell(), live=False)
        assert len(workers) == 1
        assert len(NORMAL_SCRIPT) == 2
        assert NORMAL_SCRIPT[1].content.startswith('{"id": 1')
        answer = workers[0].run("triage")
        assert '"id": 1' in answer
    finally:
        telemetry.shutdown()


def test_judge_agreement_pairs_runs():
    from experiments.analyze import judge_agreement

    a = [
        cell(None, verdict="PASS", judged=True, findings=[]),
        cell("loop", seed=1, judged=True, findings=[{"verifier": "judge"}]),
        cell("loop", seed=2, judged=True, findings=[{"verifier": "judge"}, {"verifier": "loop"}]),
    ]
    b = [
        cell(None, verdict="PASS", judged=True, findings=[]),
        cell("loop", seed=1, judged=True, findings=[]),
        cell("loop", seed=2, judged=True, findings=[{"verifier": "judge"}, {"verifier": "loop"}]),
    ]
    kappa, n = judge_agreement(a, b)
    assert n == 3
    assert kappa is not None
    # a flags [F,T,T], b flags [F,F,T] -> po=2/3, p1=2/3, p2=1/3, pe=2/3*1/3+1/3*2/3=4/9
    assert kappa == pytest.approx((2/3 - 4/9) / (1 - 4/9))
