"""Analysis functions: the paper's metrics are just math, so test them."""

import pytest

from experiments.analyze import (
    behavior_by_telemetry_matrix,
    cohen_kappa,
    condition_of,
    condition_summary,
    detection_rate,
    evidence_gap_preservation,
    false_assurance_rate_2x2,
    false_positive_rate,
    fault_recall_late,
    flip_rate,
    judge_consensus,
    per_verifier_detection,
    premature_pass_rate,
    render_condition_comparison,
    safe_abstention_rate,
    verdict_calibration,
)


def _fa_cell(behavior, telemetry, verdict):
    return {"fault": behavior, "evidence_fault": telemetry, "verdict": verdict}


def test_condition_of_maps_four_cells():
    assert condition_of(_fa_cell(None, None, "PASS")) == (
        "clean_behavior",
        "clean_telemetry",
    )
    assert condition_of(_fa_cell("loop", None, "FAIL")) == (
        "faulty_behavior",
        "clean_telemetry",
    )
    assert condition_of(_fa_cell(None, "drop_tool_result", "INCONCLUSIVE")) == (
        "clean_behavior",
        "faulty_telemetry",
    )
    assert condition_of(_fa_cell("loop", "drop_tool_result", "INCONCLUSIVE")) == (
        "faulty_behavior",
        "faulty_telemetry",
    )
    # reordering is a semantics-preserving control, not a fault
    assert condition_of(_fa_cell(None, "reorder_spans", "PASS")) == (
        "clean_behavior",
        "clean_telemetry",
    )


def test_false_assurance_rates():
    cells = [
        _fa_cell("loop", "drop_tool_result", "INCONCLUSIVE"),
        _fa_cell("loop", "drop_tool_result", "PASS"),  # the dangerous cell
        _fa_cell(None, "drop_tool_result", "INCONCLUSIVE"),
        _fa_cell(None, None, "PASS"),
    ]
    assert false_assurance_rate_2x2(cells) == pytest.approx(0.5)
    assert safe_abstention_rate(cells) == pytest.approx(2 / 3)


def test_condition_summary_rates():
    cells = [
        _fa_cell(None, None, "PASS"),
        _fa_cell(None, None, "PASS"),
        _fa_cell("loop", None, "FAIL"),
        _fa_cell(None, "drop_tool_result", "INCONCLUSIVE"),
    ]
    summary = condition_summary(cells)
    assert summary["clean_behavior x clean_telemetry"]["passRate"] == pytest.approx(1.0)
    assert summary["faulty_behavior x clean_telemetry"]["failRate"] == pytest.approx(
        1.0
    )
    assert summary["clean_behavior x faulty_telemetry"][
        "inconclusiveRate"
    ] == pytest.approx(1.0)


def test_behavior_by_telemetry_matrix_pass_rates():
    cells = [
        _fa_cell("loop", "drop_tool_result", "INCONCLUSIVE"),
        _fa_cell("loop", "drop_tool_result", "PASS"),
        _fa_cell("loop", None, "FAIL"),
    ]
    matrix = behavior_by_telemetry_matrix(cells)
    assert matrix["loop"]["drop_tool_result"] == pytest.approx(0.5)
    assert matrix["loop"]["clean"] == pytest.approx(0.0)


def test_condition_comparison_render():
    contract = [_fa_cell("loop", "drop_tool_result", "INCONCLUSIVE")]
    baseline = [_fa_cell("loop", "drop_tool_result", "PASS")]
    out = render_condition_comparison(contract, baseline)
    assert "0% / 100%" in out
    assert "100% / 0%" in out
    assert "false assurance: contract 0%   baseline 100%" in out


def test_evidence_gap_preservation_under_judge():
    cells = [
        {
            "fault": None,
            "evidence_fault": "drop_tool_result",
            "verdict": "INCONCLUSIVE",
            "judged": True,
            "findings": [
                {"verifier": "evidence", "kind": "evidence_gap", "severity": 0},
                {"verifier": "judge", "severity": 1},
            ],
        },
        {
            # deterministic critical + gap: FAIL is correct, not an override
            "fault": "policy_violation",
            "evidence_fault": "truncate_final",
            "verdict": "FAIL",
            "judged": True,
            "findings": [
                {"verifier": "policy", "severity": 1},
                {"verifier": "evidence", "kind": "evidence_gap", "severity": 0},
            ],
        },
        {
            "fault": "policy_violation",
            "evidence_fault": None,
            "verdict": "FAIL",
            "judged": True,
            "findings": [{"verifier": "policy", "severity": 1}],
        },
    ]
    preserved, n = evidence_gap_preservation(cells)
    assert n == 1  # only the judge-free-gap cell qualifies
    assert preserved == pytest.approx(1.0)
    assert evidence_gap_preservation(cells[1:]) == (None, 0)


def test_detection_rate():
    verdicts = ["FAIL", "PASS", "FAIL", "FLAGGED", "PASS"]
    truth = [True, True, True, False, False]
    assert detection_rate(verdicts, truth) == pytest.approx(2 / 3)
    assert false_positive_rate(verdicts, truth) == pytest.approx(1 / 2)


def test_detection_rate_edge_cases():
    assert detection_rate(["PASS"], [False]) == 0.0
    assert detection_rate(["FAIL"], [False]) == 0.0  # no positives -> 0
    assert false_positive_rate(["FAIL"], [True]) == 0.0  # no negatives -> 0


def test_per_verifier_detection():
    findings = [
        [{"verifier": "schema"}, {"verifier": "budget"}],
        [{"verifier": "schema"}],
        [],
        [{"verifier": "loop"}],
    ]
    truth = [True, True, True, False]
    out = per_verifier_detection(findings, truth)
    assert out["schema"] == pytest.approx(2 / 3)
    assert out["loop"] == pytest.approx(0.0)
    assert out["budget"] == pytest.approx(1 / 3)


def test_kappa_perfect_agreement():
    labels = [True, False, True, False, True]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_chance_level_agreement():
    # Uncorrelated raters agree exactly as often as chance -> kappa 0.
    a = [True, False, True, False, True, False, True, False]
    b = [True, True, False, False, True, True, False, False]
    assert cohen_kappa(a, b) == pytest.approx(0.0)


def test_kappa_negative_for_worse_than_chance():
    # Both raters constant but opposite: observed agreement is 0,
    # below the chance baseline -> negative kappa.
    a = [True, True, False]
    b = [False, False, True]
    assert cohen_kappa(a, b) < 0


def test_kappa_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa([True], [])


def test_judge_consensus():
    votes = [[True, False, True], [False, True, True], [True, True, False]]
    assert judge_consensus(votes) == [True, True, True]


def _causal_cell(serials, closed=True, fault=None):
    return {
        "checked": True,
        "serializationVerdicts": list(serials),
        "closed": closed,
        "evidenceFault": fault,
        "budget": {"totalTokens": 100, "durationMs": 50},
    }


def test_flip_rate():
    cells = [
        _causal_cell(["PASS", "PASS", "PASS"]),
        _causal_cell(["PASS", "FAIL"]),  # a flip
        _causal_cell(["INCONCLUSIVE", "INCONCLUSIVE"]),
    ]
    rate, n = flip_rate(cells)
    assert n == 3
    assert rate == pytest.approx(1 / 3)
    # No multi-serialization traces -> None
    assert flip_rate([_causal_cell(["PASS"])]) == (None, 0)


def test_premature_pass_rate():
    cells = [
        _causal_cell(["PASS"], closed=False),  # premature PASS
        _causal_cell(["INCONCLUSIVE"], closed=False),
        _causal_cell(["PASS"], closed=True),
    ]
    rate, n = premature_pass_rate(cells)
    assert n == 2
    assert rate == pytest.approx(0.5)
    assert premature_pass_rate([_causal_cell(["PASS"], closed=True)]) == (None, 0)


def test_fault_recall_late():
    cells = [
        _causal_cell(["INCONCLUSIVE"], fault="orphan_link_target"),  # caught
        _causal_cell(["PASS"], fault="orphan_link_target"),  # missed
        _causal_cell(["PASS"], fault="truncate_closed"),  # missed
        _causal_cell(["PASS"], fault=None),
    ]
    recall = fault_recall_late(cells)
    assert recall["orphan_link_target"] == pytest.approx(0.5)
    assert recall["truncate_closed"] == pytest.approx(0.0)
    assert fault_recall_late([_causal_cell(["PASS"])]) == {}


def test_verdict_calibration():
    cells = [
        _causal_cell(["PASS", "PASS"], fault="truncate_closed"),
        _causal_cell(["INCONCLUSIVE"], fault="truncate_closed"),
    ]
    cal = verdict_calibration(cells)
    key = "closed x truncate_closed"
    assert key in cal
    assert cal[key]["passRate"] == pytest.approx(0.5)
    assert cal[key]["inconclusiveRate"] == pytest.approx(0.5)
