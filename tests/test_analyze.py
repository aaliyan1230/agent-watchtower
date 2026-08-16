"""Analysis functions: the paper's metrics are just math, so test them."""

import pytest

from experiments.analyze import (
    behavior_by_telemetry_matrix,
    cohen_kappa,
    condition_of,
    condition_summary,
    detection_rate,
    false_assurance_rate_2x2,
    false_positive_rate,
    judge_consensus,
    per_verifier_detection,
    render_condition_comparison,
    safe_abstention_rate,
)


def _fa_cell(behavior, telemetry, verdict):
    return {"fault": behavior, "evidence_fault": telemetry, "verdict": verdict}


def test_condition_of_maps_four_cells():
    assert condition_of(_fa_cell(None, None, "PASS")) == ("clean_behavior", "clean_telemetry")
    assert condition_of(_fa_cell("loop", None, "FAIL")) == ("faulty_behavior", "clean_telemetry")
    assert condition_of(_fa_cell(None, "drop_tool_result", "INCONCLUSIVE")) == ("clean_behavior", "faulty_telemetry")
    assert condition_of(_fa_cell("loop", "drop_tool_result", "INCONCLUSIVE")) == ("faulty_behavior", "faulty_telemetry")
    # reordering is a semantics-preserving control, not a fault
    assert condition_of(_fa_cell(None, "reorder_spans", "PASS")) == ("clean_behavior", "clean_telemetry")


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
    assert summary["faulty_behavior x clean_telemetry"]["failRate"] == pytest.approx(1.0)
    assert summary["clean_behavior x faulty_telemetry"]["inconclusiveRate"] == pytest.approx(1.0)


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
