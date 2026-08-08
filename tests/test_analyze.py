"""Analysis functions: the paper's metrics are just math, so test them."""

import pytest

from experiments.analyze import (
    cohen_kappa,
    detection_rate,
    false_positive_rate,
    judge_consensus,
    per_verifier_detection,
)


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
