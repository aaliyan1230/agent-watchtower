"""Analysis: the paper's metrics as pure functions.

All functions take artifact-derived inputs (lists of verdicts and
ground truth) and return numbers. No file I/O, no randomness — that is
what lets the experiments layer checksum artifacts and recompute
results anywhere, reproducibly.

Ground truth conventions:
- A *positive* is a run where a fault was injected (or, for judge
  agreement, a run the reference judge flagged).
- verdicts are mapped to binary: FAIL/FLAGGED = True (flagged),
  PASS = False.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def _binary(verdicts: Iterable[str]) -> list[bool]:
    """PASS -> False; anything else (FLAGGED, FAIL) -> True."""
    return [v != "PASS" for v in verdicts]


def detection_rate(
    verdicts: Sequence[str],
    ground_truth: Sequence[bool],
) -> float:
    """Fraction of injected faults the verification caught:
    TP / (TP + FN). The paper's headline number."""
    if len(verdicts) != len(ground_truth):
        raise ValueError("verdicts and ground truth must align")
    flagged = _binary(verdicts)
    positives = sum(ground_truth)
    if positives == 0:
        return 0.0
    tp = sum(f and gt for f, gt in zip(flagged, ground_truth))
    return tp / positives


def false_positive_rate(verdicts: Sequence[str], ground_truth: Sequence[bool]) -> float:
    """Fraction of clean runs (no fault) wrongly flagged. The paper's
    honesty metric: a verifier that flags everything detects everything."""
    flagged = _binary(verdicts)
    negatives = sum(not gt for gt in ground_truth)
    if negatives == 0:
        return 0.0
    fp = sum(f and not gt for f, gt in zip(flagged, ground_truth))
    return fp / negatives


def per_verifier_detection(
    findings: Sequence[Mapping[str, object]],
    ground_truth: Sequence[bool],
) -> dict[str, float]:
    """Detection rate per verifier name: for each faulted run, did the
    verifier emit at least one finding? `findings` is one list of
    finding dicts per run ([] for clean runs)."""
    if len(findings) != len(ground_truth):
        raise ValueError("findings and ground truth must align")
    names = sorted({f["verifier"] for run in findings for f in run})
    out: dict[str, float] = {}
    positives = sum(ground_truth)
    if positives == 0:
        return {n: 0.0 for n in names}
    for name in names:
        caught = sum(
            gt and any(f["verifier"] == name for f in run)
            for run, gt in zip(findings, ground_truth)
        )
        out[name] = caught / positives
    return out


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Inter-rater agreement corrected for chance (Cohen's kappa),
    on binary labels. The JUDGe-theme metric: judge agreement across
    model variants, and judge vs deterministic consensus."""
    if len(a) != len(b):
        raise ValueError("raters must label the same runs")
    n = len(a)
    if n == 0:
        raise ValueError("kappa needs at least one run")

    po = sum(x == y for x, y in zip(a, b)) / n
    p1 = sum(a) / n
    p2 = sum(b) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)

    if pe == 1.0:
        # Both raters constant: perfect agreement if identical,
        # otherwise undefined — report 0 (worst case) rather than NaN.
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def judge_consensus(judge_votes: Sequence[Sequence[bool]]) -> Sequence[bool]:
    """Majority vote across judge variants — the reference label used
    when measuring a single judge's agreement with the consensus."""
    if not judge_votes:
        return []
    width = len(judge_votes[0])
    out = []
    for i in range(width):
        votes = [r[i] for r in judge_votes]
        out.append(sum(votes) > len(votes) / 2)
    return out
