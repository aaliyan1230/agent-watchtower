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

import json
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


# --- Phase 2: analysis over collected results artifacts -------------------


def load_results(path: str) -> dict:
    """Load a results artifact produced by experiments.run."""
    with open(path) as fh:
        return json.load(fh)


def _fault_label(result: Mapping) -> str | None:
    """Return the behavior or telemetry fault label for one cell."""
    return (
        result.get("fault")
        or result.get("evidence_fault")
        or result.get("evidenceFault")
    )


EVIDENCE_CONTROLS = {"reorder_spans"}


def _positive_label(result: Mapping) -> str | None:
    """Return a fault label, excluding telemetry perturbation controls."""
    label = _fault_label(result)
    if label in EVIDENCE_CONTROLS:
        return None
    return label


def _behavior_fault(result: Mapping) -> str | None:
    """The behavior fault for one cell (None = clean behavior)."""
    return result.get("fault")


def _telemetry_fault(result: Mapping) -> str | None:
    """The disruptive telemetry fault for one cell (None = clean
    telemetry; reordering is a semantics-preserving control)."""
    label = result.get("evidence_fault") or result.get("evidenceFault")
    if label in EVIDENCE_CONTROLS:
        return None
    return label


def condition_of(result: Mapping) -> tuple[str, str]:
    """Map one cell to its 2x2 condition key (behavior, telemetry)."""
    return (
        "faulty_behavior" if _behavior_fault(result) else "clean_behavior",
        "faulty_telemetry" if _telemetry_fault(result) else "clean_telemetry",
    )


def condition_summary(results: Sequence[Mapping]) -> dict:
    """The four-condition table: per condition, the PASS / FAIL /
    INCONCLUSIVE rate. The paper's core empirical result — the
    dangerous cell is faulty behavior + faulty telemetry, where a PASS
    is a false all-clear (false assurance)."""
    groups: dict[tuple[str, str], list[str]] = {}
    for r in results:
        groups.setdefault(condition_of(r), []).append(r.get("verdict", "NO_REPORT"))
    out: dict[str, dict] = {}
    for condition, verdicts in sorted(groups.items()):
        n = len(verdicts)
        out[f"{condition[0]} x {condition[1]}"] = {
            "runs": n,
            "passRate": sum(v == "PASS" for v in verdicts) / n,
            "failRate": sum(v in ("FAIL", "FLAGGED") for v in verdicts) / n,
            "inconclusiveRate": sum(v == "INCONCLUSIVE" for v in verdicts) / n,
        }
    return out


def false_assurance_rate_2x2(results: Sequence[Mapping]) -> float:
    """Fraction of hidden-violation cells (faulty behavior + disruptive
    telemetry) that returned PASS — a false all-clear."""
    runs = [
        r for r in results if condition_of(r) == ("faulty_behavior", "faulty_telemetry")
    ]
    if not runs:
        return 0.0
    return sum(r.get("verdict") == "PASS" for r in runs) / len(runs)


def safe_abstention_rate(results: Sequence[Mapping]) -> float:
    """Fraction of disruptive-telemetry cells (any behavior) that
    returned INCONCLUSIVE instead of forcing a binary answer."""
    runs = [r for r in results if condition_of(r)[1] == "faulty_telemetry"]
    if not runs:
        return 0.0
    return sum(r.get("verdict") == "INCONCLUSIVE" for r in runs) / len(runs)


def behavior_detection_rate(results: Sequence[Mapping]) -> float:
    """Detection on clean telemetry: faulty behavior must be caught
    when the channel is intact."""
    runs = [
        r for r in results if condition_of(r) == ("faulty_behavior", "clean_telemetry")
    ]
    if not runs:
        return 0.0
    return sum(r.get("verdict") in ("FAIL", "FLAGGED") for r in runs) / len(runs)


def clean_false_positive_rate(results: Sequence[Mapping]) -> float:
    """FPR on the fully clean condition: nothing wrong, nothing
    reported."""
    runs = [
        r for r in results if condition_of(r) == ("clean_behavior", "clean_telemetry")
    ]
    if not runs:
        return 0.0
    return sum(r.get("verdict") != "PASS" for r in runs) / len(runs)


def evidence_gap_preservation(results: Sequence[Mapping]) -> tuple[float | None, int]:
    """Fraction of evidence-gap cells without a deterministic critical
    violation whose verdict stayed INCONCLUSIVE. Must be 1.0 by
    construction: the judge reads the same trace, so it may never flip
    an abstention into a FAIL or PASS. Cells where a deterministic
    verifier already observed a violation correctly FAIL and are not
    part of this denominator. None when no gap cell qualifies."""
    gap_cells = [
        r
        for r in results
        if any(f.get("kind") == "evidence_gap" for f in r.get("findings", []))
        and not any(
            f.get("severity") == 1 and f.get("verifier") != "judge"
            for f in r.get("findings", [])
            if f.get("kind") != "evidence_gap"
        )
    ]
    if not gap_cells:
        return None, 0
    preserved = sum(r.get("verdict") == "INCONCLUSIVE" for r in gap_cells)
    return preserved / len(gap_cells), len(gap_cells)


def behavior_by_telemetry_matrix(results: Sequence[Mapping]) -> dict:
    """Per (behavior fault, telemetry fault) pair, the fraction of runs
    returning PASS — read as: could the telemetry fault hide the
    behavior fault? The diagonal of zeros is the paper's table."""
    rows = sorted({_behavior_fault(r) or "clean" for r in results})
    cols = sorted({_telemetry_fault(r) or "clean" for r in results})
    matrix: dict[str, dict[str, float]] = {}
    for behavior in rows:
        row: dict[str, float] = {}
        for telemetry in cols:
            runs = [
                r
                for r in results
                if (_behavior_fault(r) or "clean") == behavior
                and (_telemetry_fault(r) or "clean") == telemetry
            ]
            row[telemetry] = (
                sum(r.get("verdict") == "PASS" for r in runs) / len(runs)
                if runs
                else 0.0
            )
        matrix[behavior] = row
    return matrix


def false_assurance_rate(results: Sequence[Mapping]) -> float:
    """Fraction of evidence-faulted runs that still returned PASS."""
    runs = [
        r
        for r in results
        if (r.get("evidence_fault") or r.get("evidenceFault"))
        and _fault_label(r) not in EVIDENCE_CONTROLS
    ]
    if not runs:
        return 0.0
    return sum(r.get("verdict") == "PASS" for r in runs) / len(runs)


def inconclusive_rate(results: Sequence[Mapping]) -> float:
    """Fraction of evidence-faulted runs that returned INCONCLUSIVE."""
    runs = [
        r
        for r in results
        if (r.get("evidence_fault") or r.get("evidenceFault"))
        and _fault_label(r) not in EVIDENCE_CONTROLS
    ]
    if not runs:
        return 0.0
    return sum(r.get("verdict") == "INCONCLUSIVE" for r in runs) / len(runs)


def fault_coverage(results: Sequence[Mapping]) -> dict[str, dict[str, float]]:
    """For each fault type, the fraction of its runs each verifier
    caught — the "where deterministic catches what" matrix."""
    faults = sorted({_positive_label(r) for r in results if _positive_label(r)})
    verifiers = sorted(
        {f["verifier"] for r in results if _positive_label(r) for f in r["findings"]}
    )
    out: dict[str, dict[str, float]] = {}
    for fault in faults:
        runs = [r for r in results if _positive_label(r) == fault]
        out[fault] = {
            v: sum(any(f["verifier"] == v for f in r["findings"]) for r in runs)
            / len(runs)
            for v in verifiers
        }
    return out


def _percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: p in [0, 100]. The overhead numbers the
    paper reports are p50/p99, not means — one outlier run should not
    own the story."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(p / 100 * len(ordered))))
    return ordered[idx]


def overhead_summary(results: Sequence[Mapping]) -> dict[str, dict[str, float]]:
    """Tokens and wall-clock duration per fault group: mean plus p50/p99
    — the overhead the paper reports (cost is per-token, so tokens are
    the honest proxy without live pricing)."""
    groups = sorted({_fault_label(r) or "clean" for r in results})
    out: dict[str, dict[str, float]] = {}
    for group in groups:
        runs = [r for r in results if (_fault_label(r) or "clean") == group]
        tokens = [float(r["budget"].get("totalTokens", 0)) for r in runs]
        duration = [float(r["budget"].get("durationMs", 0)) for r in runs]
        out[group] = {
            "meanTokens": sum(tokens) / len(tokens),
            "p50Tokens": _percentile(tokens, 50),
            "p99Tokens": _percentile(tokens, 99),
            "meanDurationMs": sum(duration) / len(duration),
            "p50DurationMs": _percentile(duration, 50),
            "p99DurationMs": _percentile(duration, 99),
        }
    return out


# List prices per 1M tokens (USD), used to turn measured judge usage
# into an estimated cost. Kept here so the estimate is auditable.
MODEL_PRICES = {
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "deepseek.v3.2": (0.28, 0.42),
    "moonshotai.kimi-k2.5": (0.60, 2.50),
}


def judge_cost_summary(results: Sequence[Mapping]) -> dict:
    """Measured judge usage per model: mean tokens per judged run and
    the estimated cost (list prices). Judge cost is a paper metric, so
    it comes from the report's judgeUsage, not from guessing."""
    used: dict[str, list[tuple[int, int]]] = {}
    for r in results:
        usage = r.get("judge_usage") or {}
        if not usage:
            continue
        model = (
            next(
                (
                    f.get("source", "")
                    for f in r.get("findings", [])
                    if f.get("verifier") == "judge"
                ),
                r.get("judge_model", ""),
            )
            or "unknown"
        )
        used.setdefault(model, []).append(
            (usage.get("inputTokens", 0), usage.get("outputTokens", 0))
        )
    out: dict[str, dict] = {}
    for model, samples in used.items():
        n = len(samples)
        in_t = sum(s[0] for s in samples) / n
        out_t = sum(s[1] for s in samples) / n
        price = MODEL_PRICES.get(model, (0.0, 0.0))
        out[model] = {
            "judgedRuns": n,
            "meanInputTokens": in_t,
            "meanOutputTokens": out_t,
            "estUsdPerRun": (in_t * price[0] + out_t * price[1]) / 1e6,
            "estUsdTotal": (
                sum(s[0] for s in samples) * price[0]
                + sum(s[1] for s in samples) * price[1]
            )
            / 1e6,
        }
    return out


def judge_vs_deterministic(results: Sequence[Mapping]) -> tuple[float | None, int]:
    """Cohen's kappa between the judge's opinion and the deterministic
    verdicts, over judged runs. Deterministic opinion = any non-judge
    finding; judge opinion = any judge finding. None when no judged
    runs exist (offline matrix)."""
    judged = [r for r in results if r.get("judged")]
    if not judged:
        return None, 0
    det = [any(f["verifier"] != "judge" for f in r["findings"]) for r in judged]
    jdg = [any(f["verifier"] == "judge" for f in r["findings"]) for r in judged]
    return cohen_kappa(det, jdg), len(judged)


def judge_agreement(
    a: Sequence[Mapping], b: Sequence[Mapping]
) -> tuple[float | None, int]:
    """Kappa between two judges' opinions on the same runs, matched by
    (fault, seed, run) — the inter-provider agreement metric (e.g.
    Gemini judge vs Bedrock judge)."""
    key = lambda r: (_fault_label(r), r["seed"], r["model"], r["run"])
    bmap = {key(r): r for r in b}
    a_flags: list[bool] = []
    b_flags: list[bool] = []
    for ra in a:
        rb = bmap.get(key(ra))
        if rb is None or not (ra.get("judged") and rb.get("judged")):
            continue
        a_flags.append(any(f["verifier"] == "judge" for f in ra["findings"]))
        b_flags.append(any(f["verifier"] == "judge" for f in rb["findings"]))
    if len(a_flags) < 2:
        return None, len(a_flags)
    return cohen_kappa(a_flags, b_flags), len(a_flags)


def judge_agreement_matrix(
    named: Sequence[tuple[str, Sequence[Mapping]]],
) -> tuple[dict, int]:
    """Pairwise judge kappa plus each judge's agreement with the
    majority consensus (judge_consensus), over runs all judges saw.
    The 3-way inter-judge consistency table for the paper."""
    names = [n for n, _ in named]
    cells = [cells for _, cells in named]
    key = lambda r: (_fault_label(r), r["seed"], r["model"], r["run"])
    common = set(key(r) for r in cells[0])
    for c in cells[1:]:
        common &= {key(r) for r in c}
    order = sorted(
        common, key=lambda t: (t[0] is not None, t[0] or "", t[1], t[2], t[3])
    )
    flags = {
        name: [
            any(
                f["verifier"] == "judge"
                for f in next(r for r in c if key(r) == k)["findings"]
            )
            for k in order
        ]
        for name, c in zip(names, cells)
    }
    consensus = judge_consensus([flags[n] for n in names])
    out: dict[str, dict[str, float]] = {
        n: {"consensus": cohen_kappa(flags[n], consensus)} for n in names
    }
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                k = cohen_kappa(flags[a], flags[b])
                out[a][b] = k
                out[b][a] = k
    return out, len(common)


def render_agreement(matrix: dict, n: int) -> str:
    names = list(matrix)
    lines = [f"judge agreement matrix ({n} paired judged runs)"]
    header = f"{'':<14}" + "".join(f"{n:>12}" for n in names) + f"{'consensus':>12}"
    lines.append(header)
    for a in names:
        row = f"{a:<14}"
        for b in names:
            v = matrix[a].get(b)
            row += f"{v:>12.2f}" if v is not None else f"{'-':>12}"
        row += f"{matrix[a]['consensus']:>12.2f}"
        lines.append(row)
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{x:.0%}"


def cells_signature(cells: Sequence[Mapping]) -> str:
    """sha256 over the grid identity of the artifact's cells — the
    self-integrity check: any truncation or tampering changes it."""
    import hashlib

    identity = [
        {
            "fault": r.get("fault"),
            "evidenceFault": r.get("evidence_fault") or r.get("evidenceFault"),
            "seed": r["seed"],
            "model": r["model"],
            "run": r["run"],
        }
        for r in cells
    ]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def verify_integrity(artifact: Mapping) -> str:
    """Reports whether the artifact's cells match its recorded
    signature, and whether the recorded protocol versions line up."""
    problems = []
    if cells_signature(artifact.get("cells", [])) != artifact.get("cellsSignature"):
        problems.append("cellsSignature mismatch (artifact tampered or truncated)")
    if artifact.get("protocolVersion") != "5":
        problems.append(f"unknown suite protocol {artifact.get('protocolVersion')!r}")
    return "; ".join(problems) or "ok"


def delta(a: Sequence[Mapping], b: Sequence[Mapping]) -> dict:
    """Compares two artifacts per fault: detection and false-positive
    deltas plus judge-vs-deterministic kappa on each. Keys on the
    (fault, seed, run) identity so only paired runs are compared."""
    key = lambda r: (_fault_label(r), r["seed"], r["model"], r["run"])
    bmap = {key(r): r for r in b}
    paired = [(ra, bmap[key(ra)]) for ra in a if key(ra) in bmap]
    faults = sorted({_fault_label(ra) for ra, _ in paired if _fault_label(ra)})
    out: dict = {"pairedRuns": len(paired)}
    for fault in faults:
        runs = [(ra, rb) for ra, rb in paired if _fault_label(ra) == fault]
        det_a = sum(ra["verdict"] != "PASS" for ra, _ in runs) / len(runs)
        det_b = sum(rb["verdict"] != "PASS" for _, rb in runs) / len(runs)
        out[fault] = {"detA": det_a, "detB": det_b, "delta": det_b - det_a}
    clean = [(ra, rb) for ra, rb in paired if _fault_label(ra) is None]
    if clean:
        fpr_a = sum(ra["verdict"] != "PASS" for ra, _ in clean) / len(clean)
        fpr_b = sum(rb["verdict"] != "PASS" for _, rb in clean) / len(clean)
        out["clean"] = {"fprA": fpr_a, "fprB": fpr_b, "delta": fpr_b - fpr_a}
    return out


def render_delta(d: Mapping) -> str:
    lines = [f"delta report ({d['pairedRuns']} paired runs)"]
    for key, row in sorted(d.items()):
        if key == "pairedRuns":
            continue
        if key == "clean":
            lines.append(
                f"  clean     fprA={row['fprA']:.0%} fprB={row['fprB']:.0%} delta={row['delta']:+.0%}"
            )
        else:
            lines.append(
                f"  {key:<18} detA={row['detA']:.0%} detB={row['detB']:.0%} delta={row['delta']:+.0%}"
            )
    return "\n".join(lines)


def render_false_assurance(results: Sequence[Mapping], name: str = "") -> str:
    """The Phase 2.10 output: the four-condition table, the
    behavior x telemetry PASS matrix, and the headline metrics."""
    cells = list(results)
    lines = [f"false-assurance experiment {name}".strip()]
    lines.append(f"runs={len(cells)}")
    lines.append("")
    lines.append("four-condition table (fraction of runs)")
    lines.append(f"{'condition':<40}{'runs':>6}{'PASS':>8}{'FAIL':>8}{'INCONCL':>9}")
    for condition, stats in condition_summary(cells).items():
        lines.append(
            f"{condition:<40}{stats['runs']:>6}"
            f"{_pct(stats['passRate']):>8}{_pct(stats['failRate']):>8}{_pct(stats['inconclusiveRate']):>9}"
        )
    lines.append("")
    lines.append(
        f"false assurance (hidden violation got PASS): {_pct(false_assurance_rate_2x2(cells))}"
    )
    lines.append(
        f"safe abstention (INCONCLUSIVE on faulty telemetry): {_pct(safe_abstention_rate(cells))}"
    )
    lines.append(
        f"behavior detection (clean telemetry): {_pct(behavior_detection_rate(cells))}"
    )
    lines.append(f"clean false positives: {_pct(clean_false_positive_rate(cells))}")
    lines.append("")
    lines.append("PASS rate: behavior x telemetry (0 = the fault is never hidden)")
    matrix = behavior_by_telemetry_matrix(cells)
    cols = sorted(next(iter(matrix.values()), {}))
    header = f"{'behavior':<18}" + "".join(f"{c:>20}" for c in cols)
    lines.append(header)
    for behavior, row in matrix.items():
        line = f"{behavior:<18}" + "".join(f"{_pct(row.get(c, 0.0)):>20}" for c in cols)
        lines.append(line)
    if any(r.get("judged") for r in cells):
        kappa, n_judged = judge_vs_deterministic(cells)
        preserved, n_gaps = evidence_gap_preservation(cells)
        lines.append("")
        if kappa is not None:
            lines.append(
                f"judge vs deterministic kappa (n={n_judged} judged runs): {kappa:.2f}"
            )
        if preserved is not None:
            lines.append(
                f"evidence gaps preserved under judge: {_pct(preserved)} ({n_gaps} gap cells)"
            )
    return "\n".join(lines)


def render_condition_comparison(
    contract: Sequence[Mapping], baseline: Sequence[Mapping]
) -> str:
    """Side-by-side four-condition comparison between the evidence
    contract and a baseline (behavior-only checks). The baseline's
    PASS rate on hidden-violation cells is its false assurance; the
    contract must drive it to zero by abstaining."""
    a = condition_summary(contract)
    b = condition_summary(baseline)
    lines = ["condition comparison: evidence contract vs behavior-only baseline"]
    lines.append(f"{'condition':<40}{'contract':>28}{'baseline':>28}")
    lines.append(f"{'':<40}{'PASS / INCONCL':>28}{'PASS / INCONCL':>28}")
    for condition in a:
        ca, cb = (
            a[condition],
            b.get(condition, {"runs": 0, "passRate": 0.0, "inconclusiveRate": 0.0}),
        )
        lines.append(
            f"{condition:<40}"
            f"{_pct(ca['passRate']) + ' / ' + _pct(ca['inconclusiveRate']):>28}"
            f"{_pct(cb['passRate']) + ' / ' + _pct(cb['inconclusiveRate']):>28}"
        )
    lines.append("")
    lines.append(
        f"false assurance: contract {_pct(false_assurance_rate_2x2(contract))}   "
        f"baseline {_pct(false_assurance_rate_2x2(baseline))}"
    )
    lines.append(
        f"safe abstention: contract {_pct(safe_abstention_rate(contract))}   "
        f"baseline {_pct(safe_abstention_rate(baseline))}"
    )
    return "\n".join(lines)


def render_table(results: Sequence[Mapping]) -> str:
    """The paper's headline table: per-fault detection, per-verifier
    coverage, false positives, overhead, judge agreement."""
    cells = list(results)
    verdicts = [r["verdict"] for r in cells]
    truth = [_positive_label(r) is not None for r in cells]
    det = detection_rate(verdicts, truth)
    fpr = false_positive_rate(verdicts, truth)

    lines = ["watchtower experiment results"]
    lines.append(f"runs={len(cells)}")
    lines.append(
        f"overall detection: {_pct(det)}   false positives (clean runs): {_pct(fpr)}"
    )
    if any(
        (r.get("evidence_fault") or r.get("evidenceFault"))
        and _fault_label(r) not in EVIDENCE_CONTROLS
        for r in cells
    ):
        lines.append(
            f"evidence false assurance: {_pct(false_assurance_rate(cells))}   "
            f"inconclusive: {_pct(inconclusive_rate(cells))}"
        )
    lines.append("")
    lines.append("coverage: fault x verifier (fraction of faulted runs caught)")
    coverage = fault_coverage(cells)
    verifiers = sorted({v for row in coverage.values() for v in row})
    header = (
        f"{'fault':<18}" + "".join(f"{v:>10}" for v in verifiers) + f"{'detected':>10}"
    )
    lines.append(header)
    for fault, row in sorted(coverage.items()):
        fault_runs = [r for r in cells if _positive_label(r) == fault]
        detected = sum(r["verdict"] != "PASS" for r in fault_runs) / len(fault_runs)
        line = (
            f"{fault:<18}"
            + "".join(f"{_pct(row.get(v, 0.0)):>10}" for v in verifiers)
            + f"{_pct(detected):>10}"
        )
        lines.append(line)
    lines.append("")
    lines.append("overhead (per run, tokens / ms)")
    for group, stats in overhead_summary(cells).items():
        lines.append(
            f"  {group:<18} tokens p50={stats['p50Tokens']:>6.0f} p99={stats['p99Tokens']:>7.0f} "
            f"duration p50={stats['p50DurationMs']:>5.0f}ms p99={stats['p99DurationMs']:>6.0f}ms"
        )
    judge_cost = judge_cost_summary(cells)
    if judge_cost:
        lines.append("")
        lines.append("judge cost (measured usage, est. at list prices)")
        for model, stats in judge_cost.items():
            lines.append(
                f"  {model:<22} runs={stats['judgedRuns']:>4} "
                f"in={stats['meanInputTokens']:>6.0f} out={stats['meanOutputTokens']:>6.0f} "
                f"~${stats['estUsdPerRun']:.4f}/run total ~${stats['estUsdTotal']:.4f}"
            )
    kappa, n_judged = judge_vs_deterministic(cells)
    if kappa is not None:
        lines.append("")
        lines.append(
            f"judge vs deterministic agreement (n={n_judged} judged runs): kappa={kappa:.2f}"
        )
    return "\n".join(lines)


# --- CausalTrace: order-invariance and premature-pass metrics ---------------
#
# These metrics consume the per-cell causal artifact produced by the
# serialization sweep. Each cell is one trace with:
#   - causal metadata: `checksum` (canonical DAG sha), `closed` (closure
#     marker present), `serializationVerdicts` (verdict per topological
#     ordering), `evidenceFault` (any causal transport fault), `budget`.
# Like the rest of the module they are pure functions: lists in, numbers
# out.


def _serialization_verdicts(result: Mapping) -> list[str]:
    return list(result.get("serializationVerdicts") or [])


def flip_rate(results: Sequence[Mapping]) -> tuple[float | None, int]:
    """Fraction of multi-serialization traces whose verdict differs across
    valid topological orderings. None when no trace has >1 serialization.
    The headline CausalTrace number: order-invariance means it is 0."""
    cells = [r for r in results if len(_serialization_verdicts(r)) > 1]
    if not cells:
        return None, 0
    flips = sum(len(set(_serialization_verdicts(r))) > 1 for r in cells)
    return flips / len(cells), len(cells)


def premature_pass_rate(results: Sequence[Mapping]) -> tuple[float | None, int]:
    """Fraction of *unclosed* traces (no closure marker) that still produced
    at least one PASS across serializations — the "no premature PASS"
    contract. None when no unclosed trace exists."""
    cells = [r for r in results if not r.get("closed")]
    if not cells:
        return None, 0
    premature = sum(any(v == "PASS" for v in _serialization_verdicts(r)) for r in cells)
    return premature / len(cells), len(cells)


def fault_recall_late(results: Sequence[Mapping]) -> dict[str, float]:
    """Per causal evidence fault, the fraction of its traces whose verdict
    is not PASS (the fault was caught) *when it co-occurs with late
    arrival*. Late arrival is modeled by the sweep reordering spans across
    valid serializations; a fault that is invisible in every order is a
    miss. Empty dict when no causal faults present."""
    faults = sorted({r.get("evidenceFault") for r in results if r.get("evidenceFault")})
    out: dict[str, float] = {}
    for fault in faults:
        cells = [r for r in results if r.get("evidenceFault") == fault]
        caught = sum(
            any(v != "PASS" for v in _serialization_verdicts(r)) for r in cells
        )
        out[fault] = caught / len(cells)
    return out


def tokens_and_time_to_verdict(results: Sequence[Mapping]) -> dict[str, dict]:
    """Per-rendering tokens and wall-clock latency. The budget field is the
    measured tokens/duration for the (single) verification pass; the sweep
    reports the number of serializations it had to render. Reported as
    p50/p99 like the other overhead metrics."""
    cells = list(results)
    if not cells:
        return {}
    tokens = [float(r.get("budget", {}).get("totalTokens", 0)) for r in cells]
    duration = [float(r.get("budget", {}).get("durationMs", 0)) for r in cells]
    serializations = [len(_serialization_verdicts(r)) for r in cells]
    return {
        "tokensPerRun": {
            "mean": sum(tokens) / len(tokens),
            "p50": _percentile(tokens, 50),
            "p99": _percentile(tokens, 99),
        },
        "durationMsPerRun": {
            "mean": sum(duration) / len(duration),
            "p50": _percentile(duration, 50),
            "p99": _percentile(duration, 99),
        },
        "serializationsPerRun": {
            "mean": sum(serializations) / len(serializations),
            "p50": _percentile(serializations, 50),
            "p99": _percentile(serializations, 99),
        },
    }


def verdict_calibration(results: Sequence[Mapping]) -> dict[str, dict]:
    """PASS / FAIL / INCONCLUSIVE distribution per (closed, evidenceFault)
    condition over the first serialization of each trace."""
    groups: dict[str, list[str]] = {}
    for r in results:
        verdicts = _serialization_verdicts(r)
        verdict = verdicts[0] if verdicts else r.get("verdict", "NO_REPORT")
        closed = "closed" if r.get("closed") else "unclosed"
        fault = r.get("evidenceFault") or "clean"
        groups.setdefault(f"{closed} x {fault}", []).append(verdict)
    out: dict[str, dict] = {}
    for key, verdicts in sorted(groups.items()):
        n = len(verdicts)
        out[key] = {
            "runs": n,
            "passRate": sum(v == "PASS" for v in verdicts) / n,
            "failRate": sum(v in ("FAIL", "FLAGGED") for v in verdicts) / n,
            "inconclusiveRate": sum(v == "INCONCLUSIVE" for v in verdicts) / n,
        }
    return out


def render_causal(results: Sequence[Mapping]) -> str:
    """The CausalTrace table: flip rate, premature-pass rate, per-fault
    late recall, tokens/time-to-verdict, and verdict calibration."""
    lines = ["causal order-invariance experiment"]
    lines.append(f"traces={len(results)}")
    flip, n_flip = flip_rate(results)
    premature, n_prem = premature_pass_rate(results)
    if flip is not None:
        lines.append(
            f"flip rate (verdict differs across serializations): {_pct(flip)}  (n={n_flip} multi-serialization traces)"
        )
    if premature is not None:
        lines.append(
            f"premature PASS on unclosed traces: {_pct(premature)}  (n={n_prem})"
        )
    recall = fault_recall_late(results)
    if recall:
        lines.append("")
        lines.append("fault recall under late arrival (catch rate per causal fault)")
        for fault, rate in recall.items():
            lines.append(f"  {fault:<20} {_pct(rate)}")
    lines.append("")
    lines.append("tokens / time to verdict")
    for key, stats in tokens_and_time_to_verdict(results).items():
        lines.append(
            f"  {key:<26} mean={stats['mean']:>10.1f} p50={stats['p50']:>10.1f} p99={stats['p99']:>10.1f}"
        )
    lines.append("")
    lines.append("verdict calibration (first serialization)")
    lines.append(f"{'condition':<34}{'runs':>6}{'PASS':>8}{'FAIL':>8}{'INCONCL':>9}")
    for key, stats in verdict_calibration(results).items():
        lines.append(
            f"{key:<34}{stats['runs']:>6}"
            f"{_pct(stats['passRate']):>8}{_pct(stats['failRate']):>8}{_pct(stats['inconclusiveRate']):>9}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="analyze experiment results")
    parser.add_argument("--results", type=str, default="artifacts/results.json")
    parser.add_argument(
        "--compare",
        type=str,
        action="append",
        default=[],
        help="additional artifacts; pairwise judge kappa + consensus",
    )
    parser.add_argument(
        "--delta",
        type=str,
        default="",
        help="second artifact; report per-fault detection deltas",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check artifact integrity (signature, protocol)",
    )
    parser.add_argument(
        "--false-assurance",
        action="store_true",
        help="render the four-condition false-assurance analysis",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="render the CausalTrace order-invariance analysis",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        help="behavior-only artifact for the condition comparison",
    )
    args = parser.parse_args()
    artifact = load_results(args.results)
    a = artifact["cells"]
    if args.verify:
        print("integrity:", verify_integrity(artifact))
    if args.false_assurance:
        if args.baseline:
            print(render_condition_comparison(a, load_results(args.baseline)["cells"]))
        else:
            print(render_false_assurance(a, name=artifact.get("configName", "")))
    elif args.causal:
        print(render_causal(a))
    else:
        print(render_table(a))
    if args.compare:
        named = [(artifact.get("judgeBackend", "primary"), a)]
        for path in args.compare:
            other = load_results(path)
            named.append((other.get("judgeBackend", path), other["cells"]))
        matrix, n = judge_agreement_matrix(named)
        print()
        print(render_agreement(matrix, n))
    if args.delta:
        print()
        print(render_delta(delta(a, load_results(args.delta)["cells"])))


if __name__ == "__main__":
    main()
