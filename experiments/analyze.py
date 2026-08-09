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


def fault_coverage(results: Sequence[Mapping]) -> dict[str, dict[str, float]]:
    """For each fault type, the fraction of its runs each verifier
    caught — the "where deterministic catches what" matrix."""
    faults = sorted({r["fault"] for r in results if r["fault"]})
    verifiers = sorted({f["verifier"] for r in results if r["fault"] for f in r["findings"]})
    out: dict[str, dict[str, float]] = {}
    for fault in faults:
        runs = [r for r in results if r["fault"] == fault]
        out[fault] = {
            v: sum(any(f["verifier"] == v for f in r["findings"]) for r in runs) / len(runs)
            for v in verifiers
        }
    return out


def overhead_summary(results: Sequence[Mapping]) -> dict[str, dict[str, float]]:
    """Mean tokens and wall-clock duration per fault group — the
    overhead the paper reports (cost is per-token, so tokens are the
    honest proxy without live pricing)."""
    groups = sorted({r["fault"] or "clean" for r in results})
    out: dict[str, dict[str, float]] = {}
    for group in groups:
        runs = [r for r in results if (r["fault"] or "clean") == group]
        tokens = [r["budget"].get("totalTokens", 0) for r in runs]
        duration = [r["budget"].get("durationMs", 0) for r in runs]
        out[group] = {"meanTokens": sum(tokens) / len(tokens), "meanDurationMs": sum(duration) / len(duration)}
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


def judge_agreement(a: Sequence[Mapping], b: Sequence[Mapping]) -> tuple[float | None, int]:
    """Kappa between two judges' opinions on the same runs, matched by
    (fault, seed, run) — the inter-provider agreement metric (e.g.
    Gemini judge vs Bedrock judge)."""
    key = lambda r: (r["fault"], r["seed"], r["model"], r["run"])
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


def _pct(x: float) -> str:
    return f"{x:.0%}"


def cells_signature(cells: Sequence[Mapping]) -> str:
    """sha256 over the grid identity of the artifact's cells — the
    self-integrity check: any truncation or tampering changes it."""
    import hashlib

    identity = [{"fault": r["fault"], "seed": r["seed"], "model": r["model"], "run": r["run"]} for r in cells]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def verify_integrity(artifact: Mapping) -> str:
    """Reports whether the artifact's cells match its recorded
    signature, and whether the recorded protocol versions line up."""
    problems = []
    if cells_signature(artifact.get("cells", [])) != artifact.get("cellsSignature"):
        problems.append("cellsSignature mismatch (artifact tampered or truncated)")
    if artifact.get("protocolVersion") != "1":
        problems.append(f"unknown suite protocol {artifact.get('protocolVersion')!r}")
    return "; ".join(problems) or "ok"


def delta(a: Sequence[Mapping], b: Sequence[Mapping]) -> dict:
    """Compares two artifacts per fault: detection and false-positive
    deltas plus judge-vs-deterministic kappa on each. Keys on the
    (fault, seed, run) identity so only paired runs are compared."""
    key = lambda r: (r["fault"], r["seed"], r["model"], r["run"])
    bmap = {key(r): r for r in b}
    paired = [(ra, bmap[key(ra)]) for ra in a if key(ra) in bmap]
    faults = sorted({ra["fault"] for ra, _ in paired if ra["fault"]})
    out: dict = {"pairedRuns": len(paired)}
    for fault in faults:
        runs = [(ra, rb) for ra, rb in paired if ra["fault"] == fault]
        det_a = sum(ra["verdict"] != "PASS" for ra, _ in runs) / len(runs)
        det_b = sum(rb["verdict"] != "PASS" for _, rb in runs) / len(runs)
        out[fault] = {"detA": det_a, "detB": det_b, "delta": det_b - det_a}
    clean = [(ra, rb) for ra, rb in paired if ra["fault"] is None]
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
            lines.append(f"  clean     fprA={row['fprA']:.0%} fprB={row['fprB']:.0%} delta={row['delta']:+.0%}")
        else:
            lines.append(f"  {key:<18} detA={row['detA']:.0%} detB={row['detB']:.0%} delta={row['delta']:+.0%}")
    return "\n".join(lines)


def render_table(results: Sequence[Mapping]) -> str:
    """The paper's headline table: per-fault detection, per-verifier
    coverage, false positives, overhead, judge agreement."""
    cells = list(results)
    verdicts = [r["verdict"] for r in cells]
    truth = [r["fault"] is not None for r in cells]
    det = detection_rate(verdicts, truth)
    fpr = false_positive_rate(verdicts, truth)

    lines = ["watchtower experiment results"]
    lines.append(f"runs={len(cells)}")
    lines.append(f"overall detection: {_pct(det)}   false positives (clean runs): {_pct(fpr)}")
    lines.append("")
    lines.append("coverage: fault x verifier (fraction of faulted runs caught)")
    coverage = fault_coverage(cells)
    verifiers = sorted({v for row in coverage.values() for v in row})
    header = f"{'fault':<18}" + "".join(f"{v:>10}" for v in verifiers) + f"{'detected':>10}"
    lines.append(header)
    for fault, row in sorted(coverage.items()):
        fault_runs = [r for r in cells if r["fault"] == fault]
        detected = sum(r["verdict"] != "PASS" for r in fault_runs) / len(fault_runs)
        line = f"{fault:<18}" + "".join(f"{_pct(row.get(v, 0.0)):>10}" for v in verifiers) + f"{_pct(detected):>10}"
        lines.append(line)
    lines.append("")
    lines.append("overhead (mean per run)")
    for group, stats in overhead_summary(cells).items():
        lines.append(f"  {group:<18} tokens={stats['meanTokens']:>6.0f}  duration={stats['meanDurationMs']:>6.0f}ms")
    kappa, n_judged = judge_vs_deterministic(cells)
    if kappa is not None:
        lines.append("")
        lines.append(f"judge vs deterministic agreement (n={n_judged} judged runs): kappa={kappa:.2f}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="analyze experiment results")
    parser.add_argument("--results", type=str, default="artifacts/results.json")
    parser.add_argument("--compare", type=str, default="", help="second artifact; report kappa between the two judges")
    parser.add_argument("--delta", type=str, default="", help="second artifact; report per-fault detection deltas")
    parser.add_argument("--verify", action="store_true", help="check artifact integrity (signature, protocol)")
    args = parser.parse_args()
    artifact = load_results(args.results)
    a = artifact["cells"]
    if args.verify:
        print("integrity:", verify_integrity(artifact))
    print(render_table(a))
    if args.compare:
        b = load_results(args.compare)["cells"]
        kappa, n = judge_agreement(a, b)
        line = f"judge A vs judge B agreement (n={n} paired judged runs): kappa={kappa:.2f}" if kappa is not None else f"judge A vs judge B: not enough paired judged runs (n={n})"
        print()
        print(line)
    if args.delta:
        print()
        print(render_delta(delta(a, load_results(args.delta)["cells"])))


if __name__ == "__main__":
    main()
