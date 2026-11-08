"""End-to-end check of the decision layer through the real server.

Held-out tau-bench runs are posted from their recorded clean traces, with
each run's transport fault re-applied, to a server that loads a fitted
decision model. The client follows every decision as a deployment would:
a replay re-sends the clean trace, anything else is final. Probes are
charged at the model's prices and outcomes at the run's label, so the
result is directly comparable to `watchtower decide` on the same split.
The judge runs live, and only when a decision buys it.

Usage: python -m experiments.decide_replay --model artifacts/decide-model-taubench.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harness.server import REPO_ROOT, spawn_server
from harness.trace_artifacts import apply_evidence_fault

from .replay import load_trace, post_envelope
from .run import judge_server_env

ENDPOINT = "http://127.0.0.1:4318"
ADDR = "127.0.0.1:4318"
CONFIG = REPO_ROOT / "watchtower" / "testdata" / "taubench_config.json"


def outcome_cost(action: str, faulty: bool, costs: dict[str, float]) -> float:
    if action == "accept":
        return costs["miss"] if faulty else 0.0
    if action == "halt":
        return 0.0 if faulty else costs["falseHalt"]
    if action == "escalate":
        return costs["human"]
    raise ValueError(f"not a final action: {action}")


def follow(trace: dict[str, Any], faulted: dict[str, Any], endpoint: str) -> dict[str, Any]:
    """Post the run as the collector saw it and act on the decision.
    Returns the probes bought and the final action."""
    report = post_envelope(endpoint, faulted)
    decision = report["decision"]
    probes = ["judge"] if decision["first"] == "judge" else []
    final = decision["action"]
    if final == "replay":
        probes.append("replay")
        replayed = post_envelope(endpoint, trace)["decision"]
        if replayed["first"] == "judge":
            probes.append("judge")
        final = replayed["action"]
        if final == "replay":
            # The recollection still has gaps; replaying again cannot help.
            final = replayed["fallback"]
    return {"first": decision["first"], "final": final, "probes": probes, "verdict": report["verdict"]}


def summarize(rows: list[dict[str, Any]], costs: dict[str, float]) -> dict[str, Any]:
    n = len(rows)
    probe_price = {"judge": costs["judge"], "replay": costs["replay"]}
    total = sum(
        sum(probe_price[p] for p in r["probes"]) + outcome_cost(r["final"], r["faulty"], costs) for r in rows
    )
    count = lambda pred: sum(1 for r in rows if pred(r))
    return {
        "runs": n,
        "costPerRun": total / n,
        "acceptedFaulty": count(lambda r: r["final"] == "accept" and r["faulty"]) / n,
        "acceptPer100": 100 * count(lambda r: r["final"] == "accept") / n,
        "falseHaltPer100": 100 * count(lambda r: r["final"] == "halt" and not r["faulty"]) / n,
        "escalatePer100": 100 * count(lambda r: r["final"] == "escalate") / n,
        "judgeCallsPer100": 100 * sum(r["probes"].count("judge") for r in rows) / n,
        "replaysPer100": 100 * sum(r["probes"].count("replay") for r in rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="replay held-out runs through a server with a decision model")
    parser.add_argument("--results", type=Path, default=Path("artifacts/results-taubench-test.json"))
    parser.add_argument("--traces", type=Path, default=Path("artifacts/traces-taubench-test"))
    parser.add_argument("--model", type=Path, required=True, help="model saved by `watchtower decide --save`")
    parser.add_argument("--test-seeds", type=str, default="4,5")
    parser.add_argument("--out", type=Path, default=Path("artifacts/results-decide-replay.json"))
    args = parser.parse_args()

    seeds = {int(s) for s in args.test_seeds.split(",")}
    cells = [c for c in json.loads(args.results.read_text())["cells"] if c["seed"] in seeds]
    paths = {}
    for line in (args.traces / "manifest.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("capture") == "clean-before-transport-mutation":
            paths[rec["traceId"]] = args.traces / rec["tracePath"]
    costs = json.loads(args.model.read_text())["costs"]

    config = json.loads(CONFIG.read_text())
    config["decision"] = {"model": str(args.model.resolve())}
    served_config = args.out.with_suffix(".config.json")
    served_config.write_text(json.dumps(config, indent=2))

    proc = spawn_server(ENDPOINT, ADDR, str(served_config), env=judge_server_env(str(CONFIG)))
    try:
        rows = []
        for i, cell in enumerate(cells, 1):
            trace = load_trace(paths[cell["traceId"]])
            faulted = apply_evidence_fault(trace, cell["evidence_fault"]) if cell["evidence_fault"] else trace
            row = follow(trace, faulted, ENDPOINT)
            row.update(task=cell["task"], condition=cell["condition"], faulty=cell["faulty"])
            rows.append(row)
            print(f"[{i}/{len(cells)}] task={cell['task']:<3} {cell['condition']:<17} "
                  f"{row['verdict']:<13} {row['first']:>8} -> {row['final']:<8} faulty={cell['faulty']}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    summary = summarize(rows, costs)
    args.out.write_text(json.dumps({"model": str(args.model), "testSeeds": sorted(seeds), "costs": costs,
                                    "summary": summary, "runs": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
