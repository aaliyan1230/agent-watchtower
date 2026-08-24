"""CausalTrace experiment runner (offline, deterministic).

Builds a grid of *distinct* synthetic multi-agent causal traces (the
plan's 30-50 executions: seeded topology variants x flash/pro shape),
applies the causal evidence faults (truncate_closed / drop_link /
orphan_link_target), runs the serialization sweep over each, attaches a
measured-style per-cell budget, and writes a checksummable results
artifact for `analyze.py --causal`.

Fully offline: no Go binary, no LLM — a synthetic causal builder stands
in for recorded runs so `make experiment-causal` is reproducible
anywhere. The Python deterministic verdict in experiments.causal
measures the causal obligations (closure, dangling links); the sweep
then asks whether any valid topological ordering flips it or emits a
premature PASS. The renderings stored on each cell feed the live judge
(experiments.causal_judge) which measures presentation order-sensitivity
of real LLM judge families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .causal import judge_renderings, sweep
from .suite import PROTOCOL_VERSION

CAUSAL_PROTOCOL = "causal-1"

# A trace whose closure marker has been stripped.
TRUNCATE_CLOSED = "truncate_closed"
# A trace with its causal links removed.
DROP_LINK = "drop_link"
# A trace whose link points at an absent span.
ORPHAN_LINK_TARGET = "orphan_link_target"
CAUSAL_FAULTS = [TRUNCATE_CLOSED, DROP_LINK, ORPHAN_LINK_TARGET]

# Trace-shape families, mirroring the real grid's model slots: "flash"
# is the small/cheap agent shape, "pro" the deeper concurrent one. The
# offline builder keys on the *shape* of the DAG, not a live model.
VARIANTS = ["flash", "pro"]

TOOLS = ["search", "retrieve", "extract", "lookup", "code_exec"]
CHAT_OPS = ["chat", "reason", "plan", "annotate"]


@dataclass
class CausalCell:
    trace_id: str
    evidenceFault: str | None
    checksum: str
    closed: bool
    serializationVerdicts: list[str] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    verdict: str = ""
    model: str = "flash"
    renderings: dict = field(default_factory=dict)


def _span(sid, parent, name, attrs, start, links=()):
    out = {
        "spanId": sid,
        "name": name,
        "startTime": f"2026-08-01T10:00:{start:02d}Z",
        "status": "ok",
        "attributes": attrs or {},
    }
    if parent:
        out["parentSpanId"] = parent
    if links:
        out["links"] = list(links)
    return out


def synthetic_spans() -> list[dict]:
    """The canonical two-worker causal trace: supervisor root -> two
    concurrent chat chains, one consuming the other's tool result via a
    data link, with a final answer and closure marker on the root.
    Kept byte-for-byte stable: the fixtures and tests pin it."""
    return [
        _span(
            "s0",
            "",
            "agent.run",
            {"watchtower.trace.closed": "true", "watchtower.completed": "true"},
            0,
        ),
        _span(
            "s1",
            "s0",
            "chat",
            {"gen_ai.operation.name": "chat", "agent.name": "worker-a"},
            1,
        ),
        _span(
            "s2",
            "s1",
            "tool.call",
            {"tool.name": "search", "agent.name": "worker-a"},
            2,
        ),
        _span(
            "s3",
            "s0",
            "chat",
            {"gen_ai.operation.name": "chat", "agent.name": "worker-a"},
            3,
        ),
        _span(
            "s4",
            "s3",
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "agent.name": "worker-b",
                "watchtower.final": "true",
                "watchtower.output": '{"id": 1}',
            },
            4,
            links=[{"spanId": "s2", "attributes": {"watchtower.link.purpose": "data"}}],
        ),
    ]


def _final_span(sid, parent, seed, start):
    """The terminal chat of the trace: final answer + output."""
    s = _span(
        sid,
        parent,
        "chat",
        {
            "gen_ai.operation.name": "chat",
            "agent.name": "worker-b",
            "watchtower.final": "true",
            "watchtower.output": json.dumps({"id": seed}),
        },
        start,
    )
    return s


def synthetic_spans_variant(seed: int, variant: str = "flash") -> list[dict]:
    """A deterministic, *distinct* multi-branch causal trace for one
    (seed, variant). The seed varies fan-out, chain depth, tool mix,
    link purpose and arrival timestamps so every grid execution is a
    different DAG (the checksums decouple); the variant controls the
    overall shape (flash = small, pro = deep concurrent). Same input
    always yields the same span list — the grid is reproducible.

    Invariants every generated trace obeys (verified by tests):
      - one root with the closure marker (closed -> PASS on the clean
        drain);
      - exactly one terminal chat with a final answer, so the clean
        trace passes and `drop_link` still passes (closed + complete);
      - at least one tool call in a non-terminal branch feeding the
        terminal chat by a causal link, so `orphan_link_target` exposes
        a dangling edge and the DAG has real sibling concurrency.
    """
    rng = random.Random(
        int(hashlib.sha256(f"{seed}:{variant}".encode()).hexdigest()[:16], 16)
    )
    if variant == "pro":
        chain_hi = 5  # deeper chains
    else:
        chain_hi = 4
    n_branches = rng.randrange(2, 5)  # 2..4 branching workers
    final_branch = rng.randrange(n_branches)  # which branch ends in the answer

    spans: list[dict] = [
        _span(
            "s0",
            "",
            "agent.run",
            {"watchtower.trace.closed": "true", "watchtower.completed": "true"},
            0,
        )
    ]
    next_id = 1
    branch_roots: list[str] = []
    feeder_tools: list[str] = []

    for b in range(n_branches):
        chain = rng.randrange(2, chain_hi)  # 2..3 (flash) / 2..4 (pro)
        prev = "s0"
        for i in range(chain):
            sid = f"s{next_id}"
            worker = f"worker-{chr(ord('a') + (seed + b * 3 + i) % 26)}"
            if i == chain - 1 and b == final_branch:
                node = _final_span(sid, prev, seed, next_id)
            elif i % 2 == 0:
                node = _span(
                    sid,
                    prev,
                    CHAT_OPS[(seed + b * 5 + i * 2) % len(CHAT_OPS)],
                    {
                        "gen_ai.operation.name": "chat",
                        "agent.name": worker,
                    },
                    next_id,
                )
            else:
                tool = TOOLS[(seed * 7 + b * 11 + i * 5) % len(TOOLS)]
                node = _span(
                    sid,
                    prev,
                    f"tool.{tool}",
                    {
                        "tool.name": tool,
                        "agent.name": worker,
                    },
                    next_id,
                )
                if b != final_branch and len(feeder_tools) < 3:
                    feeder_tools.append(sid)
            spans.append(node)
            prev = sid
            next_id += 1
        branch_roots.append(prev)

    # Causal links: 1-2 tool results from non-terminal branches feed the
    # answer chat — sibling concurrency and non-tree edges.
    final = branch_roots[final_branch]
    if feeder_tools:
        n_links = 1 + rng.randrange(2)  # 1..2
        chosen = rng.sample(feeder_tools, k=min(n_links, len(feeder_tools)))
        links = [
            {
                "spanId": src,
                "attributes": {
                    "watchtower.link.purpose": rng.choice(["data", "control", "causal"])
                },
            }
            for src in chosen
        ]
        for s in spans:
            if s["spanId"] == final:
                s["links"] = links

    # Shuffle per-span start times so arrival order (list order) is not
    # the same as timestamp order — what the sweep/judge are testing.
    times = list(range(1, len(spans) + 1))
    rng.shuffle(times)
    for s, t in zip(spans, times):
        s["startTime"] = f"2026-08-01T10:00:{t:02d}Z"
    return spans


def _strip_closed(spans):
    out = []
    for s in spans:
        s = dict(s)
        if s.get("attributes"):
            attrs = dict(s["attributes"])
            attrs.pop("watchtower.trace.closed", None)
            s["attributes"] = attrs or None
        out.append(s)
    return out


def _strip_links(spans):
    return [{**s, "links": []} if s.get("links") else s for s in spans]


def _orphan_first_link(spans):
    out = []
    for s in spans:
        s = dict(s)
        if s.get("links"):
            links = [dict(l) for l in s["links"]]
            links[0]["spanId"] = "ghost-span"
            s["links"] = links
        out.append(s)
    return out


def _apply_fault(spans, fault):
    if fault is None:
        return spans
    if fault == TRUNCATE_CLOSED:
        return _strip_closed(spans)
    if fault == DROP_LINK:
        return _strip_links(spans)
    if fault == ORPHAN_LINK_TARGET:
        return _orphan_first_link(spans)
    raise ValueError(f"unknown causal fault: {fault}")


def causality_checksum(cells: list[CausalCell]) -> str:
    return hashlib.sha256(
        json.dumps([asdict(c) for c in cells], sort_keys=True).encode()
    ).hexdigest()


def _budget_for(spans: list[dict]) -> dict:
    """Deterministic per-trace verification budget, so tokens/time-to-
    verdict is measured on real non-zero numbers even offline. Scaled by
    the DAG size; identical across a trace's fault cells (the faults do
    not change span counts)."""
    n = len(spans)
    return {
        "totalTokens": 120 + 26 * n,
        "durationMs": 40 + 9 * n,
    }


def build_cells(
    trace_id: str, seed: int = 1, variant: str = "flash", topo_limit: int = 20
) -> list[CausalCell]:
    """Sweep the generated trace plus every causal fault into cells."""
    spans = synthetic_spans_variant(seed, variant)
    budget = dict(_budget_for(spans))
    cells: list[CausalCell] = []
    for fault in [None, *CAUSAL_FAULTS]:
        faulted = _apply_fault(spans, fault)
        result = sweep(faulted, limit=topo_limit)
        verdicts = [s["verdict"] for s in result["serializations"]]
        cells.append(
            CausalCell(
                trace_id=trace_id,
                evidenceFault=fault,
                checksum=result["checksum"],
                closed=result["closed"],
                serializationVerdicts=verdicts,
                budget=budget,
                verdict=verdicts[0] if verdicts else "",
                model=variant,
                renderings=judge_renderings(faulted),
            )
        )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="run the offline CausalTrace sweep")
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/results-causal.json")
    )
    parser.add_argument(
        "--executions",
        type=int,
        default=40,
        help="number of distinct seeded executions (traces)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(VARIANTS),
        help="comma-separated trace-shape families to alternate over",
    )
    parser.add_argument(
        "--topo-limit", type=int, default=20, help="serialization bound per trace"
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    cells: list[CausalCell] = []
    for idx in range(1, args.executions + 1):
        variant = variants[(idx - 1) % len(variants)]
        cells.extend(
            build_cells(
                f"causal-{idx}-{variant}",
                seed=idx,
                variant=variant,
                topo_limit=args.topo_limit,
            )
        )
    payload = {
        "protocolVersion": PROTOCOL_VERSION,
        "causalProtocol": CAUSAL_PROTOCOL,
        "topoLimit": args.topo_limit,
        "executions": args.executions,
        "variants": variants,
        "checksum": causality_checksum(cells),
        "cells": [asdict(c) for c in cells],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"causal sweep: {len(cells)} cells ({args.executions} executions) -> {args.out}"
    )


if __name__ == "__main__":
    main()
