"""CausalTrace experiment runner (offline, deterministic).

Builds a small synthetic set of multi-agent causal traces, applies the
causal evidence faults (truncate_closed / drop_link / orphan_link_target),
runs the serialization sweep over each, and writes a checksummable results
artifact for `analyze.py --causal`.

Fully offline: no Go binary, no LLM — a synthetic causal builder stands in
for recorded runs so `make experiment-causal` is reproducible anywhere. The
Python deterministic verdict in experiments.causal measures the causal
obligations (closure, dangling links); the sweep then asks whether any
valid topological ordering flips it or emits a premature PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .causal import sweep
from .suite import PROTOCOL_VERSION

CAUSAL_PROTOCOL = "causal-1"

# A trace whose closure marker has been stripped.
TRUNCATE_CLOSED = "truncate_closed"
# A trace with its causal links removed.
DROP_LINK = "drop_link"
# A trace whose link points at an absent span.
ORPHAN_LINK_TARGET = "orphan_link_target"
CAUSAL_FAULTS = [TRUNCATE_CLOSED, DROP_LINK, ORPHAN_LINK_TARGET]


@dataclass
class CausalCell:
    trace_id: str
    evidenceFault: str | None
    checksum: str
    closed: bool
    serializationVerdicts: list[str] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    verdict: str = ""


def synthetic_spans() -> list[dict]:
    """A deterministic two-worker causal trace: supervisor root -> two
    concurrent chat chains, one consuming the other's tool result via a
    data link, with a final answer and closure marker on the root."""

    def span(sid, parent, name, attrs, start, links=()):
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

    return [
        span(
            "s0",
            "",
            "agent.run",
            {"watchtower.trace.closed": "true", "watchtower.completed": "true"},
            0,
        ),
        span(
            "s1",
            "s0",
            "chat",
            {"gen_ai.operation.name": "chat", "agent.name": "worker-a"},
            1,
        ),
        span(
            "s2",
            "s1",
            "tool.call",
            {"tool.name": "search", "agent.name": "worker-a"},
            2,
        ),
        span(
            "s3",
            "s0",
            "chat",
            {"gen_ai.operation.name": "chat", "agent.name": "worker-a"},
            3,
        ),
        span(
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


def build_cells(trace_id: str) -> list[CausalCell]:
    """Sweep the clean trace plus every causal fault into cells."""
    clean = synthetic_spans()
    cells: list[CausalCell] = []
    for fault in [None, *CAUSAL_FAULTS]:
        spans = _apply_fault(clean, fault)
        result = sweep(spans, limit=20)
        verdicts = [s["verdict"] for s in result["serializations"]]
        cells.append(
            CausalCell(
                trace_id=trace_id,
                evidenceFault=fault,
                checksum=result["checksum"],
                closed=result["closed"],
                serializationVerdicts=verdicts,
                verdict=verdicts[0] if verdicts else "",
            )
        )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="run the offline CausalTrace sweep")
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/results-causal.json")
    )
    parser.add_argument(
        "--traces", type=int, default=12, help="number of synthetic traces (seeds)"
    )
    parser.add_argument(
        "--topo-limit", type=int, default=20, help="serialization bound per trace"
    )
    args = parser.parse_args()

    cells: list[CausalCell] = []
    for seed in range(1, args.traces + 1):
        # The seed only changes trace naming here; causal structure is
        # fixed so the sweep is deterministic and exactly reproducible.
        cells.extend(build_cells(f"causal-{seed}"))
    payload = {
        "protocolVersion": PROTOCOL_VERSION,
        "causalProtocol": CAUSAL_PROTOCOL,
        "topoLimit": args.topo_limit,
        "checksum": causality_checksum(cells),
        "cells": [asdict(c) for c in cells],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"causal sweep: {len(cells)} cells -> {args.out}")


if __name__ == "__main__":
    main()
