"""CausalTrace order-invariance tooling: canonical DAG, renderer
comparator, and serialization sweep — all pure functions over the
canonical trace envelope written by harness.trace_artifacts.

The CausalTrace claim is that verification must not depend on the order
a flat span list arrives in. This module gives the experiment the two
pieces to measure that, mirroring the Go reference in
watchtower/internal/canonical so pytest runs offline with no built
binary:

  - :func:`canonical_checksum`: a deterministic byte-for-byte form of the
    causal DAG (parent edges + non-parent link edges), sorted by span and
    edge identity — never by arrival order. Two orderings of the same
    trace must hash identically.
  - :func:`topo_orders`: all valid topological orderings of the DAG,
    bounded (the plan's 10-20 per trace), so a sweep can ask "does the
    verdict flip across equivalent serializations?"
  - :func:`renderings`: the three comparator renderings — ordinary text,
    timestamp-sorted, and canonical DAG — plus :func:`deterministic_verdict`
    used to compute flip / premature-pass metrics across them.

The metric functions themselves live in experiments.analyze (pure math);
this module only normalizes a trace into the DAG + serializations the
metrics consume.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Mapping, Sequence

# The span attribute key that marks the trace as closed (definition kept
# in harness.semconv; mirrored here to avoid an import cycle into the
# experiment layer).
TRACE_CLOSED = "watchtower.trace.closed"
FINAL = "watchtower.final"
OUTPUT = "watchtower.output"
COMPLETED = "watchtower.completed"
LINK_PURPOSE = "watchtower.link.purpose"


def _attrs(span: Mapping) -> Mapping[str, str]:
    return span.get("attributes") or {}


def _links(span: Mapping) -> Sequence[Mapping]:
    return span.get("links") or ()


def _is_closed(span: Mapping) -> bool:
    return str(_attrs(span).get(TRACE_CLOSED, "")).lower() == "true"


def trace_closed(spans: Sequence[Mapping]) -> bool:
    """True iff any span carries the explicit end-of-trace marker."""
    return any(_is_closed(s) for s in spans)


def _final_present(spans: Sequence[Mapping]) -> bool:
    """True iff some LLM span declares a non-empty final answer."""
    for s in spans:
        a = _attrs(s)
        if a.get(FINAL) == "true" and str(a.get(OUTPUT, "")).strip():
            return True
    return False


def build_dag(spans: Sequence[Mapping]) -> tuple[dict, list[tuple[str, str, str]]]:
    """Return ``(nodes, edges)`` for the causal DAG of a span list.

    ``nodes`` maps spanId -> span. ``edges`` is the list of ``(from, to,
    purpose)`` triples: one per parent-child edge (purpose ``""``) and one
    per link attribute (the link's purpose, e.g. ``"data"``). Order does
    not matter; the consumers sort. Self- and empty-target links are
    ignored, matching graph.Reconstruct.
    """
    nodes: dict[str, Mapping] = {}
    for span in spans:
        sid = str(span.get("spanId", ""))
        if sid:
            nodes[sid] = span

    edges: set[tuple[str, str, str]] = set()
    for span in spans:
        sid = str(span.get("spanId", "") or "")
        parent = span.get("parentSpanId") or ""
        if parent and parent in nodes and parent != sid:
            edges.add((parent, sid, ""))
        for link in _links(span):
            target = str(link.get("spanId", "") or "")
            if not target or target == sid:
                continue
            purpose = str((link.get("attributes") or {}).get(LINK_PURPOSE, ""))
            edges.add((target, sid, purpose))  # causal: target -> carrier
    return nodes, sorted(edges)


def canonical_render(spans: Sequence[Mapping]) -> str:
    """Deterministic text form of the causal DAG, independent of the
    order the spans arrived in. Mirror of canonical.Graph.Render."""
    nodes, edges = build_dag(spans)
    lines = []
    for sid in sorted(nodes):
        span = nodes[sid]
        lines.append(f"{sid}\t{span.get('kind', '')}\t{span.get('name', '')}")
    for frm, to, purpose in edges:
        lines.append(f"{frm} > {to}\t{purpose}")
    return "\n".join(lines) + ("\n" if lines else "")


def canonical_checksum(spans: Sequence[Mapping]) -> str:
    """sha256 of the canonical render — the order-invariance fingerprint."""
    return hashlib.sha256(canonical_render(spans).encode("utf-8")).hexdigest()


def _topo_out(nodes: Mapping, edges: Sequence[tuple[str, str, str]], limit: int):
    """Iterative bounded enumeration of topological orderings.

    Returns a list of node-id lists. Deterministic: the smallest frontier
    id is expanded first, so two identical DAGs produce identical first
    ``limit`` orderings. Yields None when the DAG has a cycle.
    """
    out: list[list[str]] = []
    adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}
    indegree: dict[str, int] = {nid: 0 for nid in nodes}
    for frm, to, _purpose in edges:
        if frm in adjacency and to in indegree:
            adjacency[frm].append(to)
            indegree[to] += 1
    for nid in adjacency:
        adjacency[nid].sort()

    def frontier(ind: Mapping[str, int]) -> list[str]:
        return sorted(nid for nid, d in ind.items() if d == 0)

    stack: list[tuple[list[str], dict[str, int]]] = [([], indegree)]
    while stack and len(out) < limit:
        order, ind = stack.pop()
        if len(order) == len(nodes):
            out.append(order)
            continue
        for cand in reversed(frontier(ind)):
            next_ind = dict(ind)
            next_ind[cand] = -1
            for child in adjacency.get(cand, []):
                next_ind[child] -= 1
            stack.append((order + [cand], next_ind))
    return out


def topo_orders(spans: Sequence[Mapping], limit: int = 20) -> list[list[str]] | None:
    """Valid topological orderings of the DAG, bounded to ``limit``.
    None when the DAG is cyclic (no ordering exists) or empty."""
    nodes, edges = build_dag(spans)
    if not nodes:
        return None
    if not _is_acyclic(nodes, edges):
        return None
    return _topo_out(nodes, edges, limit)


def _is_acyclic(nodes: Mapping, edges: Sequence[tuple[str, str, str]]) -> bool:
    """Kahn-style reachability check that never completes for a cycle."""
    adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}
    indegree: dict[str, int] = {nid: 0 for nid in nodes}
    for frm, to, _purpose in edges:
        if frm in adjacency and to in indegree:
            adjacency[frm].append(to)
            indegree[to] += 1
    queue = deque(sorted(nid for nid, d in indegree.items() if d == 0))
    seen = 0
    while queue:
        cur = queue.popleft()
        seen += 1
        for child in adjacency[cur]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return seen == len(nodes)


def time_sorted(spans: Sequence[Mapping]) -> list[Mapping]:
    """(b) timestamp-sorted rendering — today's arrival-coupled behavior."""
    return sorted(
        spans, key=lambda s: (str(s.get("startTime", "")), str(s.get("spanId", "")))
    )


def text_render(spans: Sequence[Mapping]) -> str:
    """(a) ordinary text rendering in the given (arrival) order."""
    return "\n".join(f"{s.get('spanId', '')}\t{s.get('name', '')}" for s in spans)


def renderings(spans: Sequence[Mapping]) -> dict[str, str]:
    """The three comparator renderings keyed by name: text (arrival order),
    time-sorted, and canonical DAG."""
    return {
        "text": text_render(spans),
        "time_sorted": text_render(time_sorted(spans)),
        "canonical": canonical_render(spans),
    }


def _fact_line(span: Mapping) -> str:
    """One compact, fully-specified span line for the LLM judge renderings:
    id + name + parent + attributes + link targets. The ordered renderings
    are the same facts as the canonical DAG, only arranged by arrival /
    timestamp order, so a judge flip across them measures order-sensitivity
    rather than missing information."""
    attributes = _attrs(span)
    facts = [f"{k}={attributes[k]}" for k in sorted(attributes)]
    links = [
        f"{l.get('spanId')}({str((l.get('attributes') or {}).get(LINK_PURPOSE, ''))})"
        for l in _links(span)
    ]
    parts = [
        str(span.get("spanId", "")),
        str(span.get("name", "")),
        "p=" + str(span.get("parentSpanId") or "-"),
    ]
    if facts:
        parts.append(" ".join(facts))
    if links:
        parts.append("links=" + ",".join(sorted(links)))
    return "\t".join(parts)


def judge_renderings(spans: Sequence[Mapping]) -> dict[str, str]:
    """Factual renderings consumed by the live LLM judge (causal_judge).

    All three carry the *same* causal facts so that a verdict flip across
    them is a genuine order/presentation effect, not an information gap:

      - ``text``: full span facts in arrival order (today's OTLP dump);
      - ``time_sorted``: the same facts sorted by timestamp (today's
        arrival-coupled behavior);
      - ``canonical``: the DAG — every node with its facts, then every
        causal edge (parent + links) — order-invariant by construction.

    This is deliberately richer than :func:`canonical_render`, which stays
    the byte-stable Go mirror that canonical_checksum hashes.
    """
    text = "\n".join(_fact_line(s) for s in spans)
    timed = "\n".join(_fact_line(s) for s in time_sorted(spans))
    nodes, edges = build_dag(spans)
    node_lines = [
        f"{sid}\t{nodes[sid].get('kind', '')}\t{nodes[sid].get('name', '')}\t"
        + _fact_line(nodes[sid]).split("\t", 2)[-1]
        for sid in sorted(nodes)
    ]
    edge_lines = [f"{frm} > {to}\t{purpose}" for frm, to, purpose in edges]
    canonical = "\n".join(node_lines + edge_lines)
    return {
        "text": text,
        "time_sorted": timed,
        "canonical": canonical,
    }


def deterministic_verdict(spans: Sequence[Mapping]) -> str:
    """A small purely-deterministic verdict over the *causal obligations*
    the CausalTrace verifier enforces, used to measure flip and premature
    pass without shipping a built Go binary into pytest.

    Ordering mirrors report.verdictFor: an unclosed trace is INCONCLUSIVE
    (no premature PASS); a dangling causal link or missing parent is an
    evidence gap -> INCONCLUSIVE; a missing final answer fails; a clean
    closed trace passes. It is deliberately NOT a full re-implementation
    of the Go verifiers — it measures only the causal invariance the sweep
    cares about.
    """
    nodes, edges = build_dag(spans)
    if not trace_closed(spans):
        return "INCONCLUSIVE"
    for span in spans:
        parent = span.get("parentSpanId") or ""
        if parent and parent not in nodes:
            return "INCONCLUSIVE"
        for link in _links(span):
            target = str(link.get("spanId", "") or "")
            if target and target not in nodes:
                return "INCONCLUSIVE"
    return "PASS" if _final_present(spans) else "FAIL"


def sweep(
    spans: Sequence[Mapping],
    limit: int = 20,
    *,
    verdict=deterministic_verdict,
) -> dict:
    """Run the serialization sweep over one trace: for each topological
    ordering of the causal DAG, rebuild the span order and compute the
    verdict. Also reports the canonical checksum and the three renderings.

    Returns a dict with the per-serialization verdicts (for flip-rate), the
    canonical checksum (for checksum stability), and whether the trace was
    closed (for premature-pass measurement).
    """
    orders = topo_orders(spans, limit) or []
    by_id = {str(s.get("spanId", "")): s for s in spans}
    per = []
    for order in orders:
        ordered = [by_id[nid] for nid in order if nid in by_id]
        per.append({"order": order, "verdict": verdict(ordered)})
    return {
        "checksum": canonical_checksum(spans),
        "closed": trace_closed(spans),
        "serializations": per,
        "renderings": renderings(spans),
    }
