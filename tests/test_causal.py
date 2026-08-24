"""CausalTrace order-invariance tooling: pure functions, so test them."""

import pytest

from experiments.causal import (
    build_dag,
    canonical_checksum,
    canonical_render,
    deterministic_verdict,
    renderings,
    sweep,
    time_sorted,
    topo_orders,
    trace_closed,
)


def _span(sid, parent="", name="chat", attrs=None, links=(), start="0"):
    out = {"spanId": sid, "name": name, "startTime": start, "status": "ok"}
    if parent:
        out["parentSpanId"] = parent
    if attrs:
        out["attributes"] = attrs
    if links:
        out["links"] = list(links)
    return out


def _closed_trace():
    """A closed, causally-complete trace: supervisor -> chat -> final
    with a sibling tool call and a data link into the final chat."""
    root = _span(
        "s0",
        name="agent.run",
        attrs={"watchtower.trace.closed": "true", "watchtower.completed": "true"},
    )
    chat1 = _span("s1", parent="s0", attrs={"gen_ai.operation.name": "chat"})
    tool = _span(
        "s2", parent="s0", name="tool.call", attrs={"tool.name": "search"}, start="1"
    )
    chat2 = _span(
        "s3",
        parent="s0",
        attrs={
            "gen_ai.operation.name": "chat",
            "watchtower.final": "true",
            "watchtower.output": "done",
        },
        start="2",
        links=[{"spanId": "s2", "attributes": {"watchtower.link.purpose": "data"}}],
    )
    return [root, chat1, chat2, tool]


def test_build_dag_edges():
    nodes, edges = build_dag(_closed_trace())
    assert set(nodes) == {"s0", "s1", "s2", "s3"}
    # Parent edges s0->s1, s0->s2, s0->s3; link edge s2->s3 (data).
    assert ("s0", "s1", "") in edges
    assert ("s0", "s2", "") in edges
    assert ("s0", "s3", "") in edges
    assert ("s2", "s3", "data") in edges
    assert len(edges) == 4


def test_canonical_checksum_is_order_invariant():
    normal = _closed_trace()
    reversed_ = list(reversed(_closed_trace()))
    assert canonical_checksum(normal) == canonical_checksum(reversed_)
    assert canonical_render(normal) == canonical_render(reversed_)


def test_canonical_render_sorted():
    render = canonical_render(_closed_trace())
    nodes = [line.split("\t")[0] for line in render.splitlines() if " > " not in line]
    # Node block then edge block; nodes sorted.
    assert nodes == sorted(nodes)
    assert "s2 > s3\tdata" in render


def test_topo_orders_bounded_and_valid():
    spans = _closed_trace()
    orders = topo_orders(spans, limit=20)
    assert orders, "expected at least one ordering"
    assert len(orders) <= 20
    edge_pairs = {(f, t) for f, t, _ in build_dag(spans)[1]}
    node_ids = {s["spanId"] for s in spans}
    for order in orders:
        assert set(order) == node_ids
        pos = {nid: i for i, nid in enumerate(order)}
        for f, t in edge_pairs:
            assert pos[f] < pos[t], f"order violates edge {f}->{t}: {order}"
    # Deterministic: same first set.
    assert topo_orders(spans, 20) == orders


def test_topo_orders_cycle_yields_none():
    # A genuine 2-cycle via parent relationships.
    a2 = _span("a", parent="b")
    b2 = _span("b", parent="a")
    assert topo_orders([a2, b2]) is None


def test_trace_closed():
    assert trace_closed(_closed_trace()) is True
    unclosed = _closed_trace()
    unclosed[0]["attributes"] = {"watchtower.completed": "true"}
    assert trace_closed(unclosed) is False


def test_deterministic_verdict_closure():
    # Unclosed trace -> INCONCLUSIVE (no premature PASS).
    unclosed = _closed_trace()
    unclosed[0]["attributes"] = {"watchtower.completed": "true"}
    assert deterministic_verdict(unclosed) == "INCONCLUSIVE"


def test_deterministic_verdict_dangling_link_inconclusive():
    spans = _closed_trace()
    # Orphan a link target that is not in the node set.
    for s in spans:
        if s["spanId"] == "s3":
            s["links"] = [
                {"spanId": "ghost", "attributes": {"watchtower.link.purpose": "data"}}
            ]
    assert deterministic_verdict(spans) == "INCONCLUSIVE"


def test_deterministic_verdict_closed_clean_pass_and_fail():
    assert deterministic_verdict(_closed_trace()) == "PASS"
    # Remove the final-answer span: closed but incomplete -> FAIL.
    no_final = [s for s in _closed_trace() if s["spanId"] != "s3"]
    assert deterministic_verdict(no_final) == "FAIL"


def test_renderings_three_forms():
    r = renderings(_closed_trace())
    assert set(r) == {"text", "time_sorted", "canonical"}
    assert r["canonical"] != r["text"]


def test_time_sorted_sorts_by_start():
    spans = _closed_trace()
    ordered = time_sorted(spans)
    starts = [s["startTime"] for s in ordered]
    assert starts == sorted(starts)


def test_sweep_no_flips_on_closed_clean_trace():
    result = sweep(_closed_trace(), limit=20)
    assert result["closed"] is True
    assert len(result["checksum"]) == 64
    verdicts = {s["verdict"] for s in result["serializations"]}
    # Every valid serialization of a closed, causally-complete trace gives
    # the same (PASS) verdict — order-invariance holds.
    assert verdicts == {"PASS"}
