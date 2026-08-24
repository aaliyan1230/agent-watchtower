"""CausalTrace runner: fault application and cell construction."""

from experiments import causal_run
from experiments.causal import (
    build_dag,
    canonical_checksum,
    deterministic_verdict,
    sweep,
    trace_closed,
)


def test_synthetic_spans_have_causal_link():
    spans = causal_run.synthetic_spans()
    nodes, edges = build_dag(spans)
    assert ("s2", "s4", "data") in edges  # worker-b chat consumes worker-a tool result
    assert trace_closed(spans) is True
    assert deterministic_verdict(spans) == "PASS"


def test_fault_strips_closure():
    spans = causal_run._apply_fault(
        causal_run.synthetic_spans(), causal_run.TRUNCATE_CLOSED
    )
    assert trace_closed(spans) is False


def test_fault_strips_links():
    spans = causal_run._apply_fault(causal_run.synthetic_spans(), causal_run.DROP_LINK)
    nodes, edges = build_dag(spans)
    assert not any(purpose == "data" for _, _, purpose in edges)


def test_fault_orphans_link_target():
    spans = causal_run._apply_fault(
        causal_run.synthetic_spans(), causal_run.ORPHAN_LINK_TARGET
    )
    assert deterministic_verdict(spans) == "INCONCLUSIVE"


def test_build_cells_clean_and_all_faults():
    cells = causal_run.build_cells("causal-1")
    faults = {c.evidenceFault for c in cells}
    assert faults == {None, *causal_run.CAUSAL_FAULTS}
    clean = next(c for c in cells if c.evidenceFault is None)
    assert clean.closed is True
    assert clean.verdict == "PASS"
    trunc = next(c for c in cells if c.evidenceFault == causal_run.TRUNCATE_CLOSED)
    assert trunc.closed is False
    orphan = next(c for c in cells if c.evidenceFault == causal_run.ORPHAN_LINK_TARGET)
    assert orphan.verdict == "INCONCLUSIVE"


def test_variant_traces_are_distinct_and_pass():
    # The grid's executions are genuinely distinct: every (seed, variant)
    # in the default range produces its own canonical checksum.
    checksums = {
        canonical_checksum(causal_run.synthetic_spans_variant(seed, variant))
        for seed in range(1, 41)
        for variant in causal_run.VARIANTS
    }
    assert len(checksums) == 80
    for seed in range(1, 41):
        for variant in causal_run.VARIANTS:
            spans = causal_run.synthetic_spans_variant(seed, variant)
            assert trace_closed(spans) is True
            assert deterministic_verdict(spans) == "PASS"


def test_variant_traces_are_deterministic():
    a = causal_run.synthetic_spans_variant(7, "pro")
    b = causal_run.synthetic_spans_variant(7, "pro")
    assert a == b
    assert canonical_checksum(a) == canonical_checksum(b)


def test_variant_sweep_no_premature_pass():
    # An unclosed variant trace must be INCONCLUSIVE in every serialization.
    spans = causal_run.synthetic_spans_variant(5, "pro")
    result = sweep(spans, limit=20)
    assert result["closed"] is True
    trunc = causal_run._apply_fault(spans, causal_run.TRUNCATE_CLOSED)
    t = sweep(trunc, limit=20)
    assert t["closed"] is False
    assert {s["verdict"] for s in t["serializations"]} == {"INCONCLUSIVE"}


def test_build_cells_carry_budget_and_renderings():
    cells = causal_run.build_cells("causal-9", seed=9, variant="pro")
    assert len(cells) == 4
    clean = cells[0]
    assert clean.budget["totalTokens"] > 0
    assert clean.budget["durationMs"] > 0
    assert set(clean.renderings) == {"text", "time_sorted", "canonical"}
    assert clean.renderings["text"] != clean.renderings["canonical"]
    assert all(c.model == "pro" for c in cells)


def test_build_cells_reproducible():
    a = causal_run.build_cells("causal-3", seed=3, variant="flash")
    b = causal_run.build_cells("causal-3", seed=3, variant="flash")
    assert (a[0].checksum, a[0].budget) == (b[0].checksum, b[0].budget)
