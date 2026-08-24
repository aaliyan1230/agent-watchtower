"""CausalTrace runner: fault application and cell construction."""

from experiments import causal_run
from experiments.causal import build_dag, deterministic_verdict, trace_closed


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
