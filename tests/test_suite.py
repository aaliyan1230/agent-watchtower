"""The experiment grid is deterministic and complete."""

from experiments.suite import build_evidence_grid, build_grid, checksum


def test_grid_has_clean_controls():
    cells = build_grid(seeds=[1, 2], models=["flash"], runs=2)
    clean = [c for c in cells if c.fault is None]
    assert len(clean) == 4  # seeds x models x runs
    faulted = [c for c in cells if c.fault is not None]
    assert len(faulted) == 6 * 4


def test_grid_is_deterministic():
    a = build_grid()
    b = build_grid()
    assert checksum(a) == checksum(b)


def test_evidence_grid_has_clean_controls_and_three_faults():
    cells = build_evidence_grid(seeds=[1], models=["flash"], runs=1)
    assert len(cells) == 4
    assert [c.evidence_fault for c in cells] == [None, "drop_parent", "duplicate_span", "reorder_spans"]
    assert all(c.fault is None for c in cells)
