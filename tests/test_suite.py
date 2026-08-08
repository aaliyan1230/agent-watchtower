"""The experiment grid is deterministic and complete."""

from experiments.suite import build_grid, checksum


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
