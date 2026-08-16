"""Experiment suite: the parameter grid, built deterministically.

Phase 2 turns this into `run matrix -> checksummed artifacts`; today it
defines the grid itself: fault type x seed x model variant x run, plus
clean runs as the false-positive control. Building the grid is a pure
function — the same seed set yields the same cells, always.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from harness.faults import EvidenceFaultKind, FaultKind

# Frozen-protocol marker: bump when the experiment definition changes
# (fault steps, scripts, configs) so old artifacts are never compared
# silently against new ones.
PROTOCOL_VERSION = "4"

MODEL_VARIANTS = ["flash", "pro"]  # cheap tier for bulk, pro tier for a stratified sample
SEEDS = [1, 2, 3, 4, 5]  # sized from pilot results in Phase 2
RUNS_PER_CELL = 3


@dataclass(frozen=True)
class ExperimentCell:
    """One run slot in the matrix. fault=None means a clean run — the
    false-positive control, without which detection rates mean nothing."""

    fault: str | None
    seed: int
    model: str
    run: int
    evidence_fault: str | None = None


def build_grid(
    faults: list[str] | None = None,
    seeds: list[int] = SEEDS,
    models: list[str] = MODEL_VARIANTS,
    runs: int = RUNS_PER_CELL,
) -> list[ExperimentCell]:
    faults = faults or [f.value for f in FaultKind]
    cells: list[ExperimentCell] = []
    for fault in faults + [None]:  # None = clean control
        for seed in seeds:
            for model in models:
                for run in range(1, runs + 1):
                    cells.append(ExperimentCell(fault=fault, seed=seed, model=model, run=run))
    return cells


def build_evidence_grid(
    seeds: list[int] = SEEDS,
    models: list[str] = MODEL_VARIANTS,
    runs: int = RUNS_PER_CELL,
) -> list[ExperimentCell]:
    """Build a clean-behavior grid with controlled telemetry faults.

    Keeping behavior clean isolates false assurance caused by evidence
    loss or duplication. The same cell shape feeds the existing runner.
    """
    evidence_faults = [None, *(f.value for f in EvidenceFaultKind)]
    cells: list[ExperimentCell] = []
    for evidence_fault in evidence_faults:
        for seed in seeds:
            for model in models:
                for run in range(1, runs + 1):
                    cells.append(
                        ExperimentCell(
                            fault=None,
                            seed=seed,
                            model=model,
                            run=run,
                            evidence_fault=evidence_fault,
                        )
                    )
    return cells


def build_false_assurance_grid(
    seeds: list[int] = SEEDS,
    models: list[str] = ["flash"],
) -> list[ExperimentCell]:
    """Cross behavior faults with telemetry faults: the 2x2 design.

    Four paired conditions fall out of the product: clean/faulty
    behavior x clean/faulty telemetry. The dangerous cell is faulty
    behavior + faulty telemetry — a hidden violation that a naive
    monitor would miss entirely. Only one model variant and one run
    per cell: with the deterministic FakeProvider the seed is the
    statistical unit, and behavior does not vary across variants.
    """
    behaviors = [None, *(f.value for f in FaultKind)]
    telemetries = [None, *(f.value for f in EvidenceFaultKind)]
    cells: list[ExperimentCell] = []
    for behavior in behaviors:
        for telemetry in telemetries:
            for seed in seeds:
                for model in models:
                    cells.append(
                        ExperimentCell(
                            fault=behavior,
                            seed=seed,
                            model=model,
                            run=1,
                            evidence_fault=telemetry,
                        )
                    )
    return cells


def build_false_assurance_pilot(
    seeds: list[int] = [1, 2],
) -> list[ExperimentCell]:
    """A small live pilot over the four conditions: representative
    behavior faults x representative telemetry faults x paired seeds.
    Sized for real model calls with the judge enabled."""
    behaviors = [None, "policy_violation", "schema_violation"]
    telemetries = [None, "drop_tool_result", "truncate_final"]
    cells: list[ExperimentCell] = []
    for behavior in behaviors:
        for telemetry in telemetries:
            for seed in seeds:
                cells.append(
                    ExperimentCell(
                        fault=behavior,
                        seed=seed,
                        model="flash",
                        run=1,
                        evidence_fault=telemetry,
                    )
                )
    return cells


def checksum(cells: list[ExperimentCell]) -> str:
    """Grid identity: same cells, same sha256. Artifacts produced from
    a grid are labelled with this so results can be attributed."""
    return hashlib.sha256(
        json.dumps([asdict(c) for c in cells], sort_keys=True).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="build the experiment grid")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--evidence", action="store_true", help="build the clean-behavior telemetry-fault grid")
    parser.add_argument("--false-assurance", action="store_true", help="build the behavior x telemetry 2x2 grid")
    args = parser.parse_args()

    if args.false_assurance:
        cells = build_false_assurance_grid()
    elif args.evidence:
        cells = build_evidence_grid()
    else:
        cells = build_grid()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocolVersion": PROTOCOL_VERSION,
        "gridKind": "false_assurance" if args.false_assurance else "evidence" if args.evidence else "behavior",
        "checksum": checksum(cells),
        "cells": [asdict(c) for c in cells],
    }
    (args.out / "grid.json").write_text(json.dumps(manifest, indent=2))
    print(f"grid: {len(cells)} cells, protocol {PROTOCOL_VERSION}, sha256 {manifest['checksum'][:12]} -> {args.out / 'grid.json'}")


if __name__ == "__main__":
    main()
