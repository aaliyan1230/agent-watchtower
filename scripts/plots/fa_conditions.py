"""Render the four-condition false-assurance figure from frozen artifacts.

Reads the contract and baseline result files and draws stacked verdict
composition per condition, one bar per configuration. Output is a vector
PDF sized for a NeurIPS text column. Deterministic: same artifacts in,
same figure out.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = [
    ("clean x clean", False, False),
    ("clean x faulty telem.", False, True),
    ("faulty x clean telem.", True, False),
    ("faulty x faulty telem.", True, True),
]
SEGMENTS = ["PASS", "FAIL / FLAGGED", "INCONCLUSIVE"]
COLORS = ["#4c9f70", "#b0433f", "#8a8fb4"]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def condition_key(cell: dict) -> tuple[bool, bool]:
    behavior = cell.get("fault") is not None
    telemetry = bool(cell.get("evidence_fault")) and cell["evidence_fault"] != "reorder_spans"
    return behavior, telemetry


def composition(cells: list[dict]) -> dict[tuple[bool, bool], tuple[float, float, float]]:
    groups: dict[tuple[bool, bool], Counter] = {}
    for cell in cells:
        groups.setdefault(condition_key(cell), Counter())[cell["verdict"]] += 1
    out: dict[tuple[bool, bool], tuple[float, float, float]] = {}
    for key, counts in groups.items():
        total = sum(counts.values())
        fail = counts.get("FAIL", 0) + counts.get("FLAGGED", 0)
        out[key] = (
            counts.get("PASS", 0) / total,
            fail / total,
            counts.get("INCONCLUSIVE", 0) / total,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    contract = composition(load(args.contract)["cells"])
    baseline = composition(load(args.baseline)["cells"])

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    width = 0.38
    xs = range(len(CONDITIONS))
    for offset, (config, comp, hatch) in enumerate(
        [(-width / 2 - 0.01, contract, ""), (width / 2 + 0.01, baseline, "//")]
    ):
        bottoms = [0.0] * len(CONDITIONS)
        for seg_idx, segment in enumerate(SEGMENTS):
            values = [comp[(b, t)][seg_idx] for _, b, t in CONDITIONS]
            ax.bar(
                [x + offset for x in xs],
                values,
                width=width,
                bottom=bottoms,
                color=COLORS[seg_idx],
                hatch=hatch if seg_idx < 2 else "",
                edgecolor="white",
                linewidth=0.5,
                label=segment if not offset else None,
            )
            for x, value, bottom in zip(xs, values, bottoms):
                if value >= 0.08:
                    ax.text(
                        x + offset, bottom + value / 2,
                        f"{value:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if seg_idx == 1 else "black",
                    )
            bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for label, *_ in CONDITIONS], fontsize=8)
    ax.set_ylabel("fraction of runs", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
