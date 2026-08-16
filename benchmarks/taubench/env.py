"""The τ-bench retail environment slice: DB snapshot, tool execution,
and the benchmark-native reward oracle.

The reward logic is tau-bench's own (envs/base.py, MIT): apply the
task's reference actions to a fresh DB copy to get the ground-truth
state hash, then compare it with the state hash produced by the
agent's actions. Required output strings must also appear in the
agent's respond action. This oracle is environment truth — no LLM is
involved — which is what makes it usable as ground truth here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tools import ALL_TOOLS
from .types import Action, Task, consistent_hash, to_hashable

DATA_DIR = Path(__file__).parent / "data"
RESPOND_ACTION_NAME = "respond"

TOOLS_BY_NAME = {tool.get_info()["function"]["name"]: tool for tool in ALL_TOOLS}


def load_data() -> dict[str, Any]:
    return {
        "users": json.loads((DATA_DIR / "users.json").read_text()),
        "orders": json.loads((DATA_DIR / "orders.json").read_text()),
        "products": json.loads((DATA_DIR / "products.json").read_text()),
    }


def data_hash(data: dict[str, Any]) -> str:
    return consistent_hash(to_hashable(data))


def apply_action(data: dict[str, Any], action: Action) -> str:
    """Execute one tool action against the DB, returning the tool's
    observation string. Mutates data exactly like tau-bench does."""
    tool = TOOLS_BY_NAME[action.name]
    return tool.invoke(data, **action.kwargs)


def reward(data_after_agent: dict[str, Any], task: Task) -> dict[str, Any]:
    """tau-bench's calculate_reward, distilled: state-hash equality
    with the reference solution plus required-output coverage. Returns
    {reward: 0|1, actionsMatch: bool, outputsFound: [..], gtDataHash}."""
    reference = load_data()
    for action in task.actions:
        if action.name != RESPOND_ACTION_NAME:
            apply_action(reference, action)
    gt_hash = data_hash(reference)

    actions_match = data_hash(data_after_agent) == gt_hash

    outputs_found: list[tuple[str, bool]] = []
    for output in task.outputs:
        found = False
        for action in task.actions:
            if action.name == RESPOND_ACTION_NAME:
                content = action.kwargs.get("content", "").lower().replace(",", "")
                if output.lower() in content:
                    found = True
                    break
        outputs_found.append((output, found))

    score = 1.0 if actions_match and all(found for _, found in outputs_found) else 0.0
    return {
        "reward": score,
        "actionsMatch": actions_match,
        "outputsFound": dict(outputs_found),
        "gtDataHash": gt_hash,
    }


def load_tasks(path: Path, limit: int = 0) -> list[Task]:
    """Load the exported task subset. The subset was exported from
    tau-bench's retail tasks_train.py with the reference actions
    intact; each task is the benchmark's own ground-truth solution."""
    raw = json.loads(path.read_text())
    if limit:
        raw = raw[:limit]
    return [Task.from_dict(d) for d in raw]


def tool_schema(tool) -> dict[str, Any]:
    """The OpenAI-compatible function schema for one tool — the
    harness's tool calling format."""
    return tool.get_info()


def policy_tools(tool_names: list[str]) -> list[dict[str, Any]]:
    return [tool_schema(TOOLS_BY_NAME[name]) for name in tool_names if name in TOOLS_BY_NAME]
