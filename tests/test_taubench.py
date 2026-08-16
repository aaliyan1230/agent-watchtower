"""The tau-bench oracle: reference actions reproduce the ground-truth
state; any deviation from them changes the environment state and must
fail the reward check."""

import copy

from benchmarks.taubench import env
from benchmarks.taubench.types import Action, Task

TASKS = env.load_tasks(env.DATA_DIR.parent / "tasks_subset.json")


def test_reference_actions_reproduce_ground_truth_state():
    for task in TASKS:
        data = env.load_data()
        for action in task.actions:
            if action.name != env.RESPOND_ACTION_NAME:
                env.apply_action(data, action)
        result = env.reward(data, task)
        assert result["reward"] == 1.0, f"task {task.user_id}: {result}"


def test_mutated_action_fails_state_check():
    task = TASKS[0]
    data = env.load_data()
    for action in task.actions:
        if action.name == env.RESPOND_ACTION_NAME:
            continue
        mutated = Action(name=action.name, kwargs={**action.kwargs})
        if "item_ids" in mutated.kwargs:
            mutated.kwargs["item_ids"] = []  # return nothing: wrong state
        env.apply_action(data, mutated)
    result = env.reward(data, task)
    assert result["reward"] == 0.0
    assert result["actionsMatch"] is False


def test_fresh_data_has_stable_hash():
    assert env.data_hash(env.load_data()) == env.data_hash(env.load_data())


def test_reward_is_deterministic():
    task = TASKS[0]
    data = env.load_data()
    for action in task.actions:
        if action.name != env.RESPOND_ACTION_NAME:
            env.apply_action(data, action)
    a = env.reward(copy.deepcopy(data), task)
    b = env.reward(copy.deepcopy(data), task)
    assert a == b


def test_task_subset_is_loaded_and_well_formed():
    assert len(TASKS) == 12
    for task in TASKS:
        assert task.instruction
        assert any(a.name != env.RESPOND_ACTION_NAME for a in task.actions)
        for action in task.actions:
            assert action.name in env.TOOLS_BY_NAME or action.name == env.RESPOND_ACTION_NAME
