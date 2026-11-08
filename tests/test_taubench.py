"""The tau-bench oracle: reference actions reproduce the ground-truth
state; any deviation from them changes the environment state and must
fail the reward check."""

import copy

from benchmarks.taubench import env
from benchmarks.taubench.types import Action, Task

TASKS = env.load_tasks(env.DATA_DIR.parent / "tasks_subset.json")
TEST_TASKS = env.load_tasks(env.TEST_TASKS_PATH)


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


def test_test_split_reference_solutions_reach_full_reward():
    """Each of the 115 test tasks must be solvable in the vendored
    environment, required outputs included when the agent says them."""
    assert len(TEST_TASKS) == 115
    for task in TEST_TASKS:
        data = env.load_data()
        for action in task.actions:
            assert action.name in env.TOOLS_BY_NAME, action.name
            env.apply_action(data, action)
        result = env.reward(data, task, responses=[" ".join(task.outputs)])
        assert result["reward"] == 1.0, f"task {task.user_id}: {result}"


def test_required_outputs_come_from_the_agent():
    task = next(t for t in TEST_TASKS if t.outputs)
    data = env.load_data()
    for action in task.actions:
        env.apply_action(data, action)
    silent = env.reward(data, task)
    assert silent["actionsMatch"] and silent["reward"] == 0.0
    spoken = env.reward(data, task, responses=[f"The answer is {', '.join(task.outputs)}."])
    assert spoken["reward"] == 1.0


def test_replay_follows_the_decision_when_there_is_one():
    from experiments.taubench import wants_replay

    gap = {"kind": "evidence_gap", "verifier": "evidence"}
    assert wants_replay({"findings": [gap]})
    assert not wants_replay({"findings": [{"verifier": "loop"}]})
    assert not wants_replay(None)
    assert wants_replay({"findings": [gap], "decision": {"action": "replay"}})
    assert not wants_replay({"findings": [gap], "decision": {"action": "escalate"}})
