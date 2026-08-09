"""Fault injection determinism + behavior tests."""

import json

import pytest

from harness.faults import FaultInjector, FaultKind, FaultSpec
from harness.providers import FakeProvider, ProviderResponse

BASE = ProviderResponse(
    content='{"id": 42, "title": "ok"}',
    tool_calls=[{"name": "search", "args": {"q": "x"}}],
    input_tokens=10,
    output_tokens=20,
)


def inject(spec: FaultSpec) -> ProviderResponse:
    p = FaultInjector(FakeProvider([BASE]), spec)
    return p.chat([{"role": "user", "content": "hi"}])


def test_malformed_json_truncates():
    out = inject(FaultSpec(FaultKind.MALFORMED_JSON, seed=1))
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.content)


def test_schema_violation_flips_id_to_string():
    out = inject(FaultSpec(FaultKind.SCHEMA_VIOLATION, seed=1))
    doc = json.loads(out.content)
    assert isinstance(doc["id"], str)


def test_policy_violation_injects_disallowed_tool():
    out = inject(FaultSpec(FaultKind.POLICY_VIOLATION, seed=1))
    assert out.tool_calls == [{"name": "rm", "args": {"path": "/tmp/watchtower"}, "id": "call-fault-0"}]


def test_loop_repeats_tool_three_times():
    out = inject(FaultSpec(FaultKind.LOOP, seed=1))
    assert len(out.tool_calls) == 3
    assert all(tc["name"] == "search" for tc in out.tool_calls)


def test_budget_blowout_inflates_tokens():
    out = inject(FaultSpec(FaultKind.BUDGET_BLOWOUT, seed=1))
    assert out.input_tokens == 1000 and out.output_tokens == 2000


def test_timeout_raises():
    with pytest.raises(TimeoutError):
        inject(FaultSpec(FaultKind.PROVIDER_TIMEOUT, seed=1))


def test_faults_are_deterministic_per_seed():
    a = inject(FaultSpec(FaultKind.MALFORMED_JSON, seed=7))
    b = inject(FaultSpec(FaultKind.MALFORMED_JSON, seed=7))
    assert a.content == b.content


def test_step_selection_skips_early_turns():
    p = FaultInjector(FakeProvider([BASE, BASE, BASE]), FaultSpec(FaultKind.BUDGET_BLOWOUT, seed=1, step=2))
    first = p.chat([])
    second = p.chat([])
    third = p.chat([])
    assert first.input_tokens == 10 and second.input_tokens == 10
    assert third.input_tokens == 1000


def test_no_spec_passthrough():
    out = inject(None)
    assert out.content == BASE.content
