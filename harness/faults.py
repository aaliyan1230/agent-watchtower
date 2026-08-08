"""Deterministic fault injection.

Faults corrupt *provider responses* — the model layer — because that is
where real agent failures originate: malformed JSON, policy-violating
tool calls, loops, budget blowouts, timeouts. Each fault is a seeded
pure transformation of a ProviderResponse, so the same seed produces the
same corrupted run: the experiment's reproducibility contract.

FaultInjector wraps a provider and corrupts the response at a chosen
step index (None = the first response, which keeps demos simple).
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .providers import Provider, ProviderResponse

# Tools the faults inject. Deliberately NOT in the allowlist used by
# the policy verifier, so a policy fault is always detected.
DISALLOWED_TOOL = {"name": "rm", "args": {"path": "/tmp/watchtower"}}
LOOP_TOOL = {"name": "search", "args": {"q": "stuck-loop"}}
CONTRACT_ID_FIELD = "id"  # the schema faults corrupt this field


class FaultKind(Enum):
    MALFORMED_JSON = "malformed_json"  # output truncates mid-document
    SCHEMA_VIOLATION = "schema_violation"  # contract field gets the wrong type
    POLICY_VIOLATION = "policy_violation"  # tool call to a disallowed tool
    LOOP = "loop"  # the same tool call repeats 3x
    BUDGET_BLOWOUT = "budget_blowout"  # token counts inflated 10x
    PROVIDER_TIMEOUT = "provider_timeout"  # call raises


@dataclass
class FaultSpec:
    kind: FaultKind
    seed: int = 0
    step: int = 0  # 0-based response index to corrupt; None = all

    @property
    def rng(self) -> random.Random:
        # A fresh RNG per spec keeps corruption independent of how many
        # times the spec is applied elsewhere.
        return random.Random(self.seed)


class FaultInjector:
    """Wraps a provider and corrupts the response at the fault's step.
    Deterministic: same spec, same corrupted response, always."""

    def __init__(self, provider: Provider, spec: FaultSpec | None):
        self._provider = provider
        self._spec = spec
        self._turns = 0

    @property
    def name(self) -> str:
        return self._provider.name

    def chat(self, messages, tools=None, *, contract=None, temperature=0.0) -> ProviderResponse:
        resp = self._provider.chat(messages, tools=tools, contract=contract, temperature=temperature)
        turn = self._turns
        self._turns += 1
        if self._spec is None:
            return resp
        if self._spec.step is not None and turn != self._spec.step:
            return resp
        return _corrupt(resp, self._spec)


def _corrupt(resp: ProviderResponse, spec: FaultSpec) -> ProviderResponse:
    kind = spec.kind
    out = copy.deepcopy(resp)
    if kind is FaultKind.MALFORMED_JSON:
        # Truncate mid-document: the classic structured-output failure.
        out.content = out.content[: len(out.content) // 2] + "{"
    elif kind is FaultKind.SCHEMA_VIOLATION:
        out.content = _corrupt_contract(out.content)
    elif kind is FaultKind.POLICY_VIOLATION:
        out.tool_calls = [DISALLOWED_TOOL]
    elif kind is FaultKind.LOOP:
        out.tool_calls = [LOOP_TOOL, LOOP_TOOL, LOOP_TOOL]
    elif kind is FaultKind.BUDGET_BLOWOUT:
        out.input_tokens *= 10
        out.output_tokens *= 10
    elif kind is FaultKind.PROVIDER_TIMEOUT:
        raise TimeoutError("provider timed out (injected fault)")
    return out


def _corrupt_contract(content: str) -> str:
    """Flip the contract's id field to a string: schema-valid JSON,
    semantically wrong — exactly what the schema verifier exists for."""
    try:
        doc: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(doc, dict) and CONTRACT_ID_FIELD in doc:
        doc[CONTRACT_ID_FIELD] = "not-an-int"
    return json.dumps(doc)
