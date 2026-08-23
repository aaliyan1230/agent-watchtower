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
# the policy verifier, so a policy fault is always detected. The id
# must be stable: live providers reject tool results whose id does not
# match an assistant tool call.
DISALLOWED_TOOL = {
    "name": "rm",
    "args": {"path": "/tmp/watchtower"},
    "id": "call-fault-0",
}
LOOP_TOOL = {"name": "search", "args": {"q": "stuck-loop"}, "id": "call-fault-1"}
CONTRACT_ID_FIELD = "id"  # the schema faults corrupt this field


class FaultKind(Enum):
    MALFORMED_JSON = "malformed_json"  # output truncates mid-document
    SCHEMA_VIOLATION = "schema_violation"  # contract field gets the wrong type
    POLICY_VIOLATION = "policy_violation"  # tool call to a disallowed tool
    LOOP = "loop"  # the same tool call repeats 3x
    BUDGET_BLOWOUT = "budget_blowout"  # token counts inflated 10x
    PROVIDER_TIMEOUT = "provider_timeout"  # call raises


class EvidenceFaultKind(Enum):
    """Faults applied after the agent has produced its spans.

    These do not change agent behavior. They model what a collector or
    transport can do to the evidence that a supervisor sees.
    """

    DROP_PARENT = "drop_parent"  # remove a root span, leaving children orphaned
    DUPLICATE_SPAN = "duplicate_span"  # send one span twice
    REORDER_SPANS = (
        "reorder_spans"  # reverse a batch; reconstruction should tolerate it
    )
    DROP_CHILD = (
        "drop_child"  # remove a worker lifecycle span, leaving its children orphaned
    )
    DROP_TOOL_RESULT = "drop_tool_result"  # remove tool result spans while calls remain
    MISMATCH_TOOL_ID = "mismatch_tool_id"  # replace a result id with an unpaired id
    TRUNCATE_FINAL = "truncate_final"  # remove the final-answer evidence markers
    LATE_SPAN = "late_span"  # move a child outside its parent's time window
    DROP_ATTRIBUTE = (
        "drop_attribute"  # remove a required correlation attribute (gen_ai.system)
    )
    SAMPLE_SPANS = "sample_spans"  # deterministic sampling: drop every other span
    DROP_LINK = "drop_link"  # remove causal links, so no edges between siblings
    ORPHAN_LINK_TARGET = (
        "orphan_link_target"  # point a link at a span that is not in the batch
    )


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

    def chat(
        self, messages, tools=None, *, contract=None, temperature=0.0
    ) -> ProviderResponse:
        resp = self._provider.chat(
            messages, tools=tools, contract=contract, temperature=temperature
        )
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
        out.tool_calls = [
            {"name": "search", "args": {"q": "stuck-loop"}, "id": f"call-fault-{i}"}
            for i in range(3)
        ]
    elif kind is FaultKind.BUDGET_BLOWOUT:
        out.input_tokens *= 100
        out.output_tokens *= 100
    elif kind is FaultKind.PROVIDER_TIMEOUT:
        raise TimeoutError("provider timed out (injected fault)")
    _sync_assistant(out)
    return out


def _sync_assistant(resp: ProviderResponse) -> None:
    """Make the echoed assistant message match the corrupted tool
    calls: live providers require tool results to answer the exact
    assistant tool calls (same ids) that preceded them. Extra fields
    (Gemini's internal thought_signature) are preserved from the
    original entries, by id when possible, else positionally."""
    if not resp.tool_calls or not resp.assistant_message:
        return
    originals = resp.assistant_message.get("tool_calls", [])
    rewritten = []
    for i, tc in enumerate(resp.tool_calls):
        orig = next((o for o in originals if o.get("id") == tc.get("id")), None)
        if orig is None and i < len(originals):
            orig = originals[i]
        entry = {
            "id": tc.get("id", ""),
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc.get("args", {})),
            },
        }
        if orig and "extra_content" in orig:
            entry["extra_content"] = orig["extra_content"]
        rewritten.append(entry)
    resp.assistant_message = {**resp.assistant_message, "tool_calls": rewritten}


def _corrupt_contract(content: str) -> str:
    """Flip the contract's id field to a string: schema-valid JSON,
    semantically wrong — exactly what the schema verifier exists for.
    Fences are stripped first: a fenced answer would otherwise bounce
    the parse and skip the corruption."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        doc: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return content
    if isinstance(doc, dict) and CONTRACT_ID_FIELD in doc:
        doc[CONTRACT_ID_FIELD] = "not-an-int"
    return json.dumps(doc)
