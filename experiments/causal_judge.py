"""CausalTrace live judge over presentation renderings.

The deterministic sweep (experiments.causal) proves the *canonical*
form is order-invariant. The live judge measures the other half of the
claim: whether a real LLM judge family flips its verdict across the
three comparator renderings of the *same trace* — ordinary text
(arrival order), timestamp-sorted, and canonical DAG. That is the
"verifier order-sensitivity" number the CausalTrace plan owes the paper.

Three judge families, chosen to be independent:
  - gemini  -> harness.providers.GeminiProvider (OpenAI-compatible)
  - bedrock -> experiments.bedrock.Client signing `deepseek.v3.2`
  - kimi    -> the same Bedrock transport signing `moonshotai.kimi-k2.5`

Offline CI stays green: every real client is injected in tests via a
mock HTTP transport, and FakeCausalJudge exercises the driver without
any network. Live runs gate on the .env keys exactly like the existing
experiment live targets.

Usage: python -m experiments.causal_judge --results artifacts/results-causal.json
                                          --judge gemini|bedrock|kimi [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, Sequence

from harness.providers import GeminiProvider

from .bedrock import Client as BedrockClient
from .bedrock import Credentials
from .causal_run import CAUSAL_PROTOCOL
from .run import aws_env
from .suite import PROTOCOL_VERSION

RENDERINGS = ("text", "time_sorted", "canonical")
LIVE_MODEL = "gemini-3.5-flash-lite"

BEDROCK_MODEL = "deepseek.v3.2"
KIMI_MODEL = "moonshotai.kimi-k2.5"

SYSTEM = (
    "You are an evidence reviewer for a causal trace verifier. You are shown ONE "
    "serialization rendering of a multi-agent execution log. Decide whether the log "
    "shows a complete, causally consistent run.\n"
    'Reply with exactly one JSON object with a single key "verdict" whose value is one '
    'of "PASS", "FAIL", "INCONCLUSIVE".\n'
    "PASS: the log is complete and internally consistent.\n"
    "FAIL: a concrete violation is visible (for example a causal dependency the "
    "ordering makes impossible, or a required step is missing).\n"
    "INCONCLUSIVE: the log appears incomplete or ambiguous (for example the final "
    "result span is absent) — do not guess."
)
USER_TEMPLATE = "Execution log:\n{render}\n\nWhat is your verdict?"


def parse_verdict(text: str) -> str:
    """Pull PASS / FAIL / INCONCLUSIVE out of a judge response. Empty when
    the model did not commit to any of the three verdicts."""
    if not text:
        return ""
    m = re.search(r"\b(PASS|FAIL|INCONCLUSIVE)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _messages(render: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(render=render)},
    ]


class CausalJudge(Protocol):
    name: str
    model: str

    def judge(self, render: str) -> tuple[str, dict]:
        """Verdict for one rendering, plus measured usage (inputTokens /
        outputTokens / durationMs)."""
        ...


class FakeCausalJudge:
    """Scripted judge for tests and dry runs: replays a fixed script,
    records every rendering it was asked to judge."""

    name = "fake"

    def __init__(self, script=(), model="fake-judge"):
        self._script = list(script)
        self.model = model
        self.calls: list[str] = []

    def judge(self, render: str) -> tuple[str, dict]:
        self.calls.append(render)
        if self._script:
            verdict, usage = self._script.pop(0)
        else:
            verdict, usage = (
                "INCONCLUSIVE",
                {
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "durationMs": 0,
                },
            )
        return verdict, usage


class GeminiCausalJudge:
    name = "gemini"

    def __init__(self, provider: GeminiProvider):
        self._provider = provider
        self.model = provider.model

    def judge(self, render: str) -> tuple[str, dict]:
        start = time.monotonic()
        resp = self._provider.chat(_messages(render), contract="verdict")
        usage = {
            "inputTokens": resp.input_tokens,
            "outputTokens": resp.output_tokens,
            "durationMs": round((time.monotonic() - start) * 1000, 1),
        }
        return parse_verdict(resp.content), usage


class BedrockCausalJudge:
    name = "bedrock"

    def __init__(self, client: BedrockClient):
        self._client = client
        self.model = client.model

    def judge(self, render: str) -> tuple[str, dict]:
        start = time.monotonic()
        text, usage = self._client.chat(SYSTEM, USER_TEMPLATE.format(render=render))
        usage["durationMs"] = round((time.monotonic() - start) * 1000, 1)
        return parse_verdict(text), usage


def make_judge(
    name: str,
    model: str | None = None,
    base_url: str | None = None,
    http=None,
) -> CausalJudge:
    """Build a live judge for one family. Keys come from the environment
    (.env or the AWS CLI session via experiments.run.aws_env) and missing
    keys fail loudly. ``base_url`` / ``http`` let tests point a client at a
    mock server / transport without any network."""
    if name == "gemini":
        provider = GeminiProvider(
            model=model or LIVE_MODEL,
            api_key=None,  # None -> read from .env
            base_url=base_url,
            client=http,
        )
        return GeminiCausalJudge(provider)
    if name in ("bedrock", "kimi"):
        env = aws_env()
        creds = Credentials(
            env.get("AWS_ACCESS_KEY_ID", ""),
            env.get("AWS_SECRET_ACCESS_KEY", ""),
            env.get("AWS_SESSION_TOKEN", ""),
        )
        if not creds.access_key or not creds.secret_key:
            raise ValueError(
                f"{name} causal judge needs AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY in .env or the AWS CLI"
            )
        region = env.get("AWS_REGION") or "us-east-1"
        model = model or (BEDROCK_MODEL if name == "bedrock" else KIMI_MODEL)
        client = BedrockClient(model, region, creds, base_url=base_url, http=http)
        judge: CausalJudge = BedrockCausalJudge(client)
        judge.name = name
        return judge
    raise ValueError(f"unknown judge {name!r}; expected gemini|bedrock|kimi")


def run_judge(
    cells: Sequence[dict],
    judge: CausalJudge,
    render_keys: Sequence[str] = RENDERINGS,
    limit: int = 0,
) -> list[dict]:
    """Judge the three renderings of each cell. Cells whose artifact carries
    no renderings (older grids) are skipped."""
    rows = list(cells)
    if limit:
        rows = rows[:limit]
    out: list[dict] = []
    for raw in rows:
        cell = asdict(raw) if not isinstance(raw, dict) else raw
        renders = cell.get("renderings") or {}
        per: dict[str, str] = {}
        usage: dict[str, dict] = {}
        for key in render_keys:
            render = renders.get(key)
            if not render:
                continue
            verdict, u = judge.judge(render)
            per[key] = verdict
            usage[key] = u
        out.append(
            {
                "traceId": cell.get("traceId") or cell.get("trace_id") or "",
                "evidenceFault": cell.get("evidenceFault"),
                "checksum": cell.get("checksum", ""),
                "renderings": per,
                "usage": usage,
            }
        )
    return out


def _artifact_checksum(cells: list[dict]) -> str:
    return hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="judge causal renderings with one LLM family (live)"
    )
    parser.add_argument(
        "--results", type=Path, default=Path("artifacts/results-causal.json")
    )
    parser.add_argument(
        "--judge", choices=["gemini", "bedrock", "kimi"], default="gemini"
    )
    parser.add_argument(
        "--model", type=str, default="", help="override the family default"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/results-causal-judge-gemini.json")
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="judge only the first N cells"
    )
    parser.add_argument(
        "--renderings",
        type=str,
        default=",".join(RENDERINGS),
        help="comma-separated rendering keys to judge",
    )
    args = parser.parse_args()

    artifact = json.loads(args.results.read_text())
    cells = artifact.get("cells", [])
    render_keys = [k.strip() for k in args.renderings.split(",") if k.strip()]
    if not cells:
        raise SystemExit(
            f"no cells in {args.results} — run `make experiment-causal` first"
        )

    judge = make_judge(args.judge, model=args.model or None)
    judged = run_judge(cells, judge, render_keys=render_keys, limit=args.limit)
    payload = {
        "protocolVersion": PROTOCOL_VERSION,
        "causalProtocol": CAUSAL_PROTOCOL,
        "judge": judge.name,
        "model": judge.model,
        "renderings": list(render_keys),
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sourceChecksum": artifact.get("checksum", ""),
        "cells": judged,
        "checksum": _artifact_checksum(judged),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    calls = sum(len(c.get("renderings") or {}) for c in judged)
    print(
        f"causal judge ({judge.name}/{judge.model}): "
        f"{len(judged)} cells, {calls} judge calls -> {args.out}"
    )


if __name__ == "__main__":
    main()
