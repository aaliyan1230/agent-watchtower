"""Live smoke test: one real Gemini run through the full pipeline.

Requires GEMINI_API_KEY in .env (fails loudly otherwise). Spawns the
Go server with the LLM judge enabled, runs a worker that must use a
tool to answer, and prints the verdict plus judge findings.

Run: .venv/bin/python -m harness.live_smoke
"""

from __future__ import annotations

from harness.env import get_api_key
from harness.providers import GeminiProvider
from harness.server import REPO_ROOT, spawn_server
from harness.supervisor import Supervisor
from harness.telemetry import HarnessTelemetry
from harness.workers import Tool, Worker

ENDPOINT = "http://127.0.0.1:4318"
ADDR = "127.0.0.1:4318"
CONFIG = str(REPO_ROOT / "watchtower" / "testdata" / "judge_config.json")


def main() -> None:
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing: add it to .env (see .env.example)")
    print("gemini key present; spawning watchtower serve with judge enabled")

    proc = spawn_server(ENDPOINT, ADDR, CONFIG, env={"GEMINI_API_KEY": api_key})
    try:
        telemetry = HarnessTelemetry(ENDPOINT, service_name="live-smoke")
        provider = GeminiProvider(model="gemini-3.5-flash-lite", api_key=api_key)
        add_tool = Tool("add", "add two numbers", {"a": {"type": "integer"}, "b": {"type": "integer"}}, lambda a, b: a + b)
        worker = Worker(
            "smoke-worker",
            "You are a calculator. Always use the add tool before answering.",
            provider,
            [add_tool],
            telemetry.tracer(),
            max_steps=3,
        )
        result = Supervisor("smoke-supervisor", telemetry.tracer()).run("What is 23 + 19? Use the add tool.", [worker])
        if not telemetry.flush():
            raise SystemExit("ingest rejected the trace")
        report = telemetry.fetch_report(result.trace_id)
        if report is None:
            raise SystemExit("no report; server rejected the run")

        print(f"worker answer: {result.answers['smoke-worker']!r}")
        print(f"verdict: {report['verdict']}")
        budget = report["budget"]
        print(f"budget: {budget['spanCount']} spans, {budget['totalTokens']} tokens")
        print(f"judged: {report['judged']}")
        for f in report.get("findings", []):
            print(f"  [{f['severity']}] {f['verifier']} ({f.get('source', '-')}): {f['message']}")
        telemetry.shutdown()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
