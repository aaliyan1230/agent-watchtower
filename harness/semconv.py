"""Semconv attribute names — the vocabulary both sides of the wire use.

These mirror the Go constants in watchtower/internal/model/semconv.go.
The wire contract is these strings; the Go side compiles against them,
so the Python side must treat them as frozen. A typo here silently
disables a verifier, so keep this file in lockstep with the Go one.
"""

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

TOOL_NAME = "tool.name"
TOOL_CALL_ID = "tool.call.id"
TOOL_RESULT_OK = "tool.result.ok"
TOOL_RESULT_MSG = "tool.result.message"
WATCHTOWER_TOOL_ARGS = "watchtower.tool.args"

AGENT_NAME = "agent.name"
AGENT_ID = "agent.id"
STEP_INDEX = "agent.step"
WATCHTOWER_COMPLETED = "watchtower.completed"
WATCHTOWER_FINAL = "watchtower.final"
WATCHTOWER_OUTCOME = "watchtower.outcome"
WATCHTOWER_TOOL_CALL_IDS = "watchtower.tool.call.ids"

WATCHTOWER_CONTRACT = "watchtower.contract"
WATCHTOWER_OUTPUT = "watchtower.output"
