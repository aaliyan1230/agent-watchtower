package model

// Semconv attribute names shared across the wire: the Python harness
// writes these, the Go verifiers read them. The gen_ai.* names follow the
// OTel GenAI semantic conventions (operation, system, model, token
// counts); tool/agent attributes align with the gen-ai agent semconv
// direction; watchtower.* are our own additions (structured-output
// contract checking).
//
// Keeping every attribute name in one place is what makes the pipeline
// machine-checkable: a typo here breaks the contract between harness and
// verifiers, and the compiler catches it on the Go side.
const (
	// GenAI operation (e.g. "chat"), system (provider), and model ids.
	GenAIOperationName  = "gen_ai.operation.name"
	GenAISystem         = "gen_ai.system"
	GenAIRequestModel   = "gen_ai.request.model"
	GenAIResponseModel  = "gen_ai.response.model"
	GenAIInputTokens    = "gen_ai.usage.input_tokens"
	GenAIOutputTokens   = "gen_ai.usage.output_tokens"
	GenAICompletionTime = "gen_ai.usage.completion_time" // ours: ms

	// Tool-call spans.
	ToolName      = "tool.name"
	ToolCallID    = "tool.call.id"
	ToolResultOK  = "tool.result.ok" // ours: "true"/"false"
	ToolResultMsg = "tool.result.message"

	// Agent spans: which agent owns this span, and what step index it is.
	AgentName = "agent.name"
	AgentID   = "agent.id"
	StepIndex = "agent.step" // ours: 0-based step number within the agent

	// Watchtower-specific: structured-output contract checking. The
	// harness stamps the contract name it is supposed to honor and the
	// raw model output; the schema verifier parses and checks it.
	WatchtowerContract = "watchtower.contract"
	WatchtowerOutput   = "watchtower.output"
)
