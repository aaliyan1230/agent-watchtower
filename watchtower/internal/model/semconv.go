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

	// Names from the current OTel GenAI agent conventions. Producers
	// that follow the spec emit these instead of the shorter tool.* /
	// agent.* / gen_ai.system names above and below, so the graph reads
	// either spelling; missing one would hide a tool call from every
	// behavior check.
	GenAIProviderName      = "gen_ai.provider.name"
	GenAIToolName          = "gen_ai.tool.name"
	GenAIToolCallID        = "gen_ai.tool.call.id"
	GenAIToolCallArguments = "gen_ai.tool.call.arguments"
	GenAIAgentName         = "gen_ai.agent.name"
	GenAIAgentID           = "gen_ai.agent.id"

	// gen_ai.operation.name values that are not model calls. The spec
	// also allows invoke_{framework_entrypoint}, so agent operations are
	// matched by the OpInvokePrefix rather than a closed list.
	OpExecuteTool  = "execute_tool"
	OpCreateAgent  = "create_agent"
	OpInvokePrefix = "invoke_"

	// Tool-call spans.
	ToolName      = "tool.name"
	ToolCallID    = "tool.call.id"
	ToolResultOK  = "tool.result.ok" // ours: "true"/"false"
	ToolResultMsg = "tool.result.message"

	// Agent spans: which agent owns this span, and what step index it is.
	AgentName = "agent.name"
	AgentID   = "agent.id"
	StepIndex = "agent.step" // ours: 0-based step number within the agent

	// Lifecycle and correlation markers emitted by the harness. These are
	// deliberately small obligations: they make absence observable without
	// pretending that a trace can prove every semantic claim.
	WatchtowerCompleted   = "watchtower.completed"
	WatchtowerFinal       = "watchtower.final"
	WatchtowerOutcome     = "watchtower.outcome"
	WatchtowerToolCallIDs = "watchtower.tool.call.ids"
	WatchtowerTraceClosed = "watchtower.trace.closed" // ours: "true" once the run is confirmed complete

	// Watchtower-specific: structured-output contract checking. The
	// harness stamps the contract name it is supposed to honor and the
	// raw model output; the schema verifier parses and checks it.
	WatchtowerContract = "watchtower.contract"
	WatchtowerOutput   = "watchtower.output"

	// Link purpose: the causal role a link edge plays in a concurrent
	// run. Producers stamp this on link attributes so the canonicalizer
	// can classify edges (e.g. data dependency vs. supervisor signal).
	LinkPurpose = "watchtower.link.purpose" // ours: "data" | "control"
)
