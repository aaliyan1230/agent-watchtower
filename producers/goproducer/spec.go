package main

// TraceSpec describes one agent run before it becomes OTel spans.
// Keeping the description separate from emission makes the variant
// logic pure and table-testable: which required field a variant drops
// is exactly the claim the cross-producer test expects to abstain.

// Semconv attribute names: gen_ai.* are stable OTel GenAI semconv;
// tool.name / tool.call.id follow the gen-ai agent semconv direction;
// watchtower.* are the small documented contract additions.
const (
	genAIOperationName = "gen_ai.operation.name"
	genAISystem        = "gen_ai.system"
	genAIRequestModel  = "gen_ai.request.model"
	genAIResponseModel = "gen_ai.response.model"
	genAIInputTokens   = "gen_ai.usage.input_tokens"
	genAIOutputTokens  = "gen_ai.usage.output_tokens"
	toolName           = "tool.name"
	toolCallID         = "tool.call.id"
	toolResultOK       = "tool.result.ok"
	agentName          = "agent.name"
	agentStep          = "agent.step"
	wtCompleted        = "watchtower.completed"
	wtFinal            = "watchtower.final"
	wtOutput           = "watchtower.output"
	wtContract         = "watchtower.contract"
	wtToolCallIDs      = "watchtower.tool.call.ids"
	wtToolArgs         = "watchtower.tool.args"
)

type spanSpec struct {
	Name     string
	Attrs    map[string]any
	Events   []eventSpec
	Children []*spanSpec
}

type eventSpec struct {
	Name string
	Attr map[string]any
}

// buildSpec returns the run description for a variant. ok reports
// whether the variant is known; for known variants every span tree is
// structurally identical — only the required evidence differs.
func buildSpec(variant string) (*spanSpec, bool) {
	switch variant {
	case "clean", "omit-genai-system", "omit-tool-result", "omit-tool-id", "omit-final":
		return buildRun(variant), true
	default:
		return nil, false
	}
}

func buildRun(variant string) *spanSpec {
	answer := `{"id": 1, "title": "broken login", "labels": ["auth"]}`
	root := &spanSpec{
		Name:  "agent.run",
		Attrs: map[string]any{agentName: "worker-go", wtCompleted: true},
		Events: []eventSpec{{
			Name: "supervisor.decision",
			Attr: map[string]any{"action": "accept"},
		}},
	}
	if variant != "omit-final" {
		root.Attrs[wtFinal] = true
		root.Attrs[wtOutput] = answer
	}

	toolTurn := &spanSpec{
		Name: "chat",
		Attrs: map[string]any{
			genAIOperationName: "chat",
			genAIRequestModel:  "gofake-worker",
			genAIResponseModel: "gofake-worker",
			genAIInputTokens:   30,
			genAIOutputTokens:  10,
			agentName:          "worker-go",
			agentStep:          1,
			wtToolCallIDs:      `["call-1"]`,
		},
	}
	if variant != "omit-genai-system" {
		toolTurn.Attrs[genAISystem] = "gofake"
	}
	toolCall := &spanSpec{
		Name: "tool.call",
		Attrs: map[string]any{
			toolName:   "search",
			agentName:  "worker-go",
			agentStep:  1,
			wtToolArgs: `{"q":"login"}`,
		},
	}
	switch variant {
	case "omit-tool-id":
		// tool.call.id stays absent; tool_pairing must abstain.
	default:
		toolCall.Attrs[toolCallID] = "call-1"
	}
	if variant != "omit-tool-result" {
		toolCall.Attrs[toolResultOK] = true
	}

	finalTurn := &spanSpec{
		Name: "chat",
		Attrs: map[string]any{
			genAIOperationName: "chat",
			genAIRequestModel:  "gofake-worker",
			genAIResponseModel: "gofake-worker",
			genAIInputTokens:   20,
			genAIOutputTokens:  15,
			agentName:          "worker-go",
			agentStep:          2,
		},
	}
	if variant != "omit-genai-system" {
		finalTurn.Attrs[genAISystem] = "gofake"
	}
	if variant != "omit-final" {
		finalTurn.Attrs[wtFinal] = true
		finalTurn.Attrs[wtOutput] = answer
		finalTurn.Attrs[wtContract] = "ticket"
	}

	root.Children = []*spanSpec{toolTurn, finalTurn}
	toolTurn.Children = []*spanSpec{toolCall}
	return root
}
