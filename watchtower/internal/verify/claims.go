package verify

import "github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"

// Provenance records where an evidence requirement comes from on the
// wire. The split is the paper's interoperability claim: a producer
// using only the stable OTel GenAI semconv gets some evidence for
// free, while the watchtower bucket is the small, documented set of
// attributes a producer must add. Findings carry this vocabulary so a
// verdict can say which side of the boundary a missing field came from.
type Provenance string

const (
	ProvenanceOTelCore   Provenance = "otel_core"   // standard span structure: name, parents, timestamps, events, status
	ProvenanceOTelGenAI  Provenance = "otel_genai"  // stable OTel GenAI semconv: gen_ai.*
	ProvenanceGenAIAgent Provenance = "genai_agent" // tool/agent markers aligned with the gen-ai agent semconv direction
	ProvenanceWatchtower Provenance = "watchtower"  // our contract additions: watchtower.*, tool.result.*, agent.step
)

// EvidenceReq names one required trace fact and where it must come
// from. Attribute requirements use the attribute name; structural
// requirements use a small token vocabulary (span.<kind>, event.<name>,
// parent.time_window) so the same Claim shape covers both.
type EvidenceReq struct {
	What       string     `json:"what"`
	Provenance Provenance `json:"provenance"`
}

// Claim is one runtime claim, the minimum evidence needed to support
// it, and the recovery action an operator should take when the
// evidence is absent or inconsistent. Claims are evidence checks, not
// behavior checks: a missing requirement makes the verdict abstain,
// never fail.
type Claim struct {
	Name      string        `json:"name"`
	Statement string        `json:"statement"`
	Requires  []EvidenceReq `json:"requires"`
	Action    string        `json:"action"`
}

// ClaimReference returns every claim the evidence contract defines,
// including disabled ones. It is the machine-checkable vocabulary of
// the contract: what each claim asserts, which wire facts it needs,
// and where those facts come from.
func ClaimReference() []Claim {
	return []Claim{
		{
			Name:      "run_root",
			Statement: "the trace contains a root agent.run span with no parent",
			Requires: []EvidenceReq{
				{What: "span.agent.run", Provenance: ProvenanceOTelCore},
				{What: "parent.none", Provenance: ProvenanceOTelCore},
			},
			Action: "recollect the run from the producer",
		},
		{
			Name:      "agent_completion",
			Statement: "every agent span carries a valid completion marker",
			Requires: []EvidenceReq{
				{What: model.WatchtowerCompleted, Provenance: ProvenanceWatchtower},
			},
			Action: "recollect the run or extend producer instrumentation",
		},
		{
			Name:      "model_correlation",
			Statement: "every LLM span carries system, request model, response model, and a span name",
			Requires: []EvidenceReq{
				{What: "span.name", Provenance: ProvenanceOTelCore},
				{What: model.GenAISystem, Provenance: ProvenanceOTelGenAI},
				{What: model.GenAIRequestModel, Provenance: ProvenanceOTelGenAI},
				{What: model.GenAIResponseModel, Provenance: ProvenanceOTelGenAI},
			},
			Action: "extend producer instrumentation to emit gen_ai.* correlation fields",
		},
		{
			Name:      "tool_pairing",
			Statement: "every tool span carries name, call id, and a result marker, and every declared tool call has a paired tool span",
			Requires: []EvidenceReq{
				{What: model.ToolName, Provenance: ProvenanceGenAIAgent},
				{What: model.ToolCallID, Provenance: ProvenanceGenAIAgent},
				{What: model.ToolResultOK, Provenance: ProvenanceWatchtower},
			},
			Action: "recollect the run with tool result spans",
		},
		{
			Name:      "final_answer",
			Statement: "the run contains a final answer span with non-empty output",
			Requires: []EvidenceReq{
				{What: model.WatchtowerFinal, Provenance: ProvenanceWatchtower},
				{What: model.WatchtowerOutput, Provenance: ProvenanceWatchtower},
			},
			Action: "recollect the run or extend producer instrumentation",
		},
		{
			Name:      "supervisor_decision",
			Statement: "the root span carries a terminal supervisor decision event",
			Requires: []EvidenceReq{
				{What: "event.supervisor.decision", Provenance: ProvenanceOTelCore},
			},
			Action: "recollect the run or extend producer instrumentation",
		},
		{
			Name:      "temporal_nesting",
			Statement: "every child span is temporally nested inside its parent",
			Requires: []EvidenceReq{
				{What: "parent.time_window", Provenance: ProvenanceOTelCore},
			},
			Action: "recollect the run with wall-clock timestamps",
		},
		{
			Name:      "trace_integrity",
			Statement: "span ids are unique and every parent reference resolves",
			Requires: []EvidenceReq{
				{What: "span.id.unique", Provenance: ProvenanceOTelCore},
				{What: "parent.resolves", Provenance: ProvenanceOTelCore},
			},
			Action: "replay from a clean recorded trace",
		},
		{
			Name:      "trace_closed",
			Statement: "the run carries an explicit end-of-trace marker, so no causally relevant spans may still arrive",
			Requires: []EvidenceReq{
				{What: model.WatchtowerTraceClosed, Provenance: ProvenanceWatchtower},
			},
			Action: "wait for the full export batch or recall the remaining spans",
		},
	}
}

// Claims maps the configured obligations to their claim definitions,
// in a fixed order. Only enabled obligations are returned.
func (o EvidenceObligations) Claims() []Claim {
	enabled := map[string]bool{
		"run_root":            o.RequireRoot,
		"agent_completion":    o.RequireAgentCompletion,
		"model_correlation":   o.RequireModelCorrelation,
		"tool_pairing":        o.RequireToolResults,
		"final_answer":        o.RequireFinalAnswer,
		"supervisor_decision": o.RequireSupervisorDecision,
		"temporal_nesting":    o.RequireTemporalNesting,
		"trace_closed":        o.RequireTraceClosed,
	}
	// trace_integrity rides on the structural findings and is always
	// part of the evidence contract when evidence checking is on.
	enabled["trace_integrity"] = true
	var out []Claim
	for _, claim := range ClaimReference() {
		if enabled[claim.Name] {
			out = append(out, claim)
		}
	}
	return out
}

// attributeProvenance is the wire-provenance registry for every
// attribute the contract uses. A test walks ClaimReference() against
// this registry, so an attribute mentioned by a claim without a
// provenance entry — or a provenance that disagrees with the semconv
// constants — fails the build's test run, not a paper draft.
func attributeProvenance() map[string]Provenance {
	return map[string]Provenance{
		model.GenAIOperationName:    ProvenanceOTelGenAI,
		model.GenAISystem:           ProvenanceOTelGenAI,
		model.GenAIRequestModel:     ProvenanceOTelGenAI,
		model.GenAIResponseModel:    ProvenanceOTelGenAI,
		model.GenAIInputTokens:      ProvenanceOTelGenAI,
		model.GenAIOutputTokens:     ProvenanceOTelGenAI,
		model.GenAICompletionTime:   ProvenanceOTelGenAI,
		model.ToolName:              ProvenanceGenAIAgent,
		model.ToolCallID:            ProvenanceGenAIAgent,
		model.ToolResultOK:          ProvenanceWatchtower,
		model.ToolResultMsg:         ProvenanceWatchtower,
		model.AgentName:             ProvenanceGenAIAgent,
		model.AgentID:               ProvenanceGenAIAgent,
		model.StepIndex:             ProvenanceWatchtower,
		model.WatchtowerCompleted:   ProvenanceWatchtower,
		model.WatchtowerFinal:       ProvenanceWatchtower,
		model.WatchtowerOutcome:     ProvenanceWatchtower,
		model.WatchtowerToolCallIDs: ProvenanceWatchtower,
		model.WatchtowerTraceClosed: ProvenanceWatchtower,
		model.WatchtowerContract:    ProvenanceWatchtower,
		model.WatchtowerOutput:      ProvenanceWatchtower,
		model.WatchtowerToolArgs:    ProvenanceWatchtower,
	}
}
