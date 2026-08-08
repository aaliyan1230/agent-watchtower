// Package graph reconstructs an agent run from a flat list of spans.
// On the wire a trace is an unordered set of records; a "run" is a
// structure: a tree (parentSpanId), an ordered sequence of steps
// (startTime), each classified by its OTel GenAI semconv attributes.
// Everything downstream — verifiers, reports, the judge — consumes the
// reconstructed Run, never raw spans.
package graph

import (
	"fmt"
	"sort"
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

// StepKind classifies a span by what the agent actually did.
type StepKind string

const (
	StepAgent StepKind = "agent" // agent lifecycle (a sub-run, a supervisor decision)
	StepLLM   StepKind = "llm"   // an LLM call (gen_ai.operation.name present)
	StepTool  StepKind = "tool"  // a tool call (tool.name present)
)

// Step is one ordered unit of an agent run, derived from a single span.
// Tool-specific fields are populated only for StepTool steps; model and
// token fields only for StepLLM steps.
type Step struct {
	Order        int // 1-based position in the run
	SpanID       string
	Agent        string // agent.name
	Kind         StepKind
	Name         string
	Model        string // gen_ai.request.model
	System       string // gen_ai.system
	Tool         string // tool.name
	ToolCallID   string // tool.call.id
	ToolOK       bool   // tool.result.ok
	ToolArgs     string // watchtower.tool.args (JSON string)
	Contract     string // watchtower.contract
	Output       string // watchtower.output (raw model output)
	InputTokens  int64
	OutputTokens int64
	StartTime    time.Time
	EndTime      time.Time
}

// Budget summarizes resource consumption for the budget verifier and
// the paper's overhead numbers.
type Budget struct {
	SpanCount     int
	LLMCallCount  int
	ToolCallCount int
	InputTokens   int64
	OutputTokens  int64
	TotalTokens   int64
	StartTime     time.Time
	EndTime       time.Time
	DurationMs    int64
}

// Run is the reconstructed agent run: one trace, ordered steps, budget.
type Run struct {
	TraceID    string
	RootSpanID string
	Steps      []Step
	Budget     Budget
}

// ToolCalls returns just the tool steps, in run order — the loop
// verifier's input.
func (r *Run) ToolCalls() []Step {
	var out []Step
	for _, s := range r.Steps {
		if s.Kind == StepTool {
			out = append(out, s)
		}
	}
	return out
}

// Reconstruct builds a Run from spans. It rejects empty input, spans
// from multiple traces (one run = one trace), and spans that fail
// validation. Orphaned spans (parent missing from the set) are treated
// as roots rather than errors: a partial trace should still verify.
func Reconstruct(spans []model.Span) (*Run, error) {
	if len(spans) == 0 {
		return nil, fmt.Errorf("reconstruct: no spans")
	}
	traceID := spans[0].TraceID
	for i := range spans {
		if spans[i].TraceID != traceID {
			return nil, fmt.Errorf("reconstruct: mixed traces %q and %q", traceID, spans[i].TraceID)
		}
		if err := spans[i].Validate(); err != nil {
			return nil, fmt.Errorf("reconstruct: %w", err)
		}
	}

	byID := make(map[string]*model.Span, len(spans))
	children := make(map[string][]*model.Span)
	for i := range spans {
		s := &spans[i]
		byID[s.SpanID] = s
		if s.ParentID != "" {
			children[s.ParentID] = append(children[s.ParentID], s)
		}
	}

	// Sibling ordering is by start time; a stable sort keeps ties in
	// input order, which preserves ingestion order for same-timestamp
	// spans (the harness emits steps with distinct timestamps).
	for _, siblings := range children {
		sort.SliceStable(siblings, func(i, j int) bool {
			return siblings[i].StartTime.Before(siblings[j].StartTime)
		})
	}

	var roots []*model.Span
	for i := range spans {
		s := &spans[i]
		if s.ParentID == "" || byID[s.ParentID] == nil {
			roots = append(roots, s)
		}
	}
	sort.SliceStable(roots, func(i, j int) bool {
		return roots[i].StartTime.Before(roots[j].StartTime)
	})

	run := &Run{TraceID: traceID, RootSpanID: roots[0].SpanID}
	var order int
	var walk func(*model.Span)
	walk = func(s *model.Span) {
		order++
		run.Steps = append(run.Steps, stepFrom(s, order))
		for _, c := range children[s.SpanID] {
			walk(c)
		}
	}
	for _, root := range roots {
		walk(root)
	}

	run.Budget = summarize(run.Steps, roots)
	return run, nil
}

// stepFrom classifies a span and copies the attributes verifiers care
// about. Classification priority is tool > llm > agent: a span stamped
// with both tool.name and gen_ai attributes is a tool call.
func stepFrom(s *model.Span, order int) Step {
	st := Step{
		Order:     order,
		SpanID:    s.SpanID,
		Name:      s.Name,
		StartTime: s.StartTime,
		EndTime:   s.EndTime,
	}
	st.Agent, _ = s.Attr(model.AgentName)
	st.Contract, _ = s.Attr(model.WatchtowerContract)
	st.Output, _ = s.Attr(model.WatchtowerOutput)

	if tool, ok := s.Attr(model.ToolName); ok {
		st.Kind = StepTool
		st.Tool = tool
		st.ToolCallID, _ = s.Attr(model.ToolCallID)
		st.ToolOK, _ = s.AttrBool(model.ToolResultOK)
		st.ToolArgs, _ = s.Attr(model.WatchtowerToolArgs)
		return st
	}
	if op, ok := s.Attr(model.GenAIOperationName); ok && op != "" {
		st.Kind = StepLLM
		st.Model, _ = s.Attr(model.GenAIRequestModel)
		st.System, _ = s.Attr(model.GenAISystem)
		st.InputTokens, _ = s.AttrInt(model.GenAIInputTokens)
		st.OutputTokens, _ = s.AttrInt(model.GenAIOutputTokens)
		return st
	}
	st.Kind = StepAgent
	return st
}

// summarize aggregates the budget over steps: span count, LLM/tool
// counts, token totals, and wall-clock duration across roots.
func summarize(steps []Step, roots []*model.Span) Budget {
	b := Budget{SpanCount: len(steps)}
	var start, end time.Time
	for _, s := range steps {
		if s.Kind == StepLLM {
			b.LLMCallCount++
			b.InputTokens += s.InputTokens
			b.OutputTokens += s.OutputTokens
		}
		if s.Kind == StepTool {
			b.ToolCallCount++
		}
		if start.IsZero() || s.StartTime.Before(start) {
			start = s.StartTime
		}
		if s.EndTime.After(end) {
			end = s.EndTime
		}
	}
	b.StartTime, b.EndTime = start, end
	b.TotalTokens = b.InputTokens + b.OutputTokens
	b.DurationMs = end.Sub(start).Milliseconds()
	return b
}
