package graph

import (
	"strings"
	"testing"
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

// syntheticTrace builds a canonical run: supervisor root -> three LLM
// steps, two of them issuing tool calls, with distinct start times so
// ordering is deterministic.
func syntheticTrace() []model.Span {
	base := time.Unix(100, 0).UTC()
	mk := func(id, parent, name string, t int, attrs map[string]string) model.Span {
		s := model.Span{
			TraceID: "trace-1", SpanID: id, ParentID: parent,
			Name: name, StartTime: base.Add(time.Duration(t) * time.Second),
			EndTime: base.Add(time.Duration(t+1) * time.Second),
			Status:  model.StatusOK, Attributes: attrs,
		}
		return s
	}
	return []model.Span{
		mk("s0", "", "agent.run", 0, map[string]string{model.AgentName: "supervisor"}),
		mk("s1", "s0", "chat", 1, map[string]string{
			model.AgentName: "worker-a", model.GenAIOperationName: "chat",
			model.GenAIRequestModel: "flash", model.GenAIInputTokens: "10",
			model.GenAIOutputTokens: "20",
		}),
		mk("s2", "s1", "tool.call", 2, map[string]string{
			model.AgentName: "worker-a", model.ToolName: "search",
			model.ToolCallID: "c1", model.ToolResultOK: "true",
			model.WatchtowerToolArgs: `{"q":"watchtower"}`,
		}),
		mk("s3", "s0", "chat", 3, map[string]string{
			model.AgentName: "worker-a", model.GenAIOperationName: "chat",
			model.GenAIRequestModel: "flash", model.GenAIInputTokens: "5",
			model.GenAIOutputTokens: "5",
		}),
		mk("s4", "s3", "tool.call", 4, map[string]string{
			model.AgentName: "worker-a", model.ToolName: "search",
			model.ToolCallID: "c2", model.ToolResultOK: "true",
			model.WatchtowerToolArgs: `{"q":"watchtower"}`,
		}),
		mk("s5", "s0", "chat", 5, map[string]string{
			model.AgentName: "worker-a", model.GenAIOperationName: "chat",
			model.GenAIRequestModel: "flash", model.GenAIInputTokens: "3",
			model.GenAIOutputTokens: "7",
		}),
	}
}

func TestReconstructHappyPath(t *testing.T) {
	run, err := Reconstruct(syntheticTrace())
	if err != nil {
		t.Fatalf("Reconstruct: %v", err)
	}
	if run.TraceID != "trace-1" || run.RootSpanID != "s0" {
		t.Fatalf("run identity = %q/%q", run.TraceID, run.RootSpanID)
	}
	if len(run.Steps) != 6 {
		t.Fatalf("steps = %d, want 6", len(run.Steps))
	}
	wantKinds := []StepKind{StepAgent, StepLLM, StepTool, StepLLM, StepTool, StepLLM}
	wantOrder := []int{1, 2, 3, 4, 5, 6}
	for i, st := range run.Steps {
		if st.Kind != wantKinds[i] {
			t.Errorf("step %d kind = %s, want %s", i, st.Kind, wantKinds[i])
		}
		if st.Order != wantOrder[i] {
			t.Errorf("step %d order = %d, want %d", i, st.Order, wantOrder[i])
		}
	}
	if st := run.Steps[1]; st.Model != "flash" || st.InputTokens != 10 || st.OutputTokens != 20 {
		t.Errorf("llm step = %+v", st)
	}
	if st := run.Steps[2]; st.Tool != "search" || !st.ToolOK || st.ToolArgs != `{"q":"watchtower"}` {
		t.Errorf("tool step = %+v", st)
	}

	b := run.Budget
	if b.SpanCount != 6 || b.LLMCallCount != 3 || b.ToolCallCount != 2 {
		t.Errorf("counts = %+v", b)
	}
	if b.InputTokens != 18 || b.OutputTokens != 32 || b.TotalTokens != 50 {
		t.Errorf("tokens = in %d out %d total %d", b.InputTokens, b.OutputTokens, b.TotalTokens)
	}
	if b.DurationMs != 6000 {
		t.Errorf("duration = %d ms, want 6000", b.DurationMs)
	}

	tc := run.ToolCalls()
	if len(tc) != 2 || tc[0].ToolCallID != "c1" || tc[1].ToolCallID != "c2" {
		t.Errorf("ToolCalls = %+v", tc)
	}
}

func TestReconstructErrors(t *testing.T) {
	base := syntheticTrace()
	tests := []struct {
		name    string
		spans   func() []model.Span
		wantErr string
	}{
		{
			name:    "empty input",
			spans:   func() []model.Span { return nil },
			wantErr: "no spans",
		},
		{
			name: "mixed traces",
			spans: func() []model.Span {
				sp := base[0]
				sp.TraceID = "trace-2"
				return []model.Span{base[0], sp}
			},
			wantErr: "mixed traces",
		},
		{
			name: "invalid span",
			spans: func() []model.Span {
				sp := base[0]
				sp.EndTime = time.Time{}
				return []model.Span{sp}
			},
			wantErr: "endTime",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Reconstruct(tt.spans())
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Reconstruct() = %v, want error containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestReconstructOrphanBecomesRoot(t *testing.T) {
	spans := syntheticTrace()
	for i := range spans {
		if spans[i].ParentID == "s0" {
			spans[i].ParentID = "missing"
		}
	}
	run, err := Reconstruct(spans)
	if err != nil {
		t.Fatalf("Reconstruct: %v", err)
	}
	if len(run.Steps) != 6 {
		t.Fatalf("steps = %d, want 6 (orphans kept as roots)", len(run.Steps))
	}
	if run.RootSpanID != "s0" {
		t.Fatalf("RootSpanID = %s, want s0 (earliest root)", run.RootSpanID)
	}
}

func TestReconstructSiblingOrderByTime(t *testing.T) {
	spans := syntheticTrace()
	// Swap the start times of s3 and s5 so the input order no longer
	// matches run order; reconstruction must sort by time.
	for i := range spans {
		if spans[i].SpanID == "s3" {
			spans[i].StartTime, spans[i].EndTime = spans[i].StartTime.Add(4*time.Second), spans[i].EndTime.Add(4*time.Second)
		}
	}
	run, err := Reconstruct(spans)
	if err != nil {
		t.Fatalf("Reconstruct: %v", err)
	}
	got := []string{}
	for _, st := range run.Steps {
		got = append(got, st.SpanID)
	}
	want := []string{"s0", "s1", "s2", "s5", "s3", "s4"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order = %v, want %v", got, want)
		}
	}
}
