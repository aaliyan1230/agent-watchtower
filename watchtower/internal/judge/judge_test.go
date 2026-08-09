package judge

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

func runnable(t *testing.T, modelOutput string) (*LLM, *graph.Run) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":` + modelOutput + `}}]}`))
	}))
	t.Cleanup(srv.Close)

	run := &graph.Run{
		TraceID: "t1",
		Steps: []graph.Step{
			{Order: 1, Kind: graph.StepLLM, Agent: "a", Model: "flash", InputTokens: 10},
			{Order: 2, Kind: graph.StepTool, Agent: "a", Tool: "search", ToolOK: false},
			{Order: 3, Kind: graph.StepLLM, Agent: "a", Model: "flash"},
		},
		Budget: graph.Budget{SpanCount: 3, TotalTokens: 40, DurationMs: 100},
	}
	return New("gemini-2.5-flash", "key", srv.URL), run
}

func TestRunMapsIssuesToFindings(t *testing.T) {
	j, run := runnable(t, `"{\"issues\":[{\"severity\":\"critical\",\"description\":\"step 2 tool failed\"},{\"severity\":\"warning\",\"description\":\"repeats itself\"}]}"`)
	f, err := j.Run(run)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(f) != 2 {
		t.Fatalf("findings = %d, want 2", len(f))
	}
	if f[0].Verifier != "judge" || f[0].Severity != verify.SeverityCritical || f[0].Source != "gemini-2.5-flash" {
		t.Errorf("finding 0 = %+v", f[0])
	}
	if f[1].Severity != verify.SeverityWarning {
		t.Errorf("finding 1 = %+v", f[1])
	}
}

func TestRunNoIssues(t *testing.T) {
	j, run := runnable(t, `"{\"issues\":[]}"`)
	f, err := j.Run(run)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(f) != 0 {
		t.Fatalf("findings = %+v, want none", f)
	}
}

func TestRunServerErrorSurfaces(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	}))
	defer srv.Close()
	j := New("m", "k", srv.URL)
	if _, err := j.Run(&graph.Run{}); err == nil || !strings.Contains(err.Error(), "500") {
		t.Fatalf("err = %v, want 500", err)
	}
}

func TestSummarizeRunMentionsEvidence(t *testing.T) {
	j, run := runnable(t, `"{}"`)
	s := summarizeRun(run)
	for _, want := range []string{"trace t1", "tool agent=a tool=search ok=false", "budget steps=3 tokens=40"} {
		if !strings.Contains(s, want) {
			t.Errorf("summary missing %q:\n%s", want, s)
		}
	}
	_ = j
}

func TestStripFences(t *testing.T) {
	tests := []struct{ in, want string }{
		{in: "```json\n{\"a\": 1}\n```", want: "{\"a\": 1}"},
		{in: "{\"a\": 1}", want: "{\"a\": 1}"},
		{in: "```\n{\"a\": 1}\n```", want: "{\"a\": 1}"},
		{in: "```json\n{\"a\": 1}", want: "{\"a\": 1}"},
	}
	for _, tt := range tests {
		if got := stripFences(tt.in); got != tt.want {
			t.Errorf("stripFences(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
