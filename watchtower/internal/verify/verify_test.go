package verify

import (
	"errors"
	"testing"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

type stubJudge struct {
	name string
	err  error
}

func (s stubJudge) Name() string { return s.name }
func (s stubJudge) Run(*graph.Run) ([]Finding, error) {
	if s.err != nil {
		return nil, s.err
	}
	return []Finding{{Verifier: s.name, Severity: SeverityWarning, Message: "judge disagrees"}}, nil
}
func (s stubJudge) Usage() (int64, int64) { return 10, 20 }

// contains reports whether a string contains a substring — the test
// suite's tiny helper so table expectations stay readable.
func contains(s, frag string) bool {
	return len(frag) == 0 || (len(frag) <= len(s) && index(s, frag) >= 0)
}

func index(s, frag string) int {
	for i := 0; i+len(frag) <= len(s); i++ {
		if s[i:i+len(frag)] == frag {
			return i
		}
	}
	return -1
}

func TestVerifyAggregate(t *testing.T) {
	run := &graph.Run{Steps: []graph.Step{
		llmStep("ticket", `{"id": "not-an-int"}`), // schema: type mismatch
		toolStep("s2", "evil_tool", "{}"),         // policy: not allowed
		toolStep("s3", "search", `{"q":"a"}`),     // loop seeds
		toolStep("s4", "search", `{"q":"a"}`),
		toolStep("s5", "search", `{"q":"a"}`),
	}, Budget: graph.Budget{SpanCount: 50, TotalTokens: 10_000}}

	cfg := Config{
		Contracts: []Contract{{Name: "ticket", Fields: map[string]FieldSpec{"id": {Required: true, Type: TypeInt}}}},
		Policy:    Policy{AllowedTools: []string{"search"}},
		Loop:      LoopConfig{Enabled: true},
		Limits:    Limits{MaxSteps: 10, MaxTotalTokens: 5_000},
		Judge:     stubJudge{name: "stub"},
	}
	f := Verify(run, cfg)
	if len(f) != 6 {
		t.Fatalf("findings = %d, want 6 (schema, policy, loop, budget x2, judge) — got %+v", len(f), f)
	}
	wantOrder := []string{"schema", "policy", "loop", "budget", "budget", "stub"}
	for i, v := range wantOrder {
		if f[i].Verifier != v {
			t.Errorf("finding %d verifier = %s, want %s", i, f[i].Verifier, v)
		}
	}
}

func TestVerifyJudgeFailureSurfacesAsFinding(t *testing.T) {
	run := &graph.Run{Steps: []graph.Step{llmStep("ticket", `{"id": 1}`)}}
	cfg := Config{
		Contracts: []Contract{{Name: "ticket", Fields: map[string]FieldSpec{"id": {Required: true, Type: TypeInt}}}},
		Judge:     stubJudge{name: "stub", err: errors.New("boom")},
	}
	f := Verify(run, cfg)
	if len(f) != 1 {
		t.Fatalf("findings = %d, want 1 (judge failure surfaces as a finding)", len(f))
	}
	if f[0].Verifier != "stub" || f[0].Severity != SeverityWarning {
		t.Errorf("judge finding = %+v, want stub/warning", f[0])
	}
}

func TestVerifyNoJudge(t *testing.T) {
	run := &graph.Run{Steps: []graph.Step{llmStep("ticket", `{"id": 1}`)}}
	cfg := Config{
		Contracts: []Contract{{Name: "ticket", Fields: map[string]FieldSpec{"id": {Required: true, Type: TypeInt}}}},
	}
	if f := Verify(run, cfg); len(f) != 0 {
		t.Fatalf("findings = %+v, want none", f)
	}
}
