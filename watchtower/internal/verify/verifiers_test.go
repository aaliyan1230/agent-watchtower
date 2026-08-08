package verify

import (
	"strings"
	"testing"
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

func t(n int) time.Time { return time.Unix(int64(100+n), 0).UTC() }

func toolStep(id, tool, args string) graph.Step {
	return graph.Step{
		Order: 1, SpanID: id, Kind: graph.StepTool,
		Tool: tool, ToolArgs: args, StartTime: t(1),
	}
}

func TestCheckPolicy(t *testing.T) {
	policy := Policy{AllowedTools: []string{"search", "read"}}
	tests := []struct {
		name     string
		steps    []graph.Step
		want     int
		wantFrag string
	}{
		{name: "all allowed", steps: []graph.Step{toolStep("s1", "search", `{"q":"x"}`), toolStep("s2", "read", `{"f":"y"}`)}},
		{name: "disallowed tool", steps: []graph.Step{toolStep("s1", "write", `{"f":"y"}`)}, want: 1, wantFrag: `"write"`},
		{name: "mixed", steps: []graph.Step{toolStep("s1", "search", "{}"), toolStep("s2", "rm", `{"p":"/"}`)}, want: 1, wantFrag: "rm"},
		{name: "no policy configured", steps: []graph.Step{toolStep("s1", "anything", "{}")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := policy
			if tt.name == "no policy configured" {
				p = Policy{}
			}
			f := CheckPolicy(&graph.Run{Steps: tt.steps}, p)
			if len(f) != tt.want {
				t.Fatalf("findings = %d, want %d (%+v)", len(f), tt.want, f)
			}
			if tt.want > 0 && !strings.Contains(f[0].Message, tt.wantFrag) {
				t.Errorf("message %q missing %q", f[0].Message, tt.wantFrag)
			}
			if tt.want > 0 && len(f[0].SpanIDs) != 1 {
				t.Errorf("finding has %d span ids, want 1", len(f[0].SpanIDs))
			}
		})
	}
}

func TestCheckLoop(t *testing.T) {
	tests := []struct {
		name     string
		steps    []graph.Step
		want     int
		wantFrag string
	}{
		{name: "distinct calls", steps: []graph.Step{
			toolStep("s1", "search", `{"q":"a"}`), toolStep("s2", "search", `{"q":"b"}`), toolStep("s3", "read", `{"f":"y"}`),
		}},
		{name: "twice is tolerated", steps: []graph.Step{
			toolStep("s1", "search", `{"q":"a"}`), toolStep("s2", "search", `{"q":"a"}`),
		}},
		{name: "thrice identical is a loop", steps: []graph.Step{
			toolStep("s1", "search", `{"q":"a"}`), toolStep("s2", "search", `{"q":"a"}`), toolStep("s3", "search", `{"q":"a"}`),
		}, want: 1, wantFrag: "3 times"},
		{name: "same tool different args ok", steps: []graph.Step{
			toolStep("s1", "search", `{"q":"a"}`), toolStep("s2", "search", `{"q":"b"}`), toolStep("s3", "search", `{"q":"c"}`),
		}},
		{name: "abab oscillation", steps: []graph.Step{
			toolStep("s1", "search", `{"q":"a"}`), toolStep("s2", "read", `{"f":"y"}`),
			toolStep("s3", "search", `{"q":"a"}`), toolStep("s4", "read", `{"f":"y"}`),
		}, want: 1, wantFrag: "oscillating"},
		{name: "abcabc not oscillation (no alternating window)", steps: []graph.Step{
			toolStep("s1", "a", "1"), toolStep("s2", "b", "2"), toolStep("s3", "c", "3"),
			toolStep("s4", "a", "1"), toolStep("s5", "b", "2"), toolStep("s6", "c", "3"),
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := CheckLoop(&graph.Run{Steps: tt.steps}, LoopConfig{})
			if len(f) != tt.want {
				t.Fatalf("findings = %d, want %d (%+v)", len(f), tt.want, f)
			}
			if tt.want > 0 {
				if f[0].Verifier != "loop" || f[0].Severity != SeverityWarning {
					t.Errorf("finding = %+v, want loop/warning", f[0])
				}
				if !strings.Contains(f[0].Message, tt.wantFrag) {
					t.Errorf("message %q missing %q", f[0].Message, tt.wantFrag)
				}
				if len(f[0].SpanIDs) < 3 {
					t.Errorf("finding evidence covers %d spans, want >= 3", len(f[0].SpanIDs))
				}
			}
		})
	}
}

func TestCheckBudget(t *testing.T) {
	tests := []struct {
		name     string
		limits   Limits
		budget   graph.Budget
		want     int
		wantFrag string
	}{
		{name: "within limits", limits: Limits{MaxSteps: 10, MaxTotalTokens: 1000, MaxDurationMs: 60_000},
			budget: graph.Budget{SpanCount: 5, TotalTokens: 200, DurationMs: 1000}},
		{name: "too many steps", limits: Limits{MaxSteps: 4}, budget: graph.Budget{SpanCount: 5}, want: 1, wantFrag: "step limit"},
		{name: "token blowout", limits: Limits{MaxTotalTokens: 100}, budget: graph.Budget{TotalTokens: 101}, want: 1, wantFrag: "token budget"},
		{name: "too slow", limits: Limits{MaxDurationMs: 100}, budget: graph.Budget{DurationMs: 101}, want: 1, wantFrag: "duration"},
		{name: "everything exceeded", limits: Limits{MaxSteps: 1, MaxTotalTokens: 1, MaxDurationMs: 1},
			budget: graph.Budget{SpanCount: 5, TotalTokens: 200, DurationMs: 1000}, want: 3},
		{name: "no limits configured", limits: Limits{}, budget: graph.Budget{SpanCount: 9999, TotalTokens: 9999, DurationMs: 9999}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := CheckBudget(&graph.Run{Budget: tt.budget}, tt.limits)
			if len(f) != tt.want {
				t.Fatalf("findings = %d, want %d (%+v)", len(f), tt.want, f)
			}
			if tt.want > 0 {
				if f[0].Verifier != "budget" || f[0].Severity != SeverityCritical {
					t.Errorf("finding = %+v, want budget/critical", f[0])
				}
				if !strings.Contains(f[0].Message, tt.wantFrag) {
					t.Errorf("message %q missing %q", f[0].Message, tt.wantFrag)
				}
			}
		})
	}
}
