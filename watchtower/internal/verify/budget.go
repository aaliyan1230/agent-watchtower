package verify

import (
	"fmt"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// Limits are run-level budgets. Zero value = limit not enforced, so a
// fresh config checks nothing until the operator opts in — matching the
// "no forced dependencies" philosophy of the repo.
type Limits struct {
	MaxSteps       int   `json:"maxSteps"`       // total steps (spans) in the run
	MaxTotalTokens int64 `json:"maxTotalTokens"` // input + output tokens
	MaxDurationMs  int64 `json:"maxDurationMs"`  // wall clock of the run
}

// CheckBudget flags runs that exceed any configured limit, reporting
// actual vs limit — the evidence is the numbers themselves.
func CheckBudget(run *graph.Run, l Limits) []Finding {
	var out []Finding
	b := run.Budget

	if l.MaxSteps > 0 && b.SpanCount > l.MaxSteps {
		out = append(out, Finding{
			Verifier: "budget", Severity: SeverityCritical,
			Message: fmt.Sprintf("step limit exceeded: %d > %d", b.SpanCount, l.MaxSteps),
			Value:   fmt.Sprintf("%d", b.SpanCount),
		})
	}
	if l.MaxTotalTokens > 0 && b.TotalTokens > l.MaxTotalTokens {
		out = append(out, Finding{
			Verifier: "budget", Severity: SeverityCritical,
			Message: fmt.Sprintf("token budget exceeded: %d > %d", b.TotalTokens, l.MaxTotalTokens),
			Value:   fmt.Sprintf("%d", b.TotalTokens),
		})
	}
	if l.MaxDurationMs > 0 && b.DurationMs > l.MaxDurationMs {
		out = append(out, Finding{
			Verifier: "budget", Severity: SeverityCritical,
			Message: fmt.Sprintf("duration budget exceeded: %dms > %dms", b.DurationMs, l.MaxDurationMs),
			Value:   fmt.Sprintf("%dms", b.DurationMs),
		})
	}
	return out
}
