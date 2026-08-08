package verify

import (
	"fmt"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// Policy is the tool allowlist: the permission boundary an agent's tool
// calls must stay inside. An empty allowlist means "no policy
// configured" and disables the check — an honest default for a tool
// that is still learning which tools exist.
type Policy struct {
	AllowedTools []string `json:"allowedTools"`
}

// CheckPolicy flags every tool call outside the allowlist.
func CheckPolicy(run *graph.Run, p Policy) []Finding {
	if len(p.AllowedTools) == 0 {
		return nil
	}
	allowed := make(map[string]bool, len(p.AllowedTools))
	for _, t := range p.AllowedTools {
		allowed[t] = true
	}
	var out []Finding
	for _, step := range run.ToolCalls() {
		if allowed[step.Tool] {
			continue
		}
		out = append(out, Finding{
			Verifier: "policy", Severity: SeverityCritical,
			Message:    fmt.Sprintf("tool %q is not on the allowlist", step.Tool),
			SpanIDs:    []string{step.SpanID},
			Timestamps: []string{step.StartTime.UTC().Format("2006-01-02T15:04:05Z")},
			Value:      step.Tool,
		})
	}
	return out
}
