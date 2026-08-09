package verify

import (
	"fmt"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// LoopConfig tunes loop detection; Enabled makes the check opt-in like
// every other verifier (zero config must verify nothing). Thresholds
// are conservative: repeated *identical* calls (same tool, same
// argument hash) and two-step oscillations (A,B,A,B) are the
// cheap-to-spot signatures of an agent stuck in a loop.
type LoopConfig struct {
	Enabled        bool `json:"enabled"`
	MaxRepetitions int  `json:"maxRepetitions"` // > this many identical calls = loop (default 2)
	MinCycleLength int  `json:"minCycleLength"` // oscillation window (default 4)
}

func (c LoopConfig) withDefaults() LoopConfig {
	if c.MaxRepetitions <= 0 {
		c.MaxRepetitions = 2
	}
	if c.MinCycleLength <= 0 {
		c.MinCycleLength = 4
	}
	return c
}

// CheckLoop flags repeated tool+args sequences and oscillations. The
// message includes the sequence position so evidence points at the
// exact spans.
func CheckLoop(run *graph.Run, cfg LoopConfig) []Finding {
	cfg = cfg.withDefaults()

	type call struct {
		tool string
		args string
	}
	var seq []call
	for _, step := range run.ToolCalls() {
		seq = append(seq, call{step.Tool, step.ToolArgs})
	}
	if len(seq) < 2 {
		return nil
	}

	var out []Finding

	counts := make(map[call]int)
	for _, c := range seq {
		counts[c]++
	}
	for c, n := range counts {
		if n <= cfg.MaxRepetitions {
			continue
		}
		var spans []string
		var times []string
		for _, step := range run.ToolCalls() {
			if step.Tool == c.tool && step.ToolArgs == c.args {
				spans = append(spans, step.SpanID)
				times = append(times, step.StartTime.UTC().Format("2006-01-02T15:04:05Z"))
			}
		}
		out = append(out, Finding{
			Verifier: "loop", Severity: SeverityWarning,
			Message:    fmt.Sprintf("tool %q called %d times with identical arguments", c.tool, n),
			SpanIDs:    spans,
			Timestamps: times,
			Value:      c.args,
		})
	}

	for i := 0; i+cfg.MinCycleLength <= len(seq); i++ {
		if seq[i] == seq[i+2] && seq[i+1] == seq[i+3] && seq[i] != seq[i+1] {
			var spans []string
			for _, step := range run.ToolCalls()[i : i+cfg.MinCycleLength] {
				spans = append(spans, step.SpanID)
			}
			out = append(out, Finding{
				Verifier: "loop", Severity: SeverityWarning,
				Message: fmt.Sprintf("oscillating between %q and %q", seq[i].tool, seq[i+1].tool),
				SpanIDs: spans,
			})
			break // one oscillation finding is enough
		}
	}
	return out
}
