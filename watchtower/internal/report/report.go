// Package report turns findings into a verdict and bundles the evidence:
// the exact span IDs, timestamps, and values behind every verdict — the
// "evidence over opinion" contract of the whole project. Verdicts are
// derived purely: any critical finding fails the run, any warning flags
// it, nothing passes it.
package report

import (
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

// Verdict is the run-level outcome an operator acts on.
type Verdict string

const (
	VerdictPass    Verdict = "PASS"
	VerdictFlagged Verdict = "FLAGGED"
	VerdictFail    Verdict = "FAIL"
)

// VerifierSummary is the per-verifier account the experiments layer
// reads for detection rates: how many findings each verifier emitted.
type VerifierSummary struct {
	Name        string          `json:"name"`
	Findings    int             `json:"findings"`
	MaxSeverity verify.Severity `json:"maxSeverity"`
}

// Report is the complete evidence bundle for one run.
type Report struct {
	TraceID     string            `json:"traceId"`
	RootSpanID  string            `json:"rootSpanId"`
	Verdict     Verdict           `json:"verdict"`
	Judged      bool              `json:"judged"`
	Findings    []verify.Finding  `json:"findings,omitempty"`
	Verifiers   []VerifierSummary `json:"verifiers"`
	Budget      graph.Budget      `json:"budget"`
	GeneratedAt time.Time         `json:"generatedAt"`
}

// Build assembles a Report from a reconstructed run and its findings.
func Build(run *graph.Run, findings []verify.Finding, judged bool) *Report {
	r := &Report{
		TraceID:     run.TraceID,
		RootSpanID:  run.RootSpanID,
		Verdict:     verdictFor(findings),
		Judged:      judged,
		Findings:    findings,
		Budget:      run.Budget,
		GeneratedAt: time.Now().UTC(),
	}
	r.Verifiers = summarize(findings, judged)
	return r
}

// verdictFor maps findings to a verdict: critical -> FAIL,
// warning -> FLAGGED, else PASS.
func verdictFor(findings []verify.Finding) Verdict {
	critical, warning := false, false
	for _, f := range findings {
		switch f.Severity {
		case verify.SeverityCritical:
			critical = true
		case verify.SeverityWarning:
			warning = true
		}
	}
	switch {
	case critical:
		return VerdictFail
	case warning:
		return VerdictFlagged
	default:
		return VerdictPass
	}
}

// summarize aggregates findings per verifier, including the judge slot
// (so the experiments can compute judge-only detection rates), and
// reports each deterministic verifier even when it found nothing.
func summarize(findings []verify.Finding, judged bool) []VerifierSummary {
	counts := make(map[string]int)
	severity := make(map[string]verify.Severity)
	for _, f := range findings {
		counts[f.Verifier]++
		if f.Severity > severity[f.Verifier] {
			severity[f.Verifier] = f.Severity
		}
	}
	out := make([]VerifierSummary, 0, len(verify.VerifierNames)+1)
	for _, name := range verify.VerifierNames {
		out = append(out, VerifierSummary{Name: name, Findings: counts[name], MaxSeverity: severity[name]})
	}
	if judged {
		out = append(out, VerifierSummary{Name: "judge", Findings: counts["judge"]})
	}
	return out
}
