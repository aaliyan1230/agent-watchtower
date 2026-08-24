// Package report turns findings into a verdict and bundles the evidence:
// the exact span IDs, timestamps, and values behind every verdict — the
// "evidence over opinion" contract of the whole project. Verdicts are
// derived purely: critical violations fail, evidence gaps become
// inconclusive, warnings flag, and a clean complete trace passes.
package report

import (
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

// ProtocolVersion freezes the verdict semantics: verdict mapping,
// finding schema, budget fields. Bump it when any of those change.
// artifacts carry it so results from different protocol versions are
// never compared silently.
const ProtocolVersion = "0.11"

// Verdict is the run-level outcome an operator acts on.
type Verdict string

const (
	VerdictPass         Verdict = "PASS"
	VerdictFlagged      Verdict = "FLAGGED"
	VerdictInconclusive Verdict = "INCONCLUSIVE"
	VerdictFail         Verdict = "FAIL"
	VerdictUnknown      Verdict = "UNKNOWN"
)

// VerifierSummary is the per-verifier account the experiments layer
// reads for detection rates: how many findings each verifier emitted.
type VerifierSummary struct {
	Name        string          `json:"name"`
	Findings    int             `json:"findings"`
	MaxSeverity verify.Severity `json:"maxSeverity"`
}

// JudgeUsage is the measured token cost of the judge pass — the
// "judge is an expensive supplement" claim needs numbers.
type JudgeUsage struct {
	InputTokens  int64 `json:"inputTokens"`
	OutputTokens int64 `json:"outputTokens"`
}

// Report is the complete evidence bundle for one run.
type Report struct {
	ProtocolVersion string            `json:"protocolVersion"`
	TraceID         string            `json:"traceId"`
	RootSpanID      string            `json:"rootSpanId"`
	Verdict         Verdict           `json:"verdict"`
	Judged          bool              `json:"judged"`
	JudgeModel      string            `json:"judgeModel,omitempty"`
	JudgeUsage      JudgeUsage        `json:"judgeUsage,omitempty"`
	Findings        []verify.Finding  `json:"findings,omitempty"`
	Verifiers       []VerifierSummary `json:"verifiers"`
	Budget          graph.Budget      `json:"budget"`
	Evidence        graph.Evidence    `json:"evidence"`
	GeneratedAt     time.Time         `json:"generatedAt"`
}

// Build assembles a Report from a reconstructed run and its findings.
func Build(run *graph.Run, findings []verify.Finding, judged bool) *Report {
	r := &Report{
		ProtocolVersion: ProtocolVersion,
		TraceID:         run.TraceID,
		RootSpanID:      run.RootSpanID,
		Verdict:         verdictFor(findings),
		Judged:          judged,
		Findings:        findings,
		Budget:          run.Budget,
		Evidence:        run.Evidence,
		GeneratedAt:     time.Now().UTC(),
	}
	r.Verifiers = summarize(findings, judged)
	return r
}

// SetJudgeUsage records the measured judge cost for a judged run.
func (r *Report) SetJudgeUsage(input, output int64) {
	r.JudgeUsage = JudgeUsage{InputTokens: input, OutputTokens: output}
}

// verdictFor maps findings to a verdict. Ordering is deliberate and
// runs from strongest-to-weakest confidence: an observed critical
// violation fails unless the trace was never declared closed; an
// unclosed trace is UNKNOWN (spans may still arrive, so even a FAIL
// seen so far cannot be trusted); a closed-but-incomplete trace
// abstains INCONCLUSIVE; a judge critical finding fails when the
// channel supports it; warnings flag. Only a closed, complete, clean
// run passes — the paper's "no premature PASS" contract.
func verdictFor(findings []verify.Finding) Verdict {
	detCritical, judgeCritical := false, false
	warning, evidenceGap, unclosed := false, false, false
	for _, f := range findings {
		if f.Kind == verify.FindingEvidenceGap {
			if f.Claim == "trace_closed" {
				unclosed = true
			} else {
				evidenceGap = true
			}
			continue
		}
		if f.Severity == verify.SeverityCritical {
			if f.Verifier == "judge" || f.Source != "" {
				judgeCritical = true
			} else {
				detCritical = true
			}
			continue
		}
		warning = true
	}
	switch {
	case unclosed:
		return VerdictUnknown
	case detCritical:
		return VerdictFail
	case evidenceGap:
		return VerdictInconclusive
	case judgeCritical:
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
