package report

import (
	"encoding/json"
	"testing"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

func TestVerdictFor(t *testing.T) {
	tests := []struct {
		name     string
		findings []verify.Finding
		want     Verdict
	}{
		{name: "clean", want: VerdictPass},
		{name: "warning flags", findings: []verify.Finding{{Verifier: "loop", Severity: verify.SeverityWarning}}, want: VerdictFlagged},
		{name: "evidence gap is inconclusive", findings: []verify.Finding{{Verifier: "evidence", Kind: verify.FindingEvidenceGap, Severity: verify.SeverityWarning}}, want: VerdictInconclusive},
		{name: "critical fails", findings: []verify.Finding{{Verifier: "schema", Severity: verify.SeverityCritical}}, want: VerdictFail},
		{name: "critical violation beats evidence gap", findings: []verify.Finding{
			{Verifier: "evidence", Kind: verify.FindingEvidenceGap, Severity: verify.SeverityWarning},
			{Verifier: "policy", Severity: verify.SeverityCritical},
		}, want: VerdictFail},
		{name: "warning then critical fails", findings: []verify.Finding{
			{Verifier: "loop", Severity: verify.SeverityWarning},
			{Verifier: "policy", Severity: verify.SeverityCritical},
		}, want: VerdictFail},
		{name: "judge critical alone fails", findings: []verify.Finding{
			{Verifier: "judge", Severity: verify.SeverityCritical, Source: "gemini-3.6-flash"},
		}, want: VerdictFail},
		{name: "judge critical never overrides evidence gap", findings: []verify.Finding{
			{Verifier: "evidence", Kind: verify.FindingEvidenceGap, Severity: verify.SeverityWarning},
			{Verifier: "judge", Severity: verify.SeverityCritical, Source: "gemini-3.6-flash"},
		}, want: VerdictInconclusive},
		{name: "deterministic critical still beats evidence gap", findings: []verify.Finding{
			{Verifier: "evidence", Kind: verify.FindingEvidenceGap, Severity: verify.SeverityWarning},
			{Verifier: "policy", Severity: verify.SeverityCritical},
			{Verifier: "judge", Severity: verify.SeverityCritical, Source: "gemini-3.6-flash"},
		}, want: VerdictFail},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := verdictFor(tt.findings); got != tt.want {
				t.Fatalf("verdictFor() = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestBuildSummarizesVerifiers(t *testing.T) {
	run := &graph.Run{
		TraceID: "t1", RootSpanID: "s0",
		Budget: graph.Budget{SpanCount: 3, TotalTokens: 10},
	}
	findings := []verify.Finding{
		{Verifier: "schema", Severity: verify.SeverityCritical, SpanIDs: []string{"s1"}},
		{Verifier: "schema", Severity: verify.SeverityCritical, SpanIDs: []string{"s2"}},
		{Verifier: "loop", Severity: verify.SeverityWarning, SpanIDs: []string{"s1", "s2", "s3"}},
	}
	r := Build(run, findings, false)
	if r.Verdict != VerdictFail {
		t.Fatalf("verdict = %s, want FAIL", r.Verdict)
	}
	if len(r.Verifiers) != 6 {
		t.Fatalf("verifiers = %d, want 6 (schema, policy, loop, budget, status, evidence)", len(r.Verifiers))
	}
	if r.Verifiers[0].Name != "schema" || r.Verifiers[0].Findings != 2 || r.Verifiers[0].MaxSeverity != verify.SeverityCritical {
		t.Errorf("schema summary = %+v", r.Verifiers[0])
	}
	if r.Verifiers[2].Name != "loop" || r.Verifiers[2].Findings != 1 || r.Verifiers[2].MaxSeverity != verify.SeverityWarning {
		t.Errorf("loop summary = %+v", r.Verifiers[2])
	}
	if r.Verifiers[3].Findings != 0 {
		t.Errorf("budget summary = %+v, want zero findings reported", r.Verifiers[3])
	}
	if r.Verifiers[5].Name != "evidence" || r.Verifiers[5].Findings != 0 {
		t.Errorf("evidence summary = %+v, want zero findings reported", r.Verifiers[5])
	}
}

func TestReportJSONHasEvidence(t *testing.T) {
	run := &graph.Run{TraceID: "t1", RootSpanID: "s0"}
	r := Build(run, []verify.Finding{{
		Verifier: "policy", Severity: verify.SeverityCritical,
		Message: "tool not allowed", SpanIDs: []string{"s7"}, Value: "rm",
	}}, false)
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var back map[string]any
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"traceId", "rootSpanId", "verdict", "verifiers", "budget", "evidence"} {
		if _, ok := back[key]; !ok {
			t.Errorf("report JSON missing %q", key)
		}
	}
	if s := back["verdict"]; s != "FAIL" {
		t.Errorf("verdict = %v, want FAIL", s)
	}
}
