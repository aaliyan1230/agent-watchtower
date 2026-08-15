package verify

import (
	"fmt"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// CheckEvidence turns structural telemetry gaps found during graph
// reconstruction into evidence findings. These findings do not claim a
// behavior violation; they prevent the report from claiming PASS when
// the trace cannot support that conclusion.
func CheckEvidence(run *graph.Run) []Finding {
	if run == nil || run.Evidence.Complete() {
		return nil
	}

	steps := make(map[string]graph.Step, len(run.Steps))
	for _, step := range run.Steps {
		steps[step.SpanID] = step
	}

	findings := make([]Finding, 0, len(run.Evidence.DuplicateSpanIDs)+len(run.Evidence.MissingParents))
	for _, spanID := range run.Evidence.DuplicateSpanIDs {
		finding := Finding{
			Verifier: "evidence",
			Kind:     FindingEvidenceGap,
			Severity: SeverityWarning,
			Message:  fmt.Sprintf("span id %q appeared more than once", spanID),
			SpanIDs:  []string{spanID},
			Value:    spanID,
		}
		if step, ok := steps[spanID]; ok {
			finding.Timestamps = []string{step.StartTime.UTC().Format("2006-01-02T15:04:05Z")}
		}
		findings = append(findings, finding)
	}
	for _, missing := range run.Evidence.MissingParents {
		finding := Finding{
			Verifier: "evidence",
			Kind:     FindingEvidenceGap,
			Severity: SeverityWarning,
			Message:  fmt.Sprintf("span %q refers to missing parent %q", missing.SpanID, missing.ParentID),
			SpanIDs:  []string{missing.SpanID},
			Value:    missing.ParentID,
		}
		if step, ok := steps[missing.SpanID]; ok {
			finding.Timestamps = []string{step.StartTime.UTC().Format("2006-01-02T15:04:05Z")}
		}
		findings = append(findings, finding)
	}
	return findings
}
