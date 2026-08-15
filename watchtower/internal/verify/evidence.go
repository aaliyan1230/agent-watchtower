package verify

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

const supervisorDecisionEvent = "supervisor.decision"

// CheckEvidence turns structural telemetry gaps found during graph
// reconstruction, plus configured evidence-obligation failures, into
// evidence findings. These findings do not claim a behavior violation;
// they prevent the report from claiming PASS when the trace cannot
// support that conclusion.
func CheckEvidence(run *graph.Run, obligations EvidenceObligations) []Finding {
	if run == nil {
		return nil
	}

	steps := make(map[string]graph.Step, len(run.Steps))
	for _, step := range run.Steps {
		steps[step.SpanID] = step
	}

	findings := structuralFindings(run, steps)
	if obligations.RequireRoot {
		findings = append(findings, checkRoot(run, steps))
	}
	if obligations.RequireAgentCompletion {
		findings = append(findings, checkAgentCompletion(run.Steps, steps)...)
	}
	if obligations.RequireModelCorrelation {
		findings = append(findings, checkModelCorrelation(run.Steps, steps)...)
	}
	if obligations.RequireToolResults {
		findings = append(findings, checkToolResults(run.Steps, steps)...)
	}
	if obligations.RequireFinalAnswer {
		findings = append(findings, checkFinalAnswer(run, steps))
	}
	if obligations.RequireSupervisorDecision {
		findings = append(findings, checkSupervisorDecision(run, steps))
	}
	if obligations.RequireTemporalNesting {
		findings = append(findings, checkTemporalNesting(run.Steps, steps)...)
	}
	return compactFindings(findings)
}

func structuralFindings(run *graph.Run, steps map[string]graph.Step) []Finding {
	findings := make([]Finding, 0, len(run.Evidence.DuplicateSpanIDs)+len(run.Evidence.MissingParents))
	for _, spanID := range run.Evidence.DuplicateSpanIDs {
		findings = append(findings, evidenceFinding(
			fmt.Sprintf("span id %q appeared more than once", spanID),
			[]string{spanID}, spanID, steps,
		))
	}
	for _, missing := range run.Evidence.MissingParents {
		findings = append(findings, evidenceFinding(
			fmt.Sprintf("span %q refers to missing parent %q", missing.SpanID, missing.ParentID),
			[]string{missing.SpanID}, missing.ParentID, steps,
		))
	}
	return findings
}

func checkRoot(run *graph.Run, steps map[string]graph.Step) Finding {
	root, ok := steps[run.RootSpanID]
	if ok && root.Kind == graph.StepAgent && root.Name == "agent.run" && root.ParentID == "" {
		return Finding{}
	}
	spanIDs := []string{}
	if ok {
		spanIDs = []string{root.SpanID}
	}
	return evidenceFinding(
		"run has no complete root agent span",
		spanIDs,
		"agent.run with no parent",
		steps,
	)
}

func checkAgentCompletion(runSteps []graph.Step, steps map[string]graph.Step) []Finding {
	var findings []Finding
	for _, step := range runSteps {
		if step.Kind != graph.StepAgent {
			continue
		}
		completed, valid := boolAttr(step, model.WatchtowerCompleted)
		if valid && completed {
			continue
		}
		findings = append(findings, evidenceFinding(
			fmt.Sprintf("agent span %q has no valid completion marker", step.SpanID),
			[]string{step.SpanID}, model.WatchtowerCompleted, steps,
		))
	}
	return findings
}

func checkModelCorrelation(runSteps []graph.Step, steps map[string]graph.Step) []Finding {
	var findings []Finding
	for _, step := range runSteps {
		if step.Kind != graph.StepLLM {
			continue
		}
		missing := make([]string, 0, 4)
		if strings.TrimSpace(step.System) == "" {
			missing = append(missing, model.GenAISystem)
		}
		if strings.TrimSpace(step.Model) == "" {
			missing = append(missing, model.GenAIRequestModel)
		}
		if step.Status != model.StatusError && strings.TrimSpace(step.ResponseModel) == "" {
			missing = append(missing, model.GenAIResponseModel)
		}
		if strings.TrimSpace(step.Name) == "" {
			missing = append(missing, "span.name")
		}
		if len(missing) == 0 {
			continue
		}
		findings = append(findings, evidenceFinding(
			fmt.Sprintf("LLM span %q is missing correlation fields: %s", step.SpanID, strings.Join(missing, ", ")),
			[]string{step.SpanID}, strings.Join(missing, ","), steps,
		))
	}
	return findings
}

func checkToolResults(runSteps []graph.Step, steps map[string]graph.Step) []Finding {
	var findings []Finding
	byCallID := make(map[string]graph.Step)
	for _, step := range runSteps {
		if step.Kind != graph.StepTool {
			continue
		}
		if strings.TrimSpace(step.ToolCallID) == "" {
			findings = append(findings, evidenceFinding(
				fmt.Sprintf("tool span %q has no call id", step.SpanID),
				[]string{step.SpanID}, model.ToolCallID, steps,
			))
		} else if previous, exists := byCallID[step.ToolCallID]; exists {
			findings = append(findings, evidenceFinding(
				fmt.Sprintf("tool call id %q is used by multiple spans", step.ToolCallID),
				[]string{previous.SpanID, step.SpanID}, step.ToolCallID, steps,
			))
		} else {
			byCallID[step.ToolCallID] = step
		}
		if _, valid := boolAttr(step, model.ToolResultOK); !valid {
			findings = append(findings, evidenceFinding(
				fmt.Sprintf("tool span %q has no valid result marker", step.SpanID),
				[]string{step.SpanID}, model.ToolResultOK, steps,
			))
		}
	}

	for _, step := range runSteps {
		if step.Kind != graph.StepLLM {
			continue
		}
		for _, callID := range step.ToolCallIDs {
			if _, exists := byCallID[callID]; exists {
				continue
			}
			findings = append(findings, evidenceFinding(
				fmt.Sprintf("LLM span %q expects tool result %q, but no matching tool span was observed", step.SpanID, callID),
				[]string{step.SpanID}, callID, steps,
			))
		}
	}
	return findings
}

func checkFinalAnswer(run *graph.Run, steps map[string]graph.Step) Finding {
	for _, step := range run.Steps {
		if step.Kind != graph.StepLLM {
			continue
		}
		final, valid := boolAttr(step, model.WatchtowerFinal)
		output := strings.TrimSpace(step.Attributes[model.WatchtowerOutput])
		if valid && final && output != "" {
			return Finding{}
		}
	}
	spanIDs := []string{}
	if run.RootSpanID != "" {
		spanIDs = []string{run.RootSpanID}
	}
	return evidenceFinding(
		"run has no observed final answer span",
		spanIDs,
		model.WatchtowerFinal,
		steps,
	)
}

func checkSupervisorDecision(run *graph.Run, steps map[string]graph.Step) Finding {
	root, ok := steps[run.RootSpanID]
	if ok {
		for _, event := range root.Events {
			if event.Name != supervisorDecisionEvent {
				continue
			}
			if action := strings.TrimSpace(event.Attributes["action"]); action != "" {
				return Finding{}
			}
		}
	}
	spanIDs := []string{}
	if ok {
		spanIDs = []string{root.SpanID}
	}
	return evidenceFinding(
		"root agent span has no terminal supervisor decision event",
		spanIDs,
		supervisorDecisionEvent,
		steps,
	)
}

func checkTemporalNesting(runSteps []graph.Step, steps map[string]graph.Step) []Finding {
	var findings []Finding
	for _, step := range runSteps {
		if step.ParentID == "" {
			continue
		}
		parent, ok := steps[step.ParentID]
		if !ok {
			continue // structuralFindings already reports the missing parent
		}
		if !step.StartTime.Before(parent.StartTime) && !step.EndTime.After(parent.EndTime) {
			continue
		}
		findings = append(findings, evidenceFinding(
			fmt.Sprintf("span %q is not temporally nested inside parent %q", step.SpanID, parent.SpanID),
			[]string{step.SpanID, parent.SpanID},
			fmt.Sprintf("child=%s..%s parent=%s..%s", formatTime(step.StartTime), formatTime(step.EndTime), formatTime(parent.StartTime), formatTime(parent.EndTime)),
			steps,
		))
	}
	return findings
}

func boolAttr(step graph.Step, key string) (bool, bool) {
	raw, ok := step.Attributes[key]
	if !ok {
		return false, false
	}
	value, err := strconv.ParseBool(raw)
	return value, err == nil
}

func evidenceFinding(message string, spanIDs []string, value string, steps map[string]graph.Step) Finding {
	timestamps := make([]string, 0, len(spanIDs))
	for _, spanID := range spanIDs {
		if step, ok := steps[spanID]; ok && !step.StartTime.IsZero() {
			timestamps = append(timestamps, formatTime(step.StartTime))
		}
	}
	return Finding{
		Verifier:   "evidence",
		Kind:       FindingEvidenceGap,
		Severity:   SeverityWarning,
		Message:    message,
		SpanIDs:    spanIDs,
		Timestamps: timestamps,
		Value:      value,
	}
}

func formatTime(t time.Time) string {
	return t.UTC().Format(time.RFC3339Nano)
}

func compactFindings(findings []Finding) []Finding {
	out := make([]Finding, 0, len(findings))
	for _, finding := range findings {
		if finding.Verifier != "" {
			out = append(out, finding)
		}
	}
	return out
}
