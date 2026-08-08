package verify

import (
	"testing"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// llmStep builds an LLM step carrying a structured-output contract and
// raw output — the schema verifier's input.
func llmStep(contract, output string) graph.Step {
	return graph.Step{Kind: graph.StepLLM, Contract: contract, Output: output, StartTime: t(1)}
}

func TestCheckSchema(t *testing.T) {
	contracts := []Contract{{
		Name: "ticket",
		Fields: map[string]FieldSpec{
			"id":     {Required: true, Type: TypeInt},
			"title":  {Required: true, Type: TypeString},
			"labels": {Required: false, Type: TypeArray},
		},
	}}
	tests := []struct {
		name     string
		steps    []graph.Step
		want     int
		wantFrag string
	}{
		{name: "clean output", steps: []graph.Step{
			llmStep("ticket", `{"id": 42, "title": "broken login"}`),
		}},
		{name: "missing required field", steps: []graph.Step{
			llmStep("ticket", `{"id": 42}`),
		}, want: 1, wantFrag: "title"},
		{name: "type mismatch", steps: []graph.Step{
			llmStep("ticket", `{"id": "42", "title": "x"}`),
		}, want: 1, wantFrag: "id"},
		{name: "not json", steps: []graph.Step{
			llmStep("ticket", `{oops`),
		}, want: 1, wantFrag: "not valid JSON"},
		{name: "not an object", steps: []graph.Step{
			llmStep("ticket", `[1,2]`),
		}, want: 1, wantFrag: "not a JSON object"},
		{name: "unknown contract", steps: []graph.Step{
			llmStep("nope", `{}`),
		}, want: 1, wantFrag: "unknown contract"},
		{name: "declared but no output", steps: []graph.Step{
			llmStep("ticket", ""),
		}, want: 1, wantFrag: "no output"},
		{name: "no contract attr ignored", steps: []graph.Step{
			{Kind: graph.StepLLM, Output: `{"id": 1}`},
		}},
		{name: "multiple violations", steps: []graph.Step{
			llmStep("ticket", `{}`),
			llmStep("ticket", `{"id": "x", "title": 3}`),
		}, want: 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run := &graph.Run{Steps: tt.steps}
			f := CheckSchema(run, contracts)
			if len(f) != tt.want {
				t.Fatalf("findings = %d, want %d (%+v)", len(f), tt.want, f)
			}
			if tt.want > 0 {
				if f[0].Verifier != "schema" || f[0].Severity != SeverityCritical {
					t.Errorf("finding = %+v, want schema/critical", f[0])
				}
				if len(f[0].SpanIDs) != 1 {
					t.Errorf("finding has %d span ids, want 1 (evidence)", len(f[0].SpanIDs))
				}
				if !contains(f[0].Message, tt.wantFrag) {
					t.Errorf("message %q missing %q", f[0].Message, tt.wantFrag)
				}
			}
		})
	}
}
