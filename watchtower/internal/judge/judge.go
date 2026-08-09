// Package judge implements verify.Judge over an LLM: the optional,
// calibrated supplement to the deterministic verifiers. The judge sees
// the reconstructed run but NOT the deterministic findings — the
// kappa/agreement analysis needs independent opinions. The transport
// is pluggable (Gemini OpenAI-compatible or Amazon Bedrock native) so
// judge agreement can be measured across providers.
package judge

import (
	"context"
	"fmt"
	"strings"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/bedrock"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/llm"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

const systemPrompt = `You are a runtime verifier for an agent run. Given the run summary, list concrete problems with the agent's behavior that are clearly visible in the evidence (failed tools, repeated actions, missing final answers, excessive resource use, suspicious outputs). Do not speculate about the task itself. Reply with JSON: {"issues": [{"severity": "warning"|"critical", "description": "..."}]}. Keep descriptions short and reference step numbers.`

// chatClient is the one-method transport both providers implement.
type chatClient interface {
	ChatJSON(ctx context.Context, system, user string, out any) error
}

// LLM is a model-backed judge; model identity is carried on findings
// so the experiments can compute per-model agreement (kappa).
type LLM struct {
	client chatClient
	model  string
}

// New builds a judge over an OpenAI-compatible endpoint (Gemini).
func New(model, apiKey, baseURL string) *LLM {
	return &LLM{client: llm.New(model, apiKey, baseURL), model: model}
}

// NewBedrock builds a judge over Amazon Bedrock's native API.
func NewBedrock(model, region string, creds bedrock.Credentials) *LLM {
	return &LLM{client: bedrock.New(model, region, creds), model: model}
}

func (j *LLM) Name() string { return "judge" }

func (j *LLM) Run(run *graph.Run) ([]verify.Finding, error) {
	var resp struct {
		Issues []struct {
			Severity    string `json:"severity"`
			Description string `json:"description"`
		} `json:"issues"`
	}
	if err := j.client.ChatJSON(context.Background(), systemPrompt, summarizeRun(run), &resp); err != nil {
		return nil, err
	}
	var out []verify.Finding
	for _, is := range resp.Issues {
		sev := verify.SeverityWarning
		if is.Severity == "critical" {
			sev = verify.SeverityCritical
		}
		out = append(out, verify.Finding{
			Verifier: "judge",
			Severity: sev,
			Message:  is.Description,
			Source:   j.model,
		})
	}
	return out, nil
}

// summarizeRun renders the run as compact evidence: one line per step
// plus the budget. Compactness matters — judge cost is a paper metric.
func summarizeRun(run *graph.Run) string {
	var b strings.Builder
	fmt.Fprintf(&b, "trace %s\n", run.TraceID)
	for _, st := range run.Steps {
		switch st.Kind {
		case graph.StepTool:
			fmt.Fprintf(&b, "%d tool agent=%s tool=%s ok=%v\n", st.Order, st.Agent, st.Tool, st.ToolOK)
		case graph.StepLLM:
			fmt.Fprintf(&b, "%d llm agent=%s model=%s in=%d out=%d\n", st.Order, st.Agent, st.Model, st.InputTokens, st.OutputTokens)
		default:
			fmt.Fprintf(&b, "%d agent name=%s\n", st.Order, st.Agent)
		}
	}
	budget := run.Budget
	fmt.Fprintf(&b, "budget steps=%d tokens=%d duration_ms=%d\n", budget.SpanCount, budget.TotalTokens, budget.DurationMs)
	return b.String()
}
