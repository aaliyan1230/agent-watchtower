package judge

import (
	"os"
	"testing"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/bedrock"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// TestBedrockJudgeLive hits the real Bedrock API. Skipped unless AWS
// credentials are in the environment (the experiment runner forwards
// the CLI's session). This is the SigV4 implementation's real test.
func TestBedrockJudgeLive(t *testing.T) {
	if os.Getenv("AWS_ACCESS_KEY_ID") == "" || os.Getenv("AWS_SECRET_ACCESS_KEY") == "" {
		t.Skip("AWS credentials not set")
	}
	creds, err := bedrock.CredentialsFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	j := NewBedrock("amazon.nova-lite-v1:0", "us-east-1", creds)
	run := &graph.Run{
		TraceID: "probe",
		Steps:   []graph.Step{{Order: 1, Kind: graph.StepLLM, Agent: "a", Model: "fake", InputTokens: 5}},
		Budget:  graph.Budget{SpanCount: 1, TotalTokens: 5, DurationMs: 1},
	}
	f, err := j.Run(run)
	if err != nil {
		t.Fatalf("judge: %v", err)
	}
	t.Logf("findings: %d", len(f))
}
