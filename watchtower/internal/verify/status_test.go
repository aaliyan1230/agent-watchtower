package verify

import (
	"testing"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

func TestCheckStatus(t *testing.T) {
	ok := graph.Step{SpanID: "s1", Kind: graph.StepLLM, Status: model.StatusOK}
	bad := graph.Step{SpanID: "s2", Kind: graph.StepLLM, Status: model.StatusError, StatusMsg: "provider timeout"}
	unset := graph.Step{SpanID: "s3", Kind: graph.StepTool, Status: model.StatusUnset}
	badNoMsg := graph.Step{SpanID: "s4", Name: "chat", Kind: graph.StepLLM, Status: model.StatusError}

	f := CheckStatus(&graph.Run{Steps: []graph.Step{ok, bad, unset, badNoMsg}})
	if len(f) != 2 {
		t.Fatalf("findings = %d, want 2 (%+v)", len(f), f)
	}
	if f[0].Message != "span ended with status=error: provider timeout" {
		t.Errorf("message = %q", f[0].Message)
	}
	if f[0].Severity != SeverityCritical || f[0].SpanIDs[0] != "s2" {
		t.Errorf("finding = %+v", f[0])
	}
	if f[1].Value != "chat" {
		t.Errorf("statusMsg fallback = %q, want span name", f[1].Value)
	}
}
