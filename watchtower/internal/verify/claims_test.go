package verify

import (
	"strings"
	"testing"
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

func TestClaimReferenceIsComplete(t *testing.T) {
	claims := ClaimReference()
	if len(claims) < 8 {
		t.Fatalf("ClaimReference() = %d claims, want at least 8", len(claims))
	}
	registry := attributeProvenance()
	seen := make(map[string]bool)
	for _, claim := range claims {
		if claim.Name == "" || claim.Statement == "" || claim.Action == "" {
			t.Errorf("claim %+v has empty name, statement, or action", claim)
		}
		if seen[claim.Name] {
			t.Errorf("duplicate claim name %q", claim.Name)
		}
		seen[claim.Name] = true
		if len(claim.Requires) == 0 {
			t.Errorf("claim %q has no evidence requirements", claim.Name)
		}
		for _, req := range claim.Requires {
			if req.What == "" || req.Provenance == "" {
				t.Errorf("claim %q has empty requirement %+v", claim.Name, req)
			}
			if req.Provenance == ProvenanceOTelCore {
				continue // structural requirement (span.<kind>, parent.<fact>, event.<name>)
			}
			prov, ok := registry[req.What]
			if !ok {
				t.Errorf("claim %q requires attribute %q with no provenance registry entry", claim.Name, req.What)
				continue
			}
			if prov != req.Provenance {
				t.Errorf("claim %q requires %q with provenance %q, registry says %q", claim.Name, req.What, req.Provenance, prov)
			}
		}
	}
}

func isAttribute(req EvidenceReq) bool {
	return strings.Contains(req.What, ".")
}

func TestObligationsMapToClaims(t *testing.T) {
	if claims := (EvidenceObligations{}).Claims(); len(claims) != 1 {
		t.Fatalf("zero obligations = %d claims, want only trace_integrity", len(claims))
	} else if claims[0].Name != "trace_integrity" {
		t.Fatalf("zero obligations = %q, want trace_integrity", claims[0].Name)
	}

	all := EvidenceObligations{
		RequireRoot:               true,
		RequireAgentCompletion:    true,
		RequireModelCorrelation:   true,
		RequireToolResults:        true,
		RequireFinalAnswer:        true,
		RequireSupervisorDecision: true,
		RequireTemporalNesting:    true,
	}
	want := []string{
		"run_root", "agent_completion", "model_correlation", "tool_pairing",
		"final_answer", "supervisor_decision", "temporal_nesting", "trace_integrity",
	}
	claims := all.Claims()
	if len(claims) != len(want) {
		t.Fatalf("all obligations = %d claims, want %d", len(claims), len(want))
	}
	for i, name := range want {
		if claims[i].Name != name {
			t.Errorf("claim %d = %q, want %q", i, claims[i].Name, name)
		}
	}
}

func TestEvidenceFindingsCarryClaimAndAction(t *testing.T) {
	run := &graph.Run{
		Steps: []graph.Step{{
			SpanID: "chat-1", Kind: graph.StepLLM, Name: "chat",
			StartTime: time.Unix(101, 0).UTC(),
			Attributes: map[string]string{
				"gen_ai.operation.name": "chat",
				"gen_ai.request.model":  "gofake-1",
				"gen_ai.response.model": "gofake-1",
				// gen_ai.system deliberately missing
			},
		}},
	}
	findings := CheckEvidence(run, EvidenceObligations{RequireModelCorrelation: true})
	if len(findings) == 0 {
		t.Fatal("expected a model correlation gap")
	}
	f := findings[0]
	if f.Claim != "model_correlation" {
		t.Errorf("claim = %q, want model_correlation", f.Claim)
	}
	if f.Action == "" {
		t.Error("evidence finding has no recovery action")
	}
	if len(f.SpanIDs) == 0 {
		t.Error("evidence finding has no span references")
	}
	if f.Kind != FindingEvidenceGap {
		t.Errorf("kind = %q, want evidence_gap", f.Kind)
	}
}
