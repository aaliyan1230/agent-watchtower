package verify

import (
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// Judge is the optional LLM-as-judge pass. It is deliberately an
// interface: the paper's calibration claims (kappa, cost/latency
// tradeoff) need a pluggable judge, and the deterministic verifiers
// must not depend on any judge implementation.
type Judge interface {
	Name() string
	Run(run *graph.Run) ([]Finding, error)
	// Usage reports the token usage of the most recent Run — judge
	// cost is a paper metric, so it is measured, not estimated.
	Usage() (input, output int64)
}

// OfflineJudge is the bootstrap default: it judges nothing. Real judge
// implementations arrive with the Python provider layer (Phase 1),
// calling a model with the reconstructed run as evidence.
type OfflineJudge struct{}

func (OfflineJudge) Name() string { return "offline" }

func (OfflineJudge) Run(*graph.Run) ([]Finding, error) { return nil, nil }

func (OfflineJudge) Usage() (int64, int64) { return 0, 0 }
