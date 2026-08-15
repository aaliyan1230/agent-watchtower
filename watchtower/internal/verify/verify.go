// Package verify holds the deterministic rule engine: pure functions
// from a reconstructed run to findings. Pure = same run in, same
// findings out, always. That property is what makes verdicts
// reproducible from a checksummed trace, and it is what makes the
// verifiers trivially testable (build a run, assert on findings).
//
// Verifiers never decide the run-level outcome. Finding kind and severity
// are data here; the report layer maps them to verdicts.
package verify

import (
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// Severity drives verdict mapping in the report layer: Critical maps to
// FAIL, Warning to FLAGGED.
type Severity int

const (
	SeverityWarning Severity = iota
	SeverityCritical
)

func (s Severity) String() string {
	switch s {
	case SeverityCritical:
		return "critical"
	default:
		return "warning"
	}
}

// FindingKind distinguishes an observed violation from a gap in the
// telemetry needed to establish a confident result.
type FindingKind string

const (
	FindingViolation   FindingKind = "violation"
	FindingEvidenceGap FindingKind = "evidence_gap"
)

// Finding is one verifier observation with evidence: the exact span IDs
// that triggered it and the values involved. Evidence over opinion.
type Finding struct {
	Verifier   string      `json:"verifier"`
	Kind       FindingKind `json:"kind,omitempty"`
	Severity   Severity    `json:"severity"`
	Message    string      `json:"message"`
	SpanIDs    []string    `json:"spanIds"`
	Timestamps []string    `json:"timestamps,omitempty"` // RFC3339 of offending spans
	Value      string      `json:"value,omitempty"`      // the offending value, if any
	Source     string      `json:"source,omitempty"`     // judge model, when a judge produced it
}

// JudgeConfig is the JSON-configurable part of the judge slot. The
// judge itself is constructed by the CLI layer (it needs env keys);
// verify only carries the config.
type JudgeConfig struct {
	Enabled bool   `json:"enabled"`
	Backend string `json:"backend"` // "" or "gemini" = OpenAI-compatible, "bedrock" = Amazon Bedrock
	Model   string `json:"model"`   // empty = provider default
	BaseURL string `json:"baseUrl"` // OpenAI-compatible override (tests / alternate endpoints)
}

// EvidenceConfig controls the structural telemetry-integrity gate.
// Keeping it opt-in preserves the zero-config behavior of the original
// verifier while experiment configs can require complete evidence.
type EvidenceConfig struct {
	Enabled bool `json:"enabled"`
}

// Config aggregates every verifier's configuration; zero values mean
// "check disabled". WithDefaults fills in sensible defaults for the
// optional knobs (loop thresholds). JSON tags map a config file onto
// this struct directly.
type Config struct {
	Contracts []Contract     `json:"contracts"`
	Policy    Policy         `json:"policy"`
	Loop      LoopConfig     `json:"loop"`
	Limits    Limits         `json:"limits"`
	Evidence  EvidenceConfig `json:"evidence"`
	JudgeCfg  JudgeConfig    `json:"judge"`
	Judge     Judge          `json:"-"` // constructed by the CLI from JudgeCfg + env keys
}

// Verify runs every enabled verifier over the run and concatenates the
// findings, keeping verifier order (schema, policy, loop, budget, status,
// evidence, judge). Every check is opt-in: zero config verifies nothing. Judge
// failures surface as findings rather than errors: the judge is a
// supplement, its outage must not hide deterministic results.
func Verify(run *graph.Run, cfg Config) []Finding {
	var out []Finding
	out = append(out, CheckSchema(run, cfg.Contracts)...)
	out = append(out, CheckPolicy(run, cfg.Policy)...)
	if cfg.Loop.Enabled {
		out = append(out, CheckLoop(run, cfg.Loop.withDefaults())...)
	}
	out = append(out, CheckBudget(run, cfg.Limits)...)
	out = append(out, CheckStatus(run)...)
	if cfg.Evidence.Enabled {
		out = append(out, CheckEvidence(run)...)
	}
	if cfg.Judge != nil {
		if js, err := cfg.Judge.Run(run); err != nil {
			out = append(out, Finding{
				Verifier: cfg.Judge.Name(),
				Severity: SeverityWarning,
				Message:  "judge failed: " + err.Error(),
			})
		} else {
			out = append(out, js...)
		}
	}
	return out
}

// VerifierNames lists the deterministic verifiers, in run order — used
// by the report and experiments for per-verifier detection rates.
var VerifierNames = []string{"schema", "policy", "loop", "budget", "status", "evidence"}
