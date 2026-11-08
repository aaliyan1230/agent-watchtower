// Package decide turns a verdict into an action. A verdict states what
// the evidence shows; an operator still has to choose whether to accept
// the run, pay for more evidence, ask a human, or halt. The policy picks
// the action with the lowest expected cost under a calibrated risk
// estimate, buys a probe only when the information is worth its price,
// and gates auto-accept with conformal risk control so the rate of
// accepted faulty runs stays below a chosen alpha.
package decide

import (
	"sort"
	"strings"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/report"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

type Action string

const (
	Accept   Action = "accept"
	Halt     Action = "halt"
	Escalate Action = "escalate"
	Replay   Action = "replay"
	Judge    Action = "judge"
)

// Costs prices each outcome in one unit. Only ratios matter; the
// defaults say a missed fault is ten times worse than a false halt and
// twenty times a human review.
type Costs struct {
	Miss      float64 `json:"miss"`
	FalseHalt float64 `json:"falseHalt"`
	Human     float64 `json:"human"`
	Replay    float64 `json:"replay"`
	Judge     float64 `json:"judge"`
}

func DefaultCosts() Costs {
	return Costs{Miss: 100, FalseHalt: 10, Human: 5, Replay: 1, Judge: 0.5}
}

// Signature is the deterministic observation the risk model conditions
// on: the verdict recomputed without the judge, the verifiers that found
// violations, and the evidence claims left unsupported. The judge is
// excluded because it is a probe the policy may choose to buy.
func Signature(findings []verify.Finding) (sig string, gaps bool) {
	var det []verify.Finding
	violations := map[string]bool{}
	missing := map[string]bool{}
	for _, f := range findings {
		if isJudge(f) {
			continue
		}
		det = append(det, f)
		if f.Kind == verify.FindingEvidenceGap {
			claim := f.Claim
			if claim == "" {
				claim = f.Verifier
			}
			missing[claim] = true
		} else {
			violations[f.Verifier] = true
		}
	}
	parts := append(prefixed("viol:", violations), prefixed("gap:", missing)...)
	return string(report.VerdictFor(det)) + "|" + strings.Join(parts, ","), len(missing) > 0
}

// JudgeFlag reports whether the judge voiced an opinion against the run.
// An outage is not an opinion.
func JudgeFlag(findings []verify.Finding) bool {
	for _, f := range findings {
		if isJudge(f) && f.Kind != verify.FindingJudgeError {
			return true
		}
	}
	return false
}

func isJudge(f verify.Finding) bool {
	return f.Verifier == "judge" || f.Source != "" || f.Kind == verify.FindingJudgeError
}

func prefixed(prefix string, set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, prefix+k)
	}
	sort.Strings(out)
	return out
}

func verdictOf(sig string) string {
	v, _, _ := strings.Cut(sig, "|")
	return v
}

// Run is one labeled run: what the deterministic layer saw, the ground
// truth, and the outcomes the two probes would reveal.
type Run struct {
	Signature string
	Gaps      bool
	Faulty    bool
	// ReplaySignature is the signature of the same behavior recollected
	// with clean telemetry; empty when no recollection exists.
	ReplaySignature string
	Judged          bool
	JudgeFlag       bool
}

// Observation is what the policy may condition on before acting.
type Observation struct {
	Signature string
	CanReplay bool
	CanJudge  bool
}

func (r Run) Observation() Observation {
	return Observation{
		Signature: r.Signature,
		CanReplay: r.Gaps && r.ReplaySignature != "",
		CanJudge:  r.Judged,
	}
}

type counts struct{ faulty, total int }

func (c counts) rate(priorMean, strength float64) float64 {
	return (float64(c.faulty) + strength*priorMean) / (float64(c.total) + strength)
}

// priorStrength is how many pseudo-runs of the verdict-level rate a
// signature's own estimate starts from, so a signature seen twice does
// not claim certainty.
const priorStrength = 2

// Model is the calibrated part of the policy: risk per signature, the
// distribution of replay outcomes, the judge's error rates, and the
// accept gate.
type Model struct {
	signature map[string]counts
	verdict   map[string]counts
	replay    map[string]map[string]int
	judge     struct{ tp, fn, fp, tn int }
	// Threshold is the accept gate: accept only at risk <= Threshold.
	// 1 leaves the gate open; a negative value closes it.
	Threshold float64
}

func Fit(runs []Run) *Model {
	m := &Model{
		signature: map[string]counts{},
		verdict:   map[string]counts{},
		replay:    map[string]map[string]int{},
		Threshold: 1,
	}
	for _, r := range runs {
		add := func(c counts) counts {
			c.total++
			if r.Faulty {
				c.faulty++
			}
			return c
		}
		m.signature[r.Signature] = add(m.signature[r.Signature])
		v := verdictOf(r.Signature)
		m.verdict[v] = add(m.verdict[v])
		if r.Gaps && r.ReplaySignature != "" {
			if m.replay[r.Signature] == nil {
				m.replay[r.Signature] = map[string]int{}
			}
			m.replay[r.Signature][r.ReplaySignature]++
		}
		if r.Judged {
			switch {
			case r.Faulty && r.JudgeFlag:
				m.judge.tp++
			case r.Faulty:
				m.judge.fn++
			case r.JudgeFlag:
				m.judge.fp++
			default:
				m.judge.tn++
			}
		}
	}
	return m
}

// Risk is the estimated probability the run is faulty, shrunk toward
// its verdict's rate, which is itself shrunk toward one half.
func (m *Model) Risk(sig string) float64 {
	prior := m.verdict[verdictOf(sig)].rate(0.5, priorStrength)
	return m.signature[sig].rate(prior, priorStrength)
}

func (m *Model) hasJudge() bool {
	j := m.judge
	return j.tp+j.fn+j.fp+j.tn > 0
}

// judgeRates are Laplace-smoothed true and false positive rates.
func (m *Model) judgeRates() (tpr, fpr float64) {
	j := m.judge
	return float64(j.tp+1) / float64(j.tp+j.fn+2), float64(j.fp+1) / float64(j.fp+j.tn+2)
}

// JudgePosterior updates a risk on the judge's answer by Bayes' rule.
func (m *Model) JudgePosterior(risk float64, flagged bool) float64 {
	tpr, fpr := m.judgeRates()
	if flagged {
		return risk * tpr / (risk*tpr + (1-risk)*fpr)
	}
	return risk * (1 - tpr) / (risk*(1-tpr) + (1-risk)*(1-fpr))
}

func terminalCosts(risk float64, c Costs) map[Action]float64 {
	return map[Action]float64{
		Accept:   risk * c.Miss,
		Halt:     (1 - risk) * c.FalseHalt,
		Escalate: c.Human,
	}
}

func cheapest(costs map[Action]float64, skip Action) Action {
	best, bestCost := Action(""), 0.0
	for _, a := range []Action{Accept, Halt, Escalate, Replay, Judge} {
		cost, ok := costs[a]
		if !ok || a == skip {
			continue
		}
		if best == "" || cost < bestCost {
			best, bestCost = a, cost
		}
	}
	return best
}

func minCost(costs map[Action]float64) float64 {
	return costs[cheapest(costs, "")]
}

// Decision is the chosen action with the expected cost of every action
// that was available, so the choice can be audited.
type Decision struct {
	Action   Action             `json:"action"`
	Risk     float64            `json:"risk"`
	Expected map[Action]float64 `json:"expected"`
	// Gated is set when accepting was cheapest but the risk gate refused.
	Gated bool `json:"gated,omitempty"`
}

// Decide picks the first action for an observation. Probes are valued by
// one-step lookahead: their price plus the expected cost of the best
// terminal action after seeing the answer. Lookahead ignores the gate so
// that whether a run is ever accepted moves monotonically with the
// threshold, which the conformal guarantee needs.
func (m *Model) Decide(obs Observation, c Costs, probes bool) Decision {
	risk := m.Risk(obs.Signature)
	expected := terminalCosts(risk, c)
	if probes && obs.CanReplay && len(m.replay[obs.Signature]) > 0 {
		outcomes := m.replay[obs.Signature]
		total := 0
		for _, n := range outcomes {
			total += n
		}
		after := 0.0
		for sig, n := range outcomes {
			after += float64(n) / float64(total) * minCost(terminalCosts(m.Risk(sig), c))
		}
		expected[Replay] = c.Replay + after
	}
	if probes && obs.CanJudge && m.hasJudge() {
		tpr, fpr := m.judgeRates()
		pFlag := risk*tpr + (1-risk)*fpr
		expected[Judge] = c.Judge +
			pFlag*minCost(terminalCosts(m.JudgePosterior(risk, true), c)) +
			(1-pFlag)*minCost(terminalCosts(m.JudgePosterior(risk, false), c))
	}
	return m.gate(Decision{Action: cheapest(expected, ""), Risk: risk, Expected: expected})
}

// Settle picks the terminal action once a probe has answered.
func (m *Model) Settle(risk float64, c Costs) Decision {
	expected := terminalCosts(risk, c)
	return m.gate(Decision{Action: cheapest(expected, ""), Risk: risk, Expected: expected})
}

func (m *Model) gate(d Decision) Decision {
	if d.Action == Accept && d.Risk > m.Threshold {
		d.Action = cheapest(map[Action]float64{Halt: d.Expected[Halt], Escalate: d.Expected[Escalate]}, "")
		d.Gated = true
	}
	return d
}
