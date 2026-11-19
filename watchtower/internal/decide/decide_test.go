package decide

import (
	"math"
	"testing"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

const (
	pass      = "PASS|"
	fail      = "FAIL|viol:schema"
	gap       = "INCONCLUSIVE|gap:tool_results"
	looped    = "FLAGGED|viol:loop"
	styleWarn = "FLAGGED|viol:style"
)

func repeat(n int, r Run) []Run {
	out := make([]Run, n)
	for i := range out {
		out[i] = r
	}
	return out
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestSignature(t *testing.T) {
	judgeWarn := verify.Finding{Verifier: "judge", Severity: verify.SeverityWarning, Source: "m"}
	tests := []struct {
		name     string
		findings []verify.Finding
		want     string
		gaps     bool
		flag     bool
	}{
		{"clean", nil, "PASS|", false, false},
		{"judge excluded from verdict", []verify.Finding{{Verifier: "loop"}, judgeWarn}, "FLAGGED|viol:loop", false, true},
		{"judge critical alone stays pass", []verify.Finding{{Verifier: "judge", Severity: verify.SeverityCritical, Source: "m"}}, "PASS|", false, true},
		{"outage is neither evidence nor opinion", []verify.Finding{{Verifier: "judge", Kind: verify.FindingJudgeError}}, "PASS|", false, false},
		{
			"violations and gaps sorted",
			[]verify.Finding{
				{Verifier: "evidence", Kind: verify.FindingEvidenceGap, Claim: "tool_results"},
				{Verifier: "schema", Severity: verify.SeverityCritical},
				{Verifier: "budget", Severity: verify.SeverityCritical},
			},
			"FAIL|viol:budget,viol:schema,gap:tool_results", true, false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sig, gaps := Signature(tt.findings)
			if sig != tt.want || gaps != tt.gaps {
				t.Fatalf("Signature() = %q, %v; want %q, %v", sig, gaps, tt.want, tt.gaps)
			}
			if got := JudgeFlag(tt.findings); got != tt.flag {
				t.Fatalf("JudgeFlag() = %v, want %v", got, tt.flag)
			}
		})
	}
}

func TestRiskShrinksTowardVerdictRate(t *testing.T) {
	runs := append(repeat(2, Run{Signature: gap, Gaps: true, Faulty: true}),
		Run{Signature: gap, Gaps: true},
		Run{Signature: pass})
	m := Fit(runs)
	tests := []struct {
		sig  string
		want float64
	}{
		{gap, (2 + 2*0.6) / 5.0},
		{"INCONCLUSIVE|gap:other", 0.6},
		{"FAIL|viol:unseen", 0.5},
		{pass, (2.0 / 3) / 3},
	}
	for _, tt := range tests {
		if got := m.Risk(tt.sig); !near(got, tt.want) {
			t.Errorf("Risk(%q) = %v, want %v", tt.sig, got, tt.want)
		}
	}
}

// fixture fits a model where PASS is safe, FAIL is faulty, the gap
// signature is a 60/40 mix that a replay resolves, and a loop warning
// is a 50/50 mix the judge separates perfectly.
func fixture() *Model {
	var runs []Run
	runs = append(runs, repeat(80, Run{Signature: pass})...)
	runs = append(runs, repeat(80, Run{Signature: fail, Faulty: true})...)
	runs = append(runs, repeat(12, Run{Signature: gap, Gaps: true, Faulty: true, ReplaySignature: fail})...)
	runs = append(runs, repeat(8, Run{Signature: gap, Gaps: true, ReplaySignature: pass})...)
	runs = append(runs, repeat(5, Run{Signature: looped, Faulty: true, Judged: true, JudgeFlag: true})...)
	runs = append(runs, repeat(5, Run{Signature: looped, Judged: true})...)
	runs = append(runs, repeat(98, Run{Signature: styleWarn})...)
	runs = append(runs, repeat(2, Run{Signature: styleWarn, Faulty: true})...)
	return Fit(runs)
}

func TestDecide(t *testing.T) {
	m := fixture()
	c := DefaultCosts()
	tests := []struct {
		name   string
		obs    Observation
		probes bool
		want   Action
	}{
		{"safe pass accepts", Observation{Signature: pass}, true, Accept},
		{"observed failure halts", Observation{Signature: fail}, true, Halt},
		{"mixed gap buys a replay", Observation{Signature: gap, CanReplay: true}, true, Replay},
		{"no replay without a recollection", Observation{Signature: gap}, true, Halt},
		{"probes disabled", Observation{Signature: gap, CanReplay: true}, false, Halt},
		{"uncertain warning buys the judge", Observation{Signature: looped, CanJudge: true}, true, Judge},
		{"no judge when the run cannot be judged", Observation{Signature: looped}, true, Escalate},
		{"rare-fault warning still buys a cheap judge", Observation{Signature: styleWarn, CanJudge: true}, true, Judge},
		{"rare-fault warning accepts without a judge", Observation{Signature: styleWarn}, true, Accept},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := m.Decide(tt.obs, c, tt.probes)
			if d.Action != tt.want {
				t.Fatalf("Decide() = %s (expected costs %v), want %s", d.Action, d.Expected, tt.want)
			}
			if d.Expected[d.Action] != minCost(d.Expected) {
				t.Fatalf("chosen %s is not the cheapest in %v", d.Action, d.Expected)
			}
		})
	}
}

func TestJudgePosterior(t *testing.T) {
	m := fixture()
	// Only FLAGGED has judged runs (5 TP, 0 FN, 0 FP, 5 TN), so leave-one-verdict-out
	// prior is 0.5 and FLAGGED gets exact Laplace rates 6/7 and 1/7 without double-counting.
	tpr, fpr := m.JudgeRates(looped)
	if !near(tpr, 6.0/7) || !near(fpr, 1.0/7) {
		t.Fatalf("JudgeRates(%q) = %v, %v; want 6/7, 1/7", looped, tpr, fpr)
	}
	if got := m.JudgePosterior(looped, 0.5, true); !near(got, 6.0/7) {
		t.Errorf("flagged posterior = %v", got)
	}
	if got := m.JudgePosterior(looped, 0.5, false); !near(got, 1.0/7) {
		t.Errorf("cleared posterior = %v", got)
	}
	// An unseen verdict borrows strength from all other verdicts (the pooled rates).
	pTPR, pFPR := m.JudgeRates(pass)
	if !near(pTPR, 6.0/7) || !near(pFPR, 1.0/7) {
		t.Fatalf("unseen verdict JudgeRates(%q) = %v, %v; want pooled 6/7, 1/7", pass, pTPR, pFPR)
	}
}

func TestJudgeRatesPerVerdictPreventsPoolingOverstatement(t *testing.T) {
	// The judge flags obvious FAIL runs reliably, yet is pure coin-flip on PASS runs
	// (30% fault rate on PASS: expected accept=30, halt=7, escalate=5).
	// A pooled estimate would credit PASS with FAIL's high TPR and wrongly buy the judge;
	// per-verdict rates see that the judge adds no information on PASS and skip it.
	var runs []Run
	runs = append(runs, repeat(100, Run{Signature: fail, Faulty: true, Judged: true, JudgeFlag: true})...)
	runs = append(runs, repeat(15, Run{Signature: pass, Faulty: true, Judged: true, JudgeFlag: true})...)
	runs = append(runs, repeat(15, Run{Signature: pass, Faulty: true, Judged: true, JudgeFlag: false})...)
	runs = append(runs, repeat(35, Run{Signature: pass, Faulty: false, Judged: true, JudgeFlag: true})...)
	runs = append(runs, repeat(35, Run{Signature: pass, Faulty: false, Judged: true, JudgeFlag: false})...)

	m := Fit(runs)
	passTPR, passFPR := m.JudgeRates(pass)
	failTPR, _ := m.JudgeRates(fail)
	if failTPR <= 0.9 {
		t.Fatalf("FAIL TPR = %v, want > 0.9", failTPR)
	}
	if math.Abs(passTPR-0.5) > 0.06 || math.Abs(passFPR-0.5) > 0.06 {
		t.Fatalf("PASS judge rates = (%v, %v), want near (0.5, 0.5)", passTPR, passFPR)
	}
	d := m.Decide(Observation{Signature: pass, CanJudge: true}, DefaultCosts(), true)
	if d.Action == Judge {
		t.Fatalf("per-verdict model wrongly bought a coin-flip judge on PASS: %+v", d)
	}

	// Legacy saved models without JudgeByVerdict fall back to pooled JudgeStats.
	legacy := *m
	legacy.JudgeByVerdict = nil
	ltpr, lfpr := legacy.JudgeRates(pass)
	wantPooledTPR := float64(115+1) / float64(130+2)
	wantPooledFPR := float64(35+1) / float64(70+2)
	if !near(ltpr, wantPooledTPR) || !near(lfpr, wantPooledFPR) {
		t.Fatalf("legacy fallback rates = (%v, %v), want (%v, %v)", ltpr, lfpr, wantPooledTPR, wantPooledFPR)
	}
}

func TestGateRefusesRiskyAccept(t *testing.T) {
	m := fixture()
	m.Threshold = -1
	d := m.Decide(Observation{Signature: pass}, DefaultCosts(), true)
	if d.Action != Escalate || !d.Gated {
		t.Fatalf("closed gate: got %s gated=%v, want escalate gated", d.Action, d.Gated)
	}
	m.Threshold = 1
	if d := m.Decide(Observation{Signature: pass}, DefaultCosts(), true); d.Action != Accept || d.Gated {
		t.Fatalf("open gate: got %s gated=%v", d.Action, d.Gated)
	}
}

func TestCalibrateAccept(t *testing.T) {
	cal := append(repeat(50, Run{Signature: pass}), repeat(47, Run{Signature: styleWarn})...)
	cal = append(cal, repeat(3, Run{Signature: styleWarn, Faulty: true})...)
	m := fixture()
	passRisk, styleRisk := m.Risk(pass), m.Risk(styleWarn)
	tests := []struct {
		name  string
		cal   []Run
		alpha float64
		want  float64
	}{
		{"both accept levels fit under alpha", cal, 0.05, styleRisk},
		{"tighter alpha keeps only the safe level", cal, 0.03, passRisk},
		{"too few runs to certify anything", repeat(10, Run{Signature: pass}), 0.05, -1},
		{"no accepts leaves the gate closed", repeat(100, Run{Signature: fail, Faulty: true}), 0.05, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := fixture()
			if got := m.CalibrateAccept(tt.cal, DefaultCosts(), true, tt.alpha); !near(got, tt.want) {
				t.Fatalf("CalibrateAccept() = %v, want %v", got, tt.want)
			}
			if m.Threshold != tt.want && !near(m.Threshold, tt.want) {
				t.Fatalf("Threshold = %v not stored", m.Threshold)
			}
		})
	}
}

func TestEvaluate(t *testing.T) {
	c := DefaultCosts()
	runs := []Run{
		{Signature: pass},
		{Signature: pass, Faulty: true, Judged: true, JudgeFlag: true},
		{Signature: gap, Gaps: true, Faulty: true, ReplaySignature: fail},
		{Signature: fail},
	}
	m := fixture()
	tests := []struct {
		policy Policy
		want   Metrics
	}{
		{RulesPolicy(c), Metrics{Policy: "rules", Runs: 4, CostPerRun: 115.0 / 4, AcceptedFaulty: 0.25, AcceptPer100: 50, FalseHaltPer100: 25, EscalatePer100: 25}},
		{JudgeFirstPolicy(c), Metrics{Policy: "judge-first", Runs: 4, CostPerRun: 15.5 / 4, AcceptPer100: 25, FalseHaltPer100: 25, EscalatePer100: 25, ProbePer100: 25}},
		{OraclePolicy(), Metrics{Policy: "oracle", Runs: 4, AcceptPer100: 50}},
	}
	for _, tt := range tests {
		t.Run(tt.policy.Name, func(t *testing.T) {
			got := Evaluate(tt.policy, runs)
			if got.Runs != tt.want.Runs || !near(got.CostPerRun, tt.want.CostPerRun) ||
				!near(got.AcceptedFaulty, tt.want.AcceptedFaulty) || !near(got.AcceptPer100, tt.want.AcceptPer100) ||
				!near(got.FalseHaltPer100, tt.want.FalseHaltPer100) || !near(got.EscalatePer100, tt.want.EscalatePer100) ||
				!near(got.ProbePer100, tt.want.ProbePer100) {
				t.Fatalf("Evaluate() = %+v, want %+v", got, tt.want)
			}
		})
	}

	replayed := BayesPolicy("bayes", m, c, true).Act(runs[2])
	if replayed.First != Replay || replayed.Final != Halt || !near(replayed.Cost, c.Replay) {
		t.Fatalf("replay path = %+v, want replay then halt at the replay price", replayed)
	}
}
