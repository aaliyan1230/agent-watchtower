package decide

import (
	"bytes"
	"encoding/xml"
	"math"
	"strconv"
	"testing"
)

func evalDMNTable(t *testing.T, table DMNDecisionTable, inputs []string) []string {
	t.Helper()
	if table.HitPolicy != "FIRST" {
		t.Fatalf("unexpected hitPolicy %q, want FIRST", table.HitPolicy)
	}
	for _, rule := range table.Rules {
		if len(rule.InputEntries) != len(inputs) {
			t.Fatalf("rule %s has %d input entries, want %d", rule.ID, len(rule.InputEntries), len(inputs))
		}
		matched := true
		for i, entry := range rule.InputEntries {
			if entry.Text == "-" {
				continue
			}
			if entry.Text != inputs[i] {
				matched = false
				break
			}
		}
		if matched {
			out := make([]string, len(rule.OutputEntries))
			for i, oe := range rule.OutputEntries {
				out[i] = oe.Text
			}
			return out
		}
	}
	t.Fatalf("no DMN rule matched inputs %v", inputs)
	return nil
}

func TestExportDMNEquivalenceWithModel(t *testing.T) {
	cases := []struct {
		name      string
		threshold float64
	}{
		{"open gate", 1},
		{"tight gate", 0.015},
		{"closed gate", -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := fixture()
			m.Costs = DefaultCosts()
			m.Alpha = 0.05
			m.Threshold = tc.threshold

			raw1, err := m.ExportDMN()
			if err != nil {
				t.Fatalf("ExportDMN: %v", err)
			}
			raw2, err := m.ExportDMN()
			if err != nil {
				t.Fatalf("second ExportDMN: %v", err)
			}
			if !bytes.Equal(raw1, raw2) {
				t.Fatal("ExportDMN output is not deterministic")
			}

			var defs DMNDefinitions
			if err := xml.Unmarshal(raw1, &defs); err != nil {
				t.Fatalf("invalid XML: %v\n%s", err, raw1)
			}
			if defs.XMLName.Space != dmnNamespace {
				t.Fatalf("namespace = %q, want %q", defs.XMLName.Space, dmnNamespace)
			}
			if len(defs.Decisions) != 2 {
				t.Fatalf("decisions = %d, want 2 (first action + judge settle)", len(defs.Decisions))
			}

			// Verify every signature (observed, unseen per known verdict, completely unseen)
			// across every (canReplay, canJudge) combination.
			testSigs := []string{
				pass,
				fail,
				gap,
				looped,
				styleWarn,
				"PASS|viol:brand_new",
				"FAIL|viol:brand_new",
				"INCONCLUSIVE|gap:brand_new",
				"FLAGGED|viol:brand_new",
				"CUSTOM|viol:unknown",
			}
			fa := defs.Decisions[0].DecisionTable
			for _, sig := range testSigs {
				for _, canReplay := range []bool{true, false} {
					for _, canJudge := range []bool{true, false} {
						want := m.Decide(Observation{Signature: sig, CanReplay: canReplay, CanJudge: canJudge}, m.Costs, true)
						var wantFB Action
						if want.Action == Replay {
							wantFB = m.Settle(want.Risk, m.Costs).Action
						}
						got := evalDMNTable(t, fa, []string{
							feelString(sig),
							feelString(verdictOf(sig)),
							strconv.FormatBool(canReplay),
							strconv.FormatBool(canJudge),
						})
						if got[0] != feelString(string(want.Action)) {
							t.Errorf("sig=%q replay=%t judge=%t: action=%s, want %s", sig, canReplay, canJudge, got[0], want.Action)
						}
						gotRisk, err := strconv.ParseFloat(got[1], 64)
						if err != nil || math.Abs(gotRisk-want.Risk) > 1e-5 {
							t.Errorf("sig=%q: risk=%s, want %.6f", sig, got[1], want.Risk)
						}
						if got[2] != strconv.FormatBool(want.Gated) {
							t.Errorf("sig=%q replay=%t judge=%t: gated=%s, want %t", sig, canReplay, canJudge, got[2], want.Gated)
						}
						if got[3] != feelString(string(wantFB)) {
							t.Errorf("sig=%q replay=%t judge=%t: fallback=%s, want %q", sig, canReplay, canJudge, got[3], wantFB)
						}
					}
				}
			}

			// Verify Judge Settle table equivalence across all signatures and judge flags.
			js := defs.Decisions[1].DecisionTable
			for _, sig := range testSigs {
				for _, flagged := range []bool{true, false} {
					wantPost := m.JudgePosterior(sig, m.Risk(sig), flagged)
					wantSettled := m.Settle(wantPost, m.Costs)
					got := evalDMNTable(t, js, []string{
						feelString(sig),
						feelString(verdictOf(sig)),
						strconv.FormatBool(flagged),
					})
					if got[0] != feelString(string(wantSettled.Action)) {
						t.Errorf("settle sig=%q flag=%t: action=%s, want %s", sig, flagged, got[0], wantSettled.Action)
					}
					gotPost, err := strconv.ParseFloat(got[1], 64)
					if err != nil || math.Abs(gotPost-wantPost) > 1e-5 {
						t.Errorf("settle sig=%q flag=%t: posterior=%s, want %.6f", sig, flagged, got[1], wantPost)
					}
					if got[2] != strconv.FormatBool(wantSettled.Gated) {
						t.Errorf("settle sig=%q flag=%t: gated=%s, want %t", sig, flagged, got[2], wantSettled.Gated)
					}
				}
			}
		})
	}
}

func TestExportDMNWithoutJudgeOmitsSettleTable(t *testing.T) {
	m := Fit(append(repeat(20, Run{Signature: pass}), repeat(20, Run{Signature: fail, Faulty: true})...))
	m.Costs = DefaultCosts()
	raw, err := m.ExportDMN()
	if err != nil {
		t.Fatalf("ExportDMN: %v", err)
	}
	var defs DMNDefinitions
	if err := xml.Unmarshal(raw, &defs); err != nil {
		t.Fatalf("invalid XML: %v", err)
	}
	if len(defs.Decisions) != 1 || defs.Decisions[0].ID != "decision_first_action" {
		t.Fatalf("decisions = %+v, want only decision_first_action", defs.Decisions)
	}
	for _, in := range defs.InputData {
		if in.Name == "judgeFlag" {
			t.Fatal("input_judgeFlag should not be declared when model has no judge stats")
		}
	}
}
