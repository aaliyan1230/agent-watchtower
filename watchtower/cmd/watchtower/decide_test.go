package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeArtifact builds a five-seed results file: clean and schema-faulty
// behavior, each recorded with complete telemetry and with a dropped tool
// result, so loadRuns has replay pairs to attach.
func writeArtifact(t *testing.T) string {
	t.Helper()
	gap := map[string]any{"verifier": "evidence", "kind": "evidence_gap", "claim": "tool_results", "severity": 0}
	schema := map[string]any{"verifier": "schema", "severity": 1}
	judge := map[string]any{"verifier": "judge", "severity": 0, "source": "m"}
	var cells []map[string]any
	for seed := 1; seed <= 5; seed++ {
		for _, c := range []struct {
			fault, evidence any
			findings        []map[string]any
		}{
			{nil, nil, nil},
			{nil, "drop_tool_result", []map[string]any{gap}},
			{"schema_violation", nil, []map[string]any{schema, judge}},
			{"schema_violation", "drop_tool_result", []map[string]any{gap, schema}},
		} {
			for run := 1; run <= 5; run++ {
				cells = append(cells, map[string]any{
					"fault": c.fault, "evidence_fault": c.evidence, "seed": seed,
					"model": "flash", "run": run, "judged": true, "findings": c.findings,
				})
			}
		}
	}
	payload, err := json.Marshal(map[string]any{"cells": cells})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "results.json")
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadRunsAttachesReplayOutcome(t *testing.T) {
	bySeed, err := loadRuns(writeArtifact(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(bySeed) != 5 || len(bySeed[1]) != 20 {
		t.Fatalf("runs per seed = %d over %d seeds", len(bySeed[1]), len(bySeed))
	}
	for _, r := range bySeed[1] {
		switch {
		case r.Signature == "INCONCLUSIVE|gap:tool_results" && r.ReplaySignature != "PASS|":
			t.Fatalf("clean behavior replays to %q, want PASS|", r.ReplaySignature)
		case r.Signature == "FAIL|viol:schema,gap:tool_results" && r.ReplaySignature != "FAIL|viol:schema":
			t.Fatalf("faulty behavior replays to %q", r.ReplaySignature)
		case !r.Gaps && r.ReplaySignature != "":
			t.Fatalf("complete run has a replay: %+v", r)
		}
	}
}

func TestDecideCommand(t *testing.T) {
	outFile := filepath.Join(t.TempDir(), "decide.json")
	text, err := runVerify(t, "decide", "--results", writeArtifact(t), "--rotate",
		"--sweep-miss", "10,100", "--output", outFile)
	if err != nil {
		t.Fatalf("decide: %v\n%s", err, text)
	}
	for _, want := range []string{"rules", "judge-first", "bayes", "mean over 30 seed rotations"} {
		if !strings.Contains(text, want) {
			t.Errorf("output lacks %q:\n%s", want, text)
		}
	}
	raw, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatal(err)
	}
	var out decideOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	if out.FitRuns != 40 || out.CalRuns != 20 || out.TestRuns != 40 || len(out.Sweep) != 2 {
		t.Fatalf("split sizes %d/%d/%d, sweep rows %d", out.FitRuns, out.CalRuns, out.TestRuns, len(out.Sweep))
	}
	byName := map[string]float64{}
	for _, m := range out.Main.Policies {
		byName[m.Policy] = m.CostPerRun
		if m.AcceptedFaulty > out.Alpha {
			t.Errorf("%s accepted faulty at %v > alpha", m.Policy, m.AcceptedFaulty)
		}
	}
	if byName["oracle"] != 0 || byName["bayes"] >= byName["rules"] {
		t.Fatalf("costs per run = %v: bayes should beat the rules when replays resolve gaps", byName)
	}
	if out.Rotation == nil || out.Rotation.Splits != 30 {
		t.Fatalf("rotation = %+v", out.Rotation)
	}
}

func TestDecideCommandRejectsEmptySplit(t *testing.T) {
	if _, err := runVerify(t, "decide", "--results", writeArtifact(t), "--test-seeds", "9"); err == nil {
		t.Fatal("a split with no runs must fail")
	}
}
