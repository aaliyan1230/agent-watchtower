package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func runVerify(t *testing.T, args ...string) (string, error) {
	t.Helper()
	root := NewRootCmd()
	var buf bytes.Buffer
	root.SetOut(&buf)
	root.SetErr(&buf)
	root.SetArgs(args)
	err := root.Execute()
	return buf.String(), err
}

func TestVerifyCommandCleanTrace(t *testing.T) {
	out, err := runVerify(t, "verify",
		"--trace-file", "../../testdata/clean_trace.json",
		"--config", "../../testdata/config.json")
	if err != nil {
		t.Fatalf("verify: %v\n%s", err, out)
	}
	var r map[string]any
	if err := json.Unmarshal([]byte(out), &r); err != nil {
		t.Fatalf("output is not JSON: %v\n%s", err, out)
	}
	if r["verdict"] != "PASS" {
		t.Fatalf("verdict = %v, want PASS\n%s", r["verdict"], out)
	}
	if r["traceId"] != "trace-clean" {
		t.Errorf("traceId = %v", r["traceId"])
	}
}

func TestVerifyCommandFaultyTrace(t *testing.T) {
	out, err := runVerify(t, "verify",
		"--trace-file", "../../testdata/faulty_trace.json",
		"--config", "../../testdata/config.json")
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	var r map[string]any
	if err := json.Unmarshal([]byte(out), &r); err != nil {
		t.Fatalf("output is not JSON: %v", err)
	}
	if r["verdict"] != "FAIL" {
		t.Fatalf("verdict = %v, want FAIL\n%s", r["verdict"], out)
	}
	// Every fault type present in the trace must be detected.
	for _, want := range []string{"schema", "policy", "loop", "budget"} {
		if !strings.Contains(out, `"name": "`+want+`"`) {
			t.Errorf("report missing verifier %q:\n%s", want, out)
		}
	}
	// The policy finding must carry the offending value as evidence.
	if !strings.Contains(out, `"rm"`) {
		t.Errorf("policy finding should cite the offending tool:\n%s", out)
	}
}

func TestVerifyCommandNoConfigDisablesChecks(t *testing.T) {
	// Without a config, every check is disabled by design; a faulty
	// trace must then pass — verification is opt-in, not presumptuous.
	out, err := runVerify(t, "verify", "--trace-file", "../../testdata/faulty_trace.json")
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if !strings.Contains(out, `"verdict": "PASS"`) {
		t.Fatalf("verdict should be PASS with no config:\n%s", out)
	}
}

func TestVerifyCommandMissingFile(t *testing.T) {
	if _, err := runVerify(t, "verify", "--trace-file", "nope.json"); err == nil {
		t.Fatal("expected error for missing trace file")
	}
}
