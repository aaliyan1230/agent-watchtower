package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCanonicalCommandEmitsStableChecksumAndSerializations(t *testing.T) {
	out, err := runVerify(t, "canonical",
		"--trace-file", "../../testdata/clean_trace.json",
		"--topo-limit", "20")
	if err != nil {
		t.Fatalf("canonical: %v\n%s", err, out)
	}
	var c struct {
		TraceID        string   `json:"traceId"`
		Closed         bool     `json:"closed"`
		Checksum       string   `json:"checksum"`
		Render         string   `json:"render"`
		TopoLimit      int      `json:"topoLimit"`
		Serializations []string `json:"serializations"`
	}
	if err := json.Unmarshal([]byte(out), &c); err != nil {
		t.Fatalf("output is not JSON: %v\n%s", err, out)
	}
	if c.TraceID != "trace-clean" {
		t.Errorf("traceId = %q", c.TraceID)
	}
	if len(c.Checksum) != 64 {
		t.Errorf("checksum length = %d, want 64", len(c.Checksum))
	}
	// DAG is s0->s1->s2 and s0->s3: s1 and s3 are independent siblings,
	// so there are 3 valid serializations (s2 must follow s1).
	if len(c.Serializations) != 3 {
		t.Fatalf("serializations = %v, want three", c.Serializations)
	}
	if c.Serializations[0] != "s0,s1,s2,s3" {
		t.Errorf("first serialization = %q, want s0,s1,s2,s3", c.Serializations[0])
	}
	if !strings.Contains(c.Render, "s0 > s1") || !strings.Contains(c.Render, "s1 > s2") || !strings.Contains(c.Render, "s0 > s3") {
		t.Errorf("render missing parent edges:\n%s", c.Render)
	}
}

func TestCanonicalCommandMissingFile(t *testing.T) {
	if _, err := runVerify(t, "canonical", "--trace-file", "nope.json"); err == nil {
		t.Fatal("expected error for missing trace file")
	}
}
