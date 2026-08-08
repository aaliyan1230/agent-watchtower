package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	collectorpb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/proto"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/report"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

// minimalOTLP builds a one-span OTLP request the way the harness sends
// it: one root agent span with semconv attributes.
func minimalOTLP(t *testing.T, attrs map[string]string) []byte {
	t.Helper()
	var kvs []*commonpb.KeyValue
	for k, v := range attrs {
		kvs = append(kvs, &commonpb.KeyValue{Key: k, Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: v}}})
	}
	span := &tracepb.Span{
		TraceId:           []byte{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8},
		SpanId:            []byte{9, 9, 9, 9, 9, 9, 9, 9},
		Name:              "agent.run",
		Kind:              tracepb.Span_SPAN_KIND_INTERNAL,
		StartTimeUnixNano: uint64(time.Date(2026, 8, 10, 10, 0, 0, 0, time.UTC).UnixNano()),
		EndTimeUnixNano:   uint64(time.Date(2026, 8, 10, 10, 0, 1, 0, time.UTC).UnixNano()),
		Status:            &tracepb.Status{Code: tracepb.Status_STATUS_CODE_OK},
		Attributes:        kvs,
	}
	req := &collectorpb.ExportTraceServiceRequest{
		ResourceSpans: []*tracepb.ResourceSpans{{
			ScopeSpans: []*tracepb.ScopeSpans{{Spans: []*tracepb.Span{span}}},
		}},
	}
	raw, err := proto.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestServeAcceptsOTLPAndExposesReport(t *testing.T) {
	cfg := verify.Config{
		Policy: verify.Policy{AllowedTools: []string{"search"}},
		Limits: verify.Limits{MaxSteps: 10},
	}
	mux := newServeMux(cfg, report.NewStore(time.Minute))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	raw := minimalOTLP(t, map[string]string{"agent.name": "worker-a"})
	resp, err := http.Post(srv.URL+"/v1/traces", "application/x-protobuf", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/x-protobuf" {
		t.Errorf("content type = %q", ct)
	}
	// The body must parse as the OTLP response proto — the official
	// exporter parses it; garbage here fails the export.
	var oresp collectorpb.ExportTraceServiceResponse
	buf := new(bytes.Buffer)
	_, _ = buf.ReadFrom(resp.Body)
	if err := proto.Unmarshal(buf.Bytes(), &oresp); err != nil {
		t.Fatalf("response is not an ExportTraceServiceResponse: %v", err)
	}

	repResp, err := http.Get(srv.URL + "/v1/reports/01020304050607080102030405060708")
	if err != nil {
		t.Fatal(err)
	}
	defer repResp.Body.Close()
	if repResp.StatusCode != http.StatusOK {
		t.Fatalf("report status = %d, want 200", repResp.StatusCode)
	}
	var r map[string]any
	if err := json.NewDecoder(repResp.Body).Decode(&r); err != nil {
		t.Fatal(err)
	}
	if r["verdict"] != "PASS" {
		t.Fatalf("verdict = %v, want PASS", r["verdict"])
	}
}

func TestServeReportNotFound(t *testing.T) {
	mux := newServeMux(verify.Config{}, report.NewStore(time.Minute))
	srv := httptest.NewServer(mux)
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/v1/reports/deadbeef")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}
}
