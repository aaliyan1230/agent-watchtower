package ingest

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	collectorpb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/proto"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

func validEnvelopeJSON(t *testing.T) []byte {
	t.Helper()
	env := model.Envelope{Spans: []model.Span{{
		TraceID: "t1", SpanID: "s1", Name: "agent.run",
		StartTime: time.Now(), EndTime: time.Now().Add(time.Second),
	}}}
	raw, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestIngest(t *testing.T) {
	tests := []struct {
		name        string
		method      string
		path        string
		contentType string
		body        []byte
		ingestErr   error
		wantStatus  int
		wantCalled  bool
		wantBody    string
	}{
		{name: "valid envelope accepted", method: "POST", path: "/v1/traces", contentType: "application/json", body: validEnvelopeJSON(t), wantStatus: 202, wantCalled: true, wantBody: `"accepted"`},
		{name: "malformed json", method: "POST", path: "/v1/traces", contentType: "application/json", body: []byte("{nope"), wantStatus: 400},
		{name: "invalid span", method: "POST", path: "/v1/traces", contentType: "application/json", body: []byte(`{"spans":[{"spanId":"x"}]}`), wantStatus: 400},
		{name: "trailing garbage", method: "POST", path: "/v1/traces", contentType: "application/json", body: append(validEnvelopeJSON(t), []byte(`{}`)...), wantStatus: 400},
		{name: "missing content type", method: "POST", path: "/v1/traces", body: validEnvelopeJSON(t), wantStatus: 415},
		{name: "unsupported content type", method: "POST", path: "/v1/traces", contentType: "text/plain", body: []byte("hi"), wantStatus: 415},
		{name: "wrong method", method: "GET", path: "/v1/traces", contentType: "application/json", wantStatus: 405},
		{name: "wrong path", method: "POST", path: "/other", contentType: "application/json", body: validEnvelopeJSON(t), wantStatus: 404},
		{name: "backend error", method: "POST", path: "/v1/traces", contentType: "application/json", body: validEnvelopeJSON(t), ingestErr: errors.New("boom"), wantStatus: 500, wantCalled: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			h := New(func(env *model.Envelope) ([]byte, error) {
				called = true
				if len(env.Spans) != 1 {
					t.Errorf("ingest received %d spans, want 1", len(env.Spans))
				}
				return []byte(`"accepted"`), tt.ingestErr
			})
			req := httptest.NewRequest(tt.method, tt.path, strings.NewReader(string(tt.body)))
			if tt.contentType != "" {
				req.Header.Set("Content-Type", tt.contentType)
			}
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d (body %s)", rec.Code, tt.wantStatus, rec.Body.String())
			}
			if called != tt.wantCalled {
				t.Fatalf("ingest called = %v, want %v", called, tt.wantCalled)
			}
			if tt.wantBody != "" && !strings.Contains(rec.Body.String(), tt.wantBody) {
				t.Fatalf("body = %q, want it to contain %q", rec.Body.String(), tt.wantBody)
			}
		})
	}
}

func TestIngestOTLPProtobuf(t *testing.T) {
	span := &tracepb.Span{
		TraceId:           []byte{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8},
		SpanId:            []byte{9, 9, 9, 9, 9, 9, 9, 9},
		Name:              "agent.run",
		StartTimeUnixNano: uint64(time.Now().Add(-time.Second).UnixNano()),
		EndTimeUnixNano:   uint64(time.Now().UnixNano()),
		Status:            &tracepb.Status{Code: tracepb.Status_STATUS_CODE_OK},
		Attributes: []*commonpb.KeyValue{
			{Key: "agent.name", Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: "worker-a"}}},
		},
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

	called := false
	h := New(func(env *model.Envelope) ([]byte, error) {
		called = true
		if len(env.Spans) != 1 {
			t.Fatalf("spans = %d, want 1", len(env.Spans))
		}
		if env.Spans[0].Attributes["agent.name"] != "worker-a" {
			t.Errorf("attrs = %v", env.Spans[0].Attributes)
		}
		return nil, nil
	})
	req2 := httptest.NewRequest(http.MethodPost, "/v1/traces", strings.NewReader(string(raw)))
	req2.Header.Set("Content-Type", "application/x-protobuf")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req2)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", rec.Code, rec.Body.String())
	}
	if !called {
		t.Fatal("handle not called")
	}
	// The response body must parse as ExportTraceServiceResponse: the
	// official OTLP exporter decodes it, so it cannot be our report.
	var oresp collectorpb.ExportTraceServiceResponse
	if err := proto.Unmarshal(rec.Body.Bytes(), &oresp); err != nil {
		t.Fatalf("response is not OTLP: %v", err)
	}
}

func TestIngestOTLPGarbage(t *testing.T) {
	h := New(func(env *model.Envelope) ([]byte, error) { return nil, nil })
	req := httptest.NewRequest(http.MethodPost, "/v1/traces", strings.NewReader("not protobuf"))
	req.Header.Set("Content-Type", "application/x-protobuf")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}
