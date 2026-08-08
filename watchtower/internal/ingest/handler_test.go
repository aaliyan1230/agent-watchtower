package ingest

import (
	"encoding/json"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

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
		name       string
		method     string
		path       string
		body       []byte
		ingestErr  error
		wantStatus int
		wantCalled bool
		wantBody   string
	}{
		{name: "valid envelope accepted", method: "POST", path: "/v1/traces", body: validEnvelopeJSON(t), wantStatus: 202, wantCalled: true, wantBody: `"accepted"`},
		{name: "malformed json", method: "POST", path: "/v1/traces", body: []byte("{nope"), wantStatus: 400},
		{name: "invalid span", method: "POST", path: "/v1/traces", body: []byte(`{"spans":[{"spanId":"x"}]}`), wantStatus: 400},
		{name: "trailing garbage", method: "POST", path: "/v1/traces", body: append(validEnvelopeJSON(t), []byte(`{}`)...), wantStatus: 400},
		{name: "wrong method", method: "GET", path: "/v1/traces", body: nil, wantStatus: 405},
		{name: "wrong path", method: "POST", path: "/other", body: validEnvelopeJSON(t), wantStatus: 404},
		{name: "backend error", method: "POST", path: "/v1/traces", body: validEnvelopeJSON(t), ingestErr: errors.New("boom"), wantStatus: 500, wantCalled: true},
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
