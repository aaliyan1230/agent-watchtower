package model

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// baseSpan returns a valid span with distinct ids; tests mutate it.
func baseSpan(id string) Span {
	return Span{
		TraceID:   "trace-1",
		SpanID:    id,
		Name:      "agent.run",
		StartTime: time.Unix(0, 0).UTC(),
		EndTime:   time.Unix(0, 1).UTC(),
	}
}

func TestSpanValidate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Span)
		wantErr string
	}{
		{name: "valid", mutate: func(s *Span) {}},
		{name: "missing trace id", mutate: func(s *Span) { s.TraceID = "" }, wantErr: "traceId is required"},
		{name: "missing span id", mutate: func(s *Span) { s.SpanID = "" }, wantErr: "spanId is required"},
		{name: "missing name", mutate: func(s *Span) { s.Name = "" }, wantErr: "name is required"},
		{name: "zero timestamps", mutate: func(s *Span) { s.StartTime, s.EndTime = time.Time{}, time.Time{} }, wantErr: "startTime and endTime are required"},
		{name: "end before start", mutate: func(s *Span) { s.EndTime = s.StartTime.Add(-time.Second) }, wantErr: "endTime before startTime"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := baseSpan("s1")
			tt.mutate(&s)
			err := s.Validate()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Validate() = %v, want error containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestSpanAttrGetters(t *testing.T) {
	s := baseSpan("s1")
	s.Attributes = map[string]string{
		GenAIInputTokens: "42",
		ToolResultOK:     "true",
	}
	if v, ok := s.Attr(GenAIInputTokens); !ok || v != "42" {
		t.Fatalf("Attr() = %q, %v; want %q, true", v, ok, "42")
	}
	if n, ok := s.AttrInt(GenAIInputTokens); !ok || n != 42 {
		t.Fatalf("AttrInt() = %d, %v; want 42, true", n, ok)
	}
	if b, ok := s.AttrBool(ToolResultOK); !ok || !b {
		t.Fatalf("AttrBool() = %v, %v; want true, true", b, ok)
	}
	if _, ok := s.Attr("missing"); ok {
		t.Fatal("Attr() on missing attribute should be false")
	}
	if _, ok := s.AttrInt("notanumber"); ok {
		t.Fatal("AttrInt() on non-numeric attribute should be false")
	}
	if _, ok := s.AttrBool("notabool"); ok {
		t.Fatal("AttrBool() on non-bool attribute should be false")
	}
}

func TestEnvelopeJSONRoundTrip(t *testing.T) {
	env := Envelope{
		Resource: map[string]string{"service.name": "demo"},
		Spans: []Span{
			baseSpan("s1"),
			{
				TraceID: "trace-1", SpanID: "s2", ParentID: "s1",
				Name: "tool.call", Kind: "client",
				StartTime: time.Unix(10, 0).UTC(), EndTime: time.Unix(11, 0).UTC(),
				Status: StatusError, StatusMsg: "timeout",
				Attributes: map[string]string{ToolName: "search"},
				Events:     []Event{{Name: "tool.result", Time: time.Unix(10, 5).UTC()}},
			},
		},
	}
	raw, err := json.Marshal(env)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Envelope
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(got.Spans) != 2 || got.Spans[1].StatusMsg != "timeout" {
		t.Fatalf("round trip lost data: %+v", got.Spans)
	}
	if err := got.Validate(); err != nil {
		t.Fatalf("Validate() after round trip: %v", err)
	}
}
