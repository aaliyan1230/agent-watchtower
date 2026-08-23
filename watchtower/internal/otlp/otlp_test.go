package otlp

import (
	"encoding/hex"
	"testing"
	"time"

	collectorpb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	resourcepb "go.opentelemetry.io/proto/otlp/resource/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/proto"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

// buildRequest assembles an ExportTraceServiceRequest the way the OTel
// SDK would: a resource, a scope, and spans nested under it.
func buildRequest(spans []*tracepb.Span) []byte {
	req := &collectorpb.ExportTraceServiceRequest{
		ResourceSpans: []*tracepb.ResourceSpans{{
			Resource: &resourcepb.Resource{
				Attributes: []*commonpb.KeyValue{
					{Key: "service.name", Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: "harness"}}},
				},
			},
			ScopeSpans: []*tracepb.ScopeSpans{{Spans: spans}},
		}},
	}
	raw, err := proto.Marshal(req)
	if err != nil {
		panic(err)
	}
	return raw
}

func protoSpan(name string, id, parent []byte) *tracepb.Span {
	return &tracepb.Span{
		TraceId:           []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10},
		SpanId:            id,
		ParentSpanId:      parent,
		Name:              name,
		Kind:              tracepb.Span_SPAN_KIND_CLIENT,
		StartTimeUnixNano: uint64(time.Date(2026, 8, 10, 10, 0, 0, 0, time.UTC).UnixNano()),
		EndTimeUnixNano:   uint64(time.Date(2026, 8, 10, 10, 0, 1, 0, time.UTC).UnixNano()),
		Status:            &tracepb.Status{Code: tracepb.Status_STATUS_CODE_OK},
		Attributes: []*commonpb.KeyValue{
			{Key: "gen_ai.system", Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: "fake"}}},
			{Key: "gen_ai.usage.input_tokens", Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_IntValue{IntValue: 42}}},
			{Key: "tool.result.ok", Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_BoolValue{BoolValue: true}}},
		},
		Events: []*tracepb.Span_Event{{
			Name:         "tool.result",
			TimeUnixNano: uint64(time.Date(2026, 8, 10, 10, 0, 0, 500*int(time.Millisecond), time.UTC).UnixNano()),
			Attributes:   []*commonpb.KeyValue{{Key: "status", Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: "done"}}}},
		}},
		Links: []*tracepb.Span_Link{{
			SpanId: []byte{0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88},
			Attributes: []*commonpb.KeyValue{{
				Key:   "watchtower.link.purpose",
				Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: "data"}},
			}},
		}},
	}
}

func TestDecodeHappyPath(t *testing.T) {
	id := []byte{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11}
	env, err := Decode(buildRequest([]*tracepb.Span{protoSpan("agent.run", id, nil)}))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if env.Resource["service.name"] != "harness" {
		t.Errorf("resource = %v", env.Resource)
	}
	if len(env.Spans) != 1 {
		t.Fatalf("spans = %d, want 1", len(env.Spans))
	}
	s := env.Spans[0]
	if s.TraceID != "0102030405060708090a0b0c0d0e0f10" {
		t.Errorf("traceId = %q", s.TraceID)
	}
	if s.SpanID != "aabbccddeeff0011" {
		t.Errorf("spanId = %q", s.SpanID)
	}
	if s.ParentID != "" {
		t.Errorf("parentSpanId = %q, want empty for zero parent", s.ParentID)
	}
	if s.Kind != "client" {
		t.Errorf("kind = %q, want client", s.Kind)
	}
	if s.Name != "agent.run" || s.Status != model.StatusOK {
		t.Errorf("name/status = %q/%q", s.Name, s.Status)
	}
	if s.StartTime.Unix() != time.Date(2026, 8, 10, 10, 0, 0, 0, time.UTC).Unix() {
		t.Errorf("start = %v", s.StartTime)
	}
	if s.Attributes["gen_ai.usage.input_tokens"] != "42" || s.Attributes["tool.result.ok"] != "true" {
		t.Errorf("attrs = %v", s.Attributes)
	}
	if len(s.Events) != 1 || s.Events[0].Name != "tool.result" || s.Events[0].Attributes["status"] != "done" {
		t.Errorf("events = %+v", s.Events)
	}
	if len(s.Links) != 1 || s.Links[0].SpanID != "1122334455667788" || s.Links[0].Attributes["watchtower.link.purpose"] != "data" {
		t.Errorf("links = %+v", s.Links)
	}
	if err := env.Validate(); err != nil {
		t.Fatalf("decoded envelope should validate: %v", err)
	}
}

func TestDecodeParentIDAndErrorStatus(t *testing.T) {
	id := []byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01}
	parent := []byte{0xaa, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xbb}
	sp := protoSpan("chat", id, parent)
	sp.Status = &tracepb.Status{Code: tracepb.Status_STATUS_CODE_ERROR, Message: "kaboom"}
	env, err := Decode(buildRequest([]*tracepb.Span{sp}))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	s := env.Spans[0]
	if s.ParentID != "aa000000000000bb" {
		t.Errorf("parentSpanId = %q", s.ParentID)
	}
	if s.Status != model.StatusError || s.StatusMsg != "kaboom" {
		t.Errorf("status = %q/%q", s.Status, s.StatusMsg)
	}
}

func TestDecodeFlattensMultipleScopes(t *testing.T) {
	spans := []*tracepb.Span{
		protoSpan("one", []byte{0x01, 0, 0, 0, 0, 0, 0, 0}, nil),
		protoSpan("two", []byte{0x02, 0, 0, 0, 0, 0, 0, 0}, nil),
	}
	raw := buildRequest(spans)
	// split the two spans across two scope groups to prove flattening
	var req collectorpb.ExportTraceServiceRequest
	if err := proto.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	req.ResourceSpans[0].ScopeSpans = []*tracepb.ScopeSpans{
		{Spans: spans[:1]},
		{Spans: spans[1:]},
	}
	split, err := proto.Marshal(&req)
	if err != nil {
		t.Fatal(err)
	}
	env, err := Decode(split)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(env.Spans) != 2 {
		t.Fatalf("spans = %d, want 2 (flattened)", len(env.Spans))
	}
}

func TestDecodeGarbage(t *testing.T) {
	if _, err := Decode([]byte("not protobuf")); err == nil {
		t.Fatal("Decode() on garbage should fail")
	}
}

func TestHexRoundTrip(t *testing.T) {
	b := []byte{0xde, 0xad, 0xbe, 0xef}
	got, err := hex.DecodeString(hexOrEmpty(b, false))
	if err != nil || string(got) != string(b) {
		t.Fatalf("hex round trip = %x, %v", got, err)
	}
}
