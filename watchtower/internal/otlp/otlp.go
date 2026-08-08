// Package otlp is the Phase 0.6 wire-format adapter: it decodes real
// OTLP/HTTP protobuf (an ExportTraceServiceRequest) into the flat span
// envelope the rest of the pipeline consumes. The pipeline never sees
// protobuf; the wire format is an interface, and this package is the
// implementation behind the /v1/traces endpoint.
package otlp

import (
	"fmt"
	"strconv"
	"time"

	collectorpb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/proto"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

// Decode parses an OTLP/HTTP protobuf payload into an Envelope.
// ResourceSpans is collapsed: the first resource's attributes become
// the envelope resource, all spans (across resources and scopes) are
// flattened — the pipeline's model is intentionally flatter than OTLP.
func Decode(raw []byte) (*model.Envelope, error) {
	var req collectorpb.ExportTraceServiceRequest
	if err := proto.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("otlp: unmarshal ExportTraceServiceRequest: %w", err)
	}
	env := &model.Envelope{}
	for _, rs := range req.ResourceSpans {
		if rs.Resource != nil && env.Resource == nil {
			env.Resource = attrsToStrings(rs.Resource.Attributes)
		}
		for _, ss := range rs.ScopeSpans {
			for _, sp := range ss.Spans {
				env.Spans = append(env.Spans, spanToModel(sp))
			}
		}
	}
	return env, nil
}

// spanToModel converts one OTLP span. OTLP ids are fixed-length byte
// slices; the wire envelope carries them as hex strings. A zero parent
// span id means "root" in OTLP, so it maps to an empty ParentID.
func spanToModel(sp *tracepb.Span) model.Span {
	s := model.Span{
		TraceID:    hexOrEmpty(sp.TraceId, false),
		SpanID:     hexOrEmpty(sp.SpanId, false),
		ParentID:   hexOrEmpty(sp.ParentSpanId, true),
		Name:       sp.Name,
		Kind:       kindName(sp.Kind),
		StartTime:  unixNano(sp.StartTimeUnixNano),
		EndTime:    unixNano(sp.EndTimeUnixNano),
		Attributes: attrsToStrings(sp.Attributes),
	}
	if sp.Status != nil {
		switch sp.Status.Code {
		case tracepb.Status_STATUS_CODE_OK:
			s.Status = model.StatusOK
		case tracepb.Status_STATUS_CODE_ERROR:
			s.Status = model.StatusError
		default:
			s.Status = model.StatusUnset
		}
		s.StatusMsg = sp.Status.Message
	}
	for _, e := range sp.Events {
		s.Events = append(s.Events, model.Event{
			Name:       e.Name,
			Time:       unixNano(e.TimeUnixNano),
			Attributes: attrsToStrings(e.Attributes),
		})
	}
	return s
}

func hexOrEmpty(b []byte, zeroIsEmpty bool) string {
	if zeroIsEmpty && len(b) > 0 {
		allZero := true
		for _, x := range b {
			if x != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			return ""
		}
	}
	return fmt.Sprintf("%x", b)
}

func unixNano(ns uint64) time.Time {
	return time.Unix(0, int64(ns)).UTC()
}

func kindName(k tracepb.Span_SpanKind) string {
	switch k {
	case tracepb.Span_SPAN_KIND_INTERNAL:
		return "internal"
	case tracepb.Span_SPAN_KIND_SERVER:
		return "server"
	case tracepb.Span_SPAN_KIND_CLIENT:
		return "client"
	case tracepb.Span_SPAN_KIND_PRODUCER:
		return "producer"
	case tracepb.Span_SPAN_KIND_CONSUMER:
		return "consumer"
	default:
		return "internal" // UNSET: treat as internal, matching OTLP semantics
	}
}

// attrsToStrings flattens OTLP AnyValues into strings, matching the
// model's flat string-attribute design. Scalars map 1:1; arrays and
// key-value lists are best-effort stringified — the semconv attributes
// Watchtower reads (token counts, tool names, flags) are all scalars.
func attrsToStrings(kvs []*commonpb.KeyValue) map[string]string {
	if len(kvs) == 0 {
		return nil
	}
	out := make(map[string]string, len(kvs))
	for _, kv := range kvs {
		out[kv.Key] = stringify(anyValueToAny(kv.Value))
	}
	return out
}

func anyValueToAny(v *commonpb.AnyValue) any {
	switch vv := v.Value.(type) {
	case *commonpb.AnyValue_StringValue:
		return vv.StringValue
	case *commonpb.AnyValue_IntValue:
		return vv.IntValue
	case *commonpb.AnyValue_DoubleValue:
		return vv.DoubleValue
	case *commonpb.AnyValue_BoolValue:
		return vv.BoolValue
	case *commonpb.AnyValue_BytesValue:
		return string(vv.BytesValue)
	case *commonpb.AnyValue_ArrayValue:
		out := make([]any, 0, len(vv.ArrayValue.Values))
		for _, e := range vv.ArrayValue.Values {
			out = append(out, anyValueToAny(e))
		}
		return out
	case *commonpb.AnyValue_KvlistValue:
		out := make(map[string]any, len(vv.KvlistValue.Values))
		for _, kv := range vv.KvlistValue.Values {
			out[kv.Key] = anyValueToAny(kv.Value)
		}
		return out
	}
	return nil
}

// stringify keeps floats tidy ("42" not "42.0") by using the same
// formatting Go's fmt does for JSON-ish scalars: integers via FormatInt,
// floats via 'g' with minimal digits.
func stringify(v any) string {
	switch vv := v.(type) {
	case string:
		return vv
	case int64:
		return strconv.FormatInt(vv, 10)
	case float64:
		return strconv.FormatFloat(vv, 'g', -1, 64)
	case bool:
		return strconv.FormatBool(vv)
	default:
		return fmt.Sprintf("%v", vv)
	}
}
