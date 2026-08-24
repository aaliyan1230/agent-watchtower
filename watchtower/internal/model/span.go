// Package model defines the span wire model Watchtower ingests and the
// OTel GenAI semantic-convention attributes it understands. The JSON
// fields deliberately mirror OTLP span fields (traceId/spanId/parentSpanId,
// kind, timestamps, status, attributes, events) so that real OTLP/HTTP
// protobuf ingestion (Phase 0.6) can swap in behind the same struct
// without touching the rest of the pipeline.
package model

import (
	"fmt"
	"strconv"
	"time"
)

// StatusCode mirrors OTLP's span status as strings so the JSON wire
// format reads naturally (an int enum would need a lookup table on both
// sides of the wire).
type StatusCode string

const (
	StatusUnset StatusCode = "unset"
	StatusOK    StatusCode = "ok"
	StatusError StatusCode = "error"
)

// Event is a timestamped annotation on a span, e.g. a tool result or a
// structured-output parse failure. Verifiers use these as evidence.
type Event struct {
	Name       string            `json:"name"`
	Time       time.Time         `json:"time"`
	Attributes map[string]string `json:"attributes,omitempty"`
}

// Link is a causal reference from this span to another span that it
// depends on — the OTLP span-link, used as the edge carrier for
// concurrent runs. A non-parent dependency is encoded as destination →
// source: the link's SpanID is the cause, the carrying span is the
// effect. Concurrent causality is what makes two valid orderings of the
// same run equivalent, so links are reconstructed into explicit causal
// edges (rather than a guessed order) before verification.
type Link struct {
	TraceID    string            `json:"traceId,omitempty"`
	SpanID     string            `json:"spanId"`
	Attributes map[string]string `json:"attributes,omitempty"`
}

// Span is the unit of ingestion: one agent action (an LLM call, a tool
// call, or an agent's lifecycle) recorded with OTel GenAI semconv
// attributes. Attributes are flat key/value strings, matching how OTLP
// encodes semconv attributes as key->AnyValue pairs.
type Span struct {
	TraceID    string            `json:"traceId"`
	SpanID     string            `json:"spanId"`
	ParentID   string            `json:"parentSpanId,omitempty"`
	Name       string            `json:"name"`
	Kind       string            `json:"kind,omitempty"`
	StartTime  time.Time         `json:"startTime"`
	EndTime    time.Time         `json:"endTime"`
	Status     StatusCode        `json:"status,omitempty"`
	StatusMsg  string            `json:"statusMessage,omitempty"`
	Attributes map[string]string `json:"attributes,omitempty"`
	Events     []Event           `json:"events,omitempty"`
	Links      []Link            `json:"links,omitempty"`
}

// Attr returns the value of an attribute, reporting whether it existed.
// Getters live on the span so verifiers can read semconv attributes
// without string-indexing into maps everywhere.
func (s *Span) Attr(name string) (string, bool) {
	if s.Attributes == nil {
		return "", false
	}
	v, ok := s.Attributes[name]
	return v, ok
}

// AttrInt parses an integer attribute (e.g. token counts, which OTLP
// sends as ints). Returns false when missing or unparsable.
func (s *Span) AttrInt(name string) (int64, bool) {
	v, ok := s.Attr(name)
	if !ok {
		return 0, false
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return 0, false
	}
	return n, true
}

// AttrBool parses a boolean attribute.
func (s *Span) AttrBool(name string) (bool, bool) {
	v, ok := s.Attr(name)
	if !ok {
		return false, false
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return false, false
	}
	return b, true
}

// Validate checks the fields the rest of the pipeline depends on. Ingest
// runs this on every span so a bad span fails loudly at the edge instead
// of producing a half-broken run graph downstream.
func (s *Span) Validate() error {
	if s.TraceID == "" {
		return fmt.Errorf("span %q: traceId is required", s.SpanID)
	}
	if s.SpanID == "" {
		return fmt.Errorf("trace %s: spanId is required", s.TraceID)
	}
	if s.Name == "" {
		return fmt.Errorf("span %s: name is required", s.SpanID)
	}
	if s.StartTime.IsZero() || s.EndTime.IsZero() {
		return fmt.Errorf("span %s: startTime and endTime are required", s.SpanID)
	}
	if s.EndTime.Before(s.StartTime) {
		return fmt.Errorf("span %s: endTime before startTime", s.SpanID)
	}
	return nil
}

// Envelope is the JSON body of an ingest request: a resource descriptor
// plus a flat list of spans, mirroring the OTLP ExportTraceServiceRequest
// hierarchy (resource -> scope -> spans) without the scope layer, which
// we don't need yet.
type Envelope struct {
	Resource map[string]string `json:"resource,omitempty"`
	Spans    []Span            `json:"spans"`
}

// Validate runs Span.Validate on every span in the envelope.
func (e *Envelope) Validate() error {
	for i := range e.Spans {
		if err := e.Spans[i].Validate(); err != nil {
			return err
		}
	}
	return nil
}
