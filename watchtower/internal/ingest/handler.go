// Package ingest exposes Watchtower's span ingestion endpoint: an HTTP
// handler speaking two wire formats for the same pipeline:
//
//   - OTLP/HTTP protobuf (application/x-protobuf): real OTLP payloads,
//     answered with the OTLP response proto. This is the Phase 0.6 wire.
//   - the legacy JSON envelope (application/json): answered with the
//     Handle callback's payload (the report), kept for the CLI workflow
//     and backward compatibility.
//
// The handler owns no state — Handle decides what each envelope means
// (verify now, store, whatever).
package ingest

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	collectorpb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/protobuf/proto"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/otlp"
)

const maxBodyBytes = 10 << 20 // 10 MiB: a trace should never approach this

// Handler is the ingest HTTP handler. Handle is called with each valid
// envelope; its return value is written as the response body for JSON
// clients (nil means no body) — the caller decides what a trace *means*
// (a report, a receipt, nothing).
type Handler struct {
	Handle func(*model.Envelope) ([]byte, error)
}

func New(handle func(*model.Envelope) ([]byte, error)) *Handler {
	return &Handler{Handle: handle}
}

// ServeHTTP accepts POST /v1/traces. Status codes: 200 accepted (OTLP),
// 202 accepted (JSON), 400 malformed/invalid, 405 wrong method,
// 404 wrong path, 415 unsupported content type, 500 backend failure.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/v1/traces" {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Content-Type may carry parameters ("application/x-protobuf; charset=..."),
	// so take the media type before the ';'.
	mediaType := strings.TrimSpace(strings.Split(r.Header.Get("Content-Type"), ";")[0])
	isProtobuf := mediaType == "application/x-protobuf" || mediaType == "application/protobuf"

	body := http.MaxBytesReader(w, r.Body, maxBodyBytes)
	var env *model.Envelope
	var err error
	switch {
	case isProtobuf:
		raw, rerr := io.ReadAll(body)
		if rerr != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", rerr), http.StatusBadRequest)
			return
		}
		env, err = otlp.Decode(raw)
	case mediaType == "application/json":
		env, err = decodeJSON(body)
	default:
		http.Error(w, "unsupported media type: use application/x-protobuf or application/json", http.StatusUnsupportedMediaType)
		return
	}
	if err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}
	if err := env.Validate(); err != nil {
		http.Error(w, fmt.Sprintf("invalid spans: %v", err), http.StatusBadRequest)
		return
	}

	if h.Handle == nil {
		http.Error(w, "ingest not configured", http.StatusInternalServerError)
		return
	}
	payload, err := h.Handle(env)
	if err != nil {
		log.Printf("ingest: %v", err)
		http.Error(w, "internal ingest error", http.StatusInternalServerError)
		return
	}

	if isProtobuf {
		// OTLP clients parse the body as ExportTraceServiceResponse; an
		// empty response means "accepted". The verdict report is
		// retrievable from the report store instead of the wire.
		resp, merr := proto.Marshal(&collectorpb.ExportTraceServiceResponse{})
		if merr != nil {
			http.Error(w, "internal error", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/x-protobuf")
		// WriteHeader must precede Write: writing the body would
		// otherwise implicitly send 200 — same outcome, but explicit.
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(resp)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	if len(payload) > 0 {
		_, _ = w.Write(payload)
	}
}

// decodeJSON parses the legacy envelope, rejecting trailing garbage
// after the JSON document.
func decodeJSON(body io.Reader) (*model.Envelope, error) {
	dec := json.NewDecoder(body)
	var env model.Envelope
	if err := dec.Decode(&env); err != nil {
		return nil, err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return nil, fmt.Errorf("multiple JSON documents")
	}
	return &env, nil
}
