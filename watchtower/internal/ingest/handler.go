// Package ingest exposes Watchtower's span ingestion endpoint: a small
// HTTP handler that decodes the JSON span envelope, validates it, and
// hands it to a caller-provided ingest function. It owns no state —
// the caller decides what happens per envelope (verify now, store for
// later, whatever Phase 0.6's real OTLP receiver will do).
package ingest

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

const maxBodyBytes = 10 << 20 // 10 MiB: a trace should never approach this

// Handler is the ingest HTTP handler. Ingest is called with each valid
// envelope; if it returns an error, the client gets a 500.
type Handler struct {
	Ingest func(*model.Envelope) error
}

func New(ingest func(*model.Envelope) error) *Handler {
	return &Handler{Ingest: ingest}
}

// ServeHTTP accepts POST /v1/traces with a JSON span envelope.
// Status codes: 202 accepted, 400 malformed/validation failure,
// 405 wrong method, 404 wrong path, 500 ingest backend failure.
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

	body := http.MaxBytesReader(w, r.Body, maxBodyBytes)
	dec := json.NewDecoder(body)

	var env model.Envelope
	if err := dec.Decode(&env); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}
	// Reject trailing garbage after the JSON document.
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		http.Error(w, "bad request: multiple JSON documents", http.StatusBadRequest)
		return
	}
	if err := env.Validate(); err != nil {
		http.Error(w, fmt.Sprintf("invalid spans: %v", err), http.StatusBadRequest)
		return
	}

	if h.Ingest == nil {
		http.Error(w, "ingest not configured", http.StatusInternalServerError)
		return
	}
	if err := h.Ingest(&env); err != nil {
		log.Printf("ingest: %v", err)
		http.Error(w, "internal ingest error", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusAccepted)
}
