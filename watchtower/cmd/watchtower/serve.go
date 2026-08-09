package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/spf13/cobra"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/ingest"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/report"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

// newServeCmd runs the ingest endpoint; every submitted envelope is
// reconstructed and verified immediately, the verdict is stored per
// trace id (GET /v1/reports/{traceId}) and, for JSON clients, returned
// in the response.
func newServeCmd() *cobra.Command {
	var addr, configFile string
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "serve the span ingestion endpoint on :4318",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig(configFile)
			if err != nil {
				return err
			}
			cfg.Judge, err = buildJudge(cfg)
			if err != nil {
				return err
			}
			log.Printf("watchtower listening on %s", addr)
			return http.ListenAndServe(addr, newServeMux(cfg, report.NewStore(5*time.Minute)))
		},
	}
	cmd.Flags().StringVar(&addr, "addr", ":4318", "listen address")
	cmd.Flags().StringVar(&configFile, "config", "", "path to a verifier config JSON file (optional)")
	return cmd
}

// newServeMux wires ingest -> verify -> report store -> retrieval
// endpoint. Split out so tests can mount it in an httptest server.
func newServeMux(cfg verify.Config, store *report.Store) http.Handler {
	h := ingest.New(func(env *model.Envelope) ([]byte, error) {
		run, err := graph.Reconstruct(env.Spans)
		if err != nil {
			return nil, fmt.Errorf("reconstruct: %w", err)
		}
		r := report.Build(run, verify.Verify(run, cfg), cfg.Judge != nil)
		store.Put(r.TraceID, r)
		log.Printf("trace %s: %s (%d findings)", r.TraceID, r.Verdict, len(r.Findings))
		return json.Marshal(r)
	})

	mux := http.NewServeMux()
	mux.Handle("/v1/traces", h)
	mux.HandleFunc("/v1/reports/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		traceID := r.URL.Path[len("/v1/reports/"):]
		rep, ok := store.Get(traceID)
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(rep)
	})
	return mux
}
