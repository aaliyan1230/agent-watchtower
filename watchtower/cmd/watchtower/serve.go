package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/spf13/cobra"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/ingest"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/report"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

// newServeCmd runs the ingest endpoint; every submitted envelope is
// reconstructed and verified immediately, and the report is returned in
// the HTTP response — a live verdict per trace.
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
			h := ingest.New(func(env *model.Envelope) ([]byte, error) {
				run, err := graph.Reconstruct(env.Spans)
				if err != nil {
					return nil, fmt.Errorf("reconstruct: %w", err)
				}
				r := report.Build(run, verify.Verify(run, cfg), cfg.Judge != nil)
				log.Printf("trace %s: %s (%d findings)", r.TraceID, r.Verdict, len(r.Findings))
				return json.Marshal(r)
			})
			log.Printf("watchtower listening on %s", addr)
			return http.ListenAndServe(addr, h)
		},
	}
	cmd.Flags().StringVar(&addr, "addr", ":4318", "listen address")
	cmd.Flags().StringVar(&configFile, "config", "", "path to a verifier config JSON file (optional)")
	return cmd
}
