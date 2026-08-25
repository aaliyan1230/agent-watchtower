package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/canonical"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// newCanonicalCmd renders a trace envelope into its canonical causal-DAG
// form: a deterministic render, its stable checksum, and the bounded set
// of valid topological serializations. It is the Go reference for the
// Python experiment's order-invariance sweep — the same trace, no matter
// the arrival order, must produce the same checksum here.
func newCanonicalCmd() *cobra.Command {
	var traceFile, outFile string
	var topoLimit int
	cmd := &cobra.Command{
		Use:   "canonical",
		Short: "render a trace envelope as a canonical causal DAG",
		RunE: func(cmd *cobra.Command, args []string) error {
			env, err := readEnvelope(traceFile)
			if err != nil {
				return err
			}
			run, err := graph.Reconstruct(env.Spans)
			if err != nil {
				return fmt.Errorf("reconstruct: %w", err)
			}
			g := canonical.FromRun(run)
			orders := g.TopoOrders(topoLimit)
			serializations := make([]string, 0, len(orders))
			for _, order := range orders {
				serializations = append(serializations, joinIDs(order))
			}
			out := canonicalOutput{
				TraceID:        run.TraceID,
				Closed:         run.Closed,
				Checksum:       g.Checksum(),
				Render:         g.Render(),
				TopoLimit:      topoLimit,
				Serializations: serializations,
			}
			payload, err := json.MarshalIndent(out, "", "  ")
			if err != nil {
				return err
			}
			if outFile != "" {
				return os.WriteFile(outFile, payload, 0o644)
			}
			_, err = fmt.Fprintln(cmd.OutOrStdout(), string(payload))
			return err
		},
	}
	cmd.Flags().StringVar(&traceFile, "trace-file", "", "path to a span envelope JSON file (required)")
	cmd.Flags().StringVar(&outFile, "output", "", "write output to this file instead of stdout (optional)")
	cmd.Flags().IntVar(&topoLimit, "topo-limit", 20, "bound on the number of topological serializations")
	_ = cmd.MarkFlagRequired("trace-file")
	return cmd
}

func joinIDs(order []string) string {
	out := ""
	for i, id := range order {
		if i > 0 {
			out += ","
		}
		out += id
	}
	return out
}

// canonicalOutput is the JSON contract for `watchtower canonical`.
type canonicalOutput struct {
	TraceID        string   `json:"traceId"`
	Closed         bool     `json:"closed"`
	Checksum       string   `json:"checksum"`
	Render         string   `json:"render"`
	TopoLimit      int      `json:"topoLimit"`
	Serializations []string `json:"serializations"`
}
