package main

import (
	"github.com/spf13/cobra"
)

// NewRootCmd builds the watchtower CLI. It is a function (rather than a
// package-level var) so tests can construct fresh commands per test case.
func NewRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "watchtower",
		Short: "runtime verification for multi-agent systems",
		Long: "Watchtower ingests OpenTelemetry GenAI spans, reconstructs " +
			"agent runs, applies deterministic verifiers (schema, tool policy, " +
			"loop detection, budgets), and emits an evidence-backed verdict per run.",
	}
	root.AddCommand(newVerifyCmd())
	root.AddCommand(newServeCmd())
	return root
}
