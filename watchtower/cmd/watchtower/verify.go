package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/report"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

// newVerifyCmd runs the whole pipeline over a trace file:
// envelope -> reconstruct -> verify -> report (JSON on stdout).
func newVerifyCmd() *cobra.Command {
	var traceFile, configFile, outFile string
	cmd := &cobra.Command{
		Use:   "verify",
		Short: "verify a trace file and emit a verdict report",
		RunE: func(cmd *cobra.Command, args []string) error {
			env, err := readEnvelope(traceFile)
			if err != nil {
				return err
			}
			cfg, err := loadConfig(configFile)
			if err != nil {
				return err
			}
			cfg.Judge, err = buildJudge(cfg)
			if err != nil {
				return err
			}
			run, err := graph.Reconstruct(env.Spans)
			if err != nil {
				return fmt.Errorf("reconstruct: %w", err)
			}
			r := report.Build(run, verify.Verify(run, cfg), cfg.Judge != nil)
			out, err := json.MarshalIndent(r, "", "  ")
			if err != nil {
				return err
			}
			if outFile != "" {
				return os.WriteFile(outFile, out, 0o644)
			}
			_, err = fmt.Fprintln(cmd.OutOrStdout(), string(out))
			return err
		},
	}
	cmd.Flags().StringVar(&traceFile, "trace-file", "", "path to a span envelope JSON file (required)")
	cmd.Flags().StringVar(&configFile, "config", "", "path to a verifier config JSON file (optional)")
	cmd.Flags().StringVar(&outFile, "output", "", "write the report to this file instead of stdout (optional)")
	_ = cmd.MarkFlagRequired("trace-file")
	return cmd
}

func readEnvelope(path string) (*model.Envelope, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read trace file: %w", err)
	}
	var env model.Envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("parse trace file: %w", err)
	}
	if err := env.Validate(); err != nil {
		return nil, fmt.Errorf("trace file: %w", err)
	}
	return &env, nil
}

// loadConfig reads a verifier config; an absent file means "all checks
// disabled" (zero Config) — an explicit choice, not a silent default.
func loadConfig(path string) (verify.Config, error) {
	var cfg verify.Config
	if path == "" {
		return cfg, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("read config: %w", err)
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return cfg, fmt.Errorf("parse config: %w", err)
	}
	return cfg, nil
}
