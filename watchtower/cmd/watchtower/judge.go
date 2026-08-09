package main

import (
	"errors"
	"os"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/bedrock"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/judge"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

// buildJudge constructs the LLM judge when the config enables it.
// Keys come from the environment only; enabling the judge without a
// key fails loudly instead of silently judging nothing.
func buildJudge(cfg verify.Config) (verify.Judge, error) {
	if !cfg.JudgeCfg.Enabled {
		return nil, nil
	}
	switch cfg.JudgeCfg.Backend {
	case "", "gemini":
		key := os.Getenv("GEMINI_API_KEY")
		if key == "" {
			return nil, errors.New("judge enabled but GEMINI_API_KEY is not set (add it to .env)")
		}
		model := cfg.JudgeCfg.Model
		if model == "" {
			// Judge calls are capped per run, so the smarter tier is
			// affordable here; workers bulk on flash-lite (see .env.example).
			model = "gemini-3.6-flash"
		}
		return judge.New(model, key, cfg.JudgeCfg.BaseURL), nil
	case "bedrock":
		creds, err := bedrock.CredentialsFromEnv()
		if err != nil {
			return nil, err
		}
		model := cfg.JudgeCfg.Model
		if model == "" {
			model = "amazon.nova-lite-v1:0"
		}
		region := os.Getenv("AWS_REGION")
		if region == "" {
			region = "us-east-1"
		}
		return judge.NewBedrock(model, region, creds), nil
	default:
		return nil, errors.New("unknown judge backend: " + cfg.JudgeCfg.Backend)
	}
}
