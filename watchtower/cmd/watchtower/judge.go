package main

import (
	"errors"
	"os"

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
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return nil, errors.New("judge enabled but GEMINI_API_KEY is not set (add it to .env)")
	}
	model := cfg.JudgeCfg.Model
	if model == "" {
		model = "gemini-2.5-flash"
	}
	return judge.New(model, key, cfg.JudgeCfg.BaseURL), nil
}
