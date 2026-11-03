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
			model = "gemini-3.8-flash"
		}
		return judge.New(model, key, cfg.JudgeCfg.BaseURL, geminiReasoning(cfg.JudgeCfg.Reasoning)), nil
	case "openrouter":
		key := os.Getenv("OPENROUTER_API_KEY")
		if key == "" {
			return nil, errors.New("judge enabled but OPENROUTER_API_KEY is not set (add it to .env)")
		}
		model := cfg.JudgeCfg.Model
		if model == "" {
			model = "deepseek/deepseek-v4.1-flash-1017"
		}
		baseURL := cfg.JudgeCfg.BaseURL
		if baseURL == "" {
			baseURL = "https://openrouter.ai/api/v1"
		}
		return judge.New(model, key, baseURL, openRouterReasoning(cfg.JudgeCfg.Reasoning)), nil
	case "bedrock":
		creds, err := bedrock.CredentialsFromEnv()
		if err != nil {
			return nil, err
		}
		model := cfg.JudgeCfg.Model
		if model == "" {
			model = "deepseek.v3.2"
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

// geminiReasoning maps a level onto the Gemini OpenAI-compat field.
func geminiReasoning(level string) map[string]any {
	switch level {
	case "":
		return nil
	case "off":
		return map[string]any{"reasoning_effort": "none"}
	default:
		return map[string]any{"reasoning_effort": level}
	}
}

// openRouterReasoning maps a level onto OpenRouter's unified reasoning
// object, which it translates per upstream vendor.
func openRouterReasoning(level string) map[string]any {
	switch level {
	case "":
		return nil
	case "off":
		return map[string]any{"reasoning": map[string]any{"enabled": false}}
	default:
		return map[string]any{"reasoning": map[string]any{"effort": level}}
	}
}
