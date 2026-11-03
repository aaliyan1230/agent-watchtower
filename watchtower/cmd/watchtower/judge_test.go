package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/report"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

func TestServeWithJudge(t *testing.T) {
	// The judge's model endpoint is a stub here; the integration
	// question is whether judge findings reach the report.
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"issues\":[{\"severity\":\"warning\",\"description\":\"step 2 looks suspicious\"}]}"}}]}`))
	}))
	defer llmSrv.Close()

	t.Setenv("GEMINI_API_KEY", "test-key")
	cfg := verify.Config{
		JudgeCfg: verify.JudgeConfig{Enabled: true, Model: "gemini-2.5-flash", BaseURL: llmSrv.URL},
	}
	j, err := buildJudge(cfg)
	if err != nil {
		t.Fatalf("buildJudge: %v", err)
	}
	cfg.Judge = j

	mux := newServeMux(cfg, report.NewStore(time.Minute))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	raw := minimalOTLP(t, map[string]string{"agent.name": "worker-a"})
	resp, err := http.Post(srv.URL+"/v1/traces", "application/x-protobuf", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	repResp, err := http.Get(srv.URL + "/v1/reports/01020304050607080102030405060708")
	if err != nil {
		t.Fatal(err)
	}
	defer repResp.Body.Close()
	var r map[string]any
	if err := json.NewDecoder(repResp.Body).Decode(&r); err != nil {
		t.Fatal(err)
	}
	if r["judged"] != true {
		t.Fatalf("judged = %v, want true", r["judged"])
	}
	rawReport, _ := json.Marshal(r)
	if !bytes.Contains(rawReport, []byte("step 2 looks suspicious")) {
		t.Fatalf("report missing judge finding: %s", rawReport)
	}
	if !bytes.Contains(rawReport, []byte("gemini-2.5-flash")) {
		t.Fatalf("report missing judge source model: %s", rawReport)
	}
}

func TestBuildJudgeRequiresKey(t *testing.T) {
	os.Unsetenv("GEMINI_API_KEY")
	if _, err := buildJudge(verify.Config{JudgeCfg: verify.JudgeConfig{Enabled: true}}); err == nil {
		t.Fatal("expected error when judge enabled without a key")
	}
	if j, err := buildJudge(verify.Config{}); err != nil || j != nil {
		t.Fatalf("disabled judge should be nil, got %v, %v", j, err)
	}
}

func TestBuildJudgeBedrock(t *testing.T) {
	t.Setenv("AWS_ACCESS_KEY_ID", "AK")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "SK")
	j, err := buildJudge(verify.Config{JudgeCfg: verify.JudgeConfig{Enabled: true, Backend: "bedrock"}})
	if err != nil {
		t.Fatalf("buildJudge: %v", err)
	}
	if j == nil {
		t.Fatal("bedrock judge should be constructed")
	}
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	if _, err := buildJudge(verify.Config{JudgeCfg: verify.JudgeConfig{Enabled: true, Backend: "bedrock"}}); err == nil {
		t.Fatal("bedrock judge without creds must fail loudly")
	}
	if _, err := buildJudge(verify.Config{JudgeCfg: verify.JudgeConfig{Enabled: true, Backend: "nope"}}); err == nil {
		t.Fatal("unknown backend must fail")
	}
}

func TestBuildJudgeOpenRouter(t *testing.T) {
	var gotAuth, gotModel string
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotModel = body.Model
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"issues\":[]}"}}]}`))
	}))
	defer llmSrv.Close()

	t.Setenv("OPENROUTER_API_KEY", "or-key")
	cfg := verify.JudgeConfig{Enabled: true, Backend: "openrouter", BaseURL: llmSrv.URL}
	j, err := buildJudge(verify.Config{JudgeCfg: cfg})
	if err != nil {
		t.Fatalf("buildJudge: %v", err)
	}
	if m, ok := j.(interface{ Model() string }); !ok || m.Model() != "deepseek/deepseek-v4.1-flash-1017" {
		t.Fatalf("default openrouter model = %v", j)
	}
	if _, err := j.Run(&graph.Run{TraceID: "t"}); err != nil {
		t.Fatalf("judge call: %v", err)
	}
	if gotAuth != "Bearer or-key" || gotModel != "deepseek/deepseek-v4.1-flash-1017" {
		t.Fatalf("auth=%q model=%q", gotAuth, gotModel)
	}

	t.Setenv("OPENROUTER_API_KEY", "")
	if _, err := buildJudge(verify.Config{JudgeCfg: cfg}); err == nil {
		t.Fatal("openrouter judge without a key must fail loudly")
	}
}

func TestBuildJudgeReasoningFields(t *testing.T) {
	cases := []struct {
		name, backend, keyEnv, level, want string
	}{
		{"gemini low", "gemini", "GEMINI_API_KEY", "low", `"reasoning_effort":"low"`},
		{"gemini off", "gemini", "GEMINI_API_KEY", "off", `"reasoning_effort":"none"`},
		{"openrouter off", "openrouter", "OPENROUTER_API_KEY", "off", `"reasoning":{"enabled":false}`},
		{"openrouter high", "openrouter", "OPENROUTER_API_KEY", "high", `"reasoning":{"effort":"high"}`},
		{"default omits field", "openrouter", "OPENROUTER_API_KEY", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var body string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				body = string(raw)
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"issues\":[]}"}}]}`))
			}))
			defer srv.Close()
			t.Setenv(tc.keyEnv, "k")
			cfg := verify.JudgeConfig{Enabled: true, Backend: tc.backend, BaseURL: srv.URL, Reasoning: tc.level}
			j, err := buildJudge(verify.Config{JudgeCfg: cfg})
			if err != nil {
				t.Fatalf("buildJudge: %v", err)
			}
			if _, err := j.Run(&graph.Run{TraceID: "t"}); err != nil {
				t.Fatalf("judge call: %v", err)
			}
			if tc.want == "" {
				if strings.Contains(body, "reasoning") {
					t.Fatalf("unexpected reasoning field: %s", body)
				}
			} else if !strings.Contains(body, tc.want) {
				t.Fatalf("body %s missing %s", body, tc.want)
			}
		})
	}
}
