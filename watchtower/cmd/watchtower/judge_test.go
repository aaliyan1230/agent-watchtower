package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

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
