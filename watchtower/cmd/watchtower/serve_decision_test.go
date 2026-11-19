package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/decide"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/report"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

type countingJudge struct {
	calls    int
	findings []verify.Finding
	err      error
}

func (j *countingJudge) Name() string          { return "judge" }
func (j *countingJudge) Usage() (int64, int64) { return 0, 0 }
func (j *countingJudge) Run(*graph.Run) ([]verify.Finding, error) {
	j.calls++
	return j.findings, j.err
}

func reconstructTestdata(t *testing.T, name string) *graph.Run {
	t.Helper()
	env, err := readEnvelope("../../testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	run, err := graph.Reconstruct(env.Spans)
	if err != nil {
		t.Fatal(err)
	}
	return run
}

func repeatRuns(n int, r decide.Run) []decide.Run {
	out := make([]decide.Run, n)
	for i := range out {
		out[i] = r
	}
	return out
}

func TestVerifyAndDecide(t *testing.T) {
	run := reconstructTestdata(t, "clean_trace.json")
	gapCfg := verify.Config{Evidence: verify.EvidenceConfig{Enabled: true, Obligations: verify.EvidenceObligations{RequireTraceClosed: true}}}
	sig, _ := decide.Signature(verify.Verify(run, verify.Config{}))
	gapSig, gaps := decide.Signature(verify.Verify(run, gapCfg))
	if !gaps {
		t.Fatalf("testdata trace should lack the closure marker; got %q", gapSig)
	}
	flagged := []verify.Finding{{Verifier: "judge", Source: "m", Message: "looks wrong"}}

	safe := decide.Fit(repeatRuns(200, decide.Run{Signature: sig}))
	uncertain := decide.Fit(append(
		repeatRuns(100, decide.Run{Signature: sig, Faulty: true, Judged: true, JudgeFlag: true}),
		repeatRuns(100, decide.Run{Signature: sig, Judged: true})...))
	replayable := decide.Fit(append(append(
		repeatRuns(10, decide.Run{Signature: gapSig, Gaps: true, Faulty: true, ReplaySignature: "FAIL|viol:schema"}),
		repeatRuns(10, decide.Run{Signature: gapSig, Gaps: true, ReplaySignature: sig})...),
		append(repeatRuns(80, decide.Run{Signature: sig}), repeatRuns(80, decide.Run{Signature: "FAIL|viol:schema", Faulty: true})...)...))
	for _, m := range []*decide.Model{safe, uncertain, replayable} {
		m.Costs = decide.DefaultCosts()
	}

	tests := []struct {
		name        string
		cfg         verify.Config
		model       *decide.Model
		judge       *countingJudge
		wantJudged  bool
		wantFirst   string
		wantAction  string
		wantFallbck string
	}{
		{"no model judges every run", verify.Config{}, nil, &countingJudge{}, true, "", "", ""},
		{"safe run accepts without the judge", verify.Config{}, safe, &countingJudge{}, false, "accept", "accept", ""},
		{"uncertain run buys the judge and settles on its answer", verify.Config{}, uncertain, &countingJudge{findings: flagged}, true, "judge", "halt", ""},
		{"judge clears an uncertain run", verify.Config{}, uncertain, &countingJudge{}, true, "judge", "accept", ""},
		{"judge outage settles on the prior risk", verify.Config{}, uncertain, &countingJudge{err: errors.New("timeout")}, true, "judge", "halt", ""},
		{"evidence gap asks the client to replay", gapCfg, replayable, &countingJudge{}, false, "replay", "replay", "halt"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := tt.cfg
			cfg.Judge = tt.judge
			r, judged := verifyAndDecide(run, cfg, tt.model, nil)
			if judged != tt.wantJudged || r.Judged != tt.wantJudged {
				t.Fatalf("judged = %v (report %v), want %v", judged, r.Judged, tt.wantJudged)
			}
			if want := map[bool]int{true: 1, false: 0}[tt.wantJudged]; tt.judge.calls != want {
				t.Fatalf("judge calls = %d, want %d", tt.judge.calls, want)
			}
			if tt.model == nil {
				if r.Decision != nil {
					t.Fatalf("decision without a model: %+v", r.Decision)
				}
				return
			}
			d := r.Decision
			if d == nil || d.First != tt.wantFirst || d.Action != tt.wantAction || d.Fallback != tt.wantFallbck {
				t.Fatalf("decision = %+v, want first=%s action=%s fallback=%s", d, tt.wantFirst, tt.wantAction, tt.wantFallbck)
			}
			if d.First == "judge" && d.Posterior == nil {
				t.Fatal("a judge probe must record its posterior")
			}
		})
	}
}

func TestVerifyAndDecideStreamingEProcessHaltsAndSkipsJudge(t *testing.T) {
	run := reconstructTestdata(t, "clean_trace.json")
	sig, _ := decide.Signature(verify.Verify(run, verify.Config{}))
	// Fit a model where sig has 50% risk and the judge flags every run.
	uncertain := decide.Fit(append(
		repeatRuns(50, decide.Run{Signature: sig, Faulty: true, Judged: true, JudgeFlag: true}),
		repeatRuns(50, decide.Run{Signature: sig, Judged: true})...))
	uncertain.Costs = decide.DefaultCosts()

	stream, err := decide.NewEProcess(decide.EProcessConfig{NullRate: 0.05, Alpha: 0.05, Signal: decide.SignalRisk})
	if err != nil {
		t.Fatal(err)
	}
	judge := &countingJudge{findings: []verify.Finding{{Verifier: "judge", Source: "m", Message: "bad"}}}
	cfg := verify.Config{Judge: judge}

	var haltStep int
	for step := 1; step <= 10; step++ {
		r, _ := verifyAndDecide(run, cfg, uncertain, stream)
		if r.Decision == nil || r.Decision.Stream == nil {
			t.Fatalf("step %d: missing stream state in report decision", step)
		}
		if r.Decision.StreamHalted {
			haltStep = step
			break
		}
	}
	if haltStep == 0 {
		t.Fatal("e-process failed to halt under repeated high-risk runs")
	}
	callsAtHalt := judge.calls
	// Subsequent runs after the stream halts must halt immediately without calling the judge.
	r, judged := verifyAndDecide(run, cfg, uncertain, stream)
	if judged || judge.calls != callsAtHalt {
		t.Fatalf("judge was still called after stream halted: calls %d -> %d", callsAtHalt, judge.calls)
	}
	if r.Decision.Action != "halt" || !r.Decision.StreamHalted {
		t.Fatalf("post-halt decision = %+v, want halt with streamHalted=true", r.Decision)
	}
}

func TestServeStreamEndpoints(t *testing.T) {
	stream, err := decide.NewEProcess(decide.EProcessConfig{NullRate: 0.05, Alpha: 0.05, Signal: decide.SignalNonPass})
	if err != nil {
		t.Fatal(err)
	}
	// Trigger a non-PASS verdict on every run by requiring trace closure.
	cfg := verify.Config{Evidence: verify.EvidenceConfig{Enabled: true, Obligations: verify.EvidenceObligations{RequireTraceClosed: true}}}
	mux := newServeMux(cfg, report.NewStore(time.Minute), nil, stream)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	raw := minimalOTLP(t, map[string]string{"agent.name": "worker-a"})
	for i := 0; i < 4; i++ {
		resp, err := http.Post(srv.URL+"/v1/traces", "application/x-protobuf", bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}

	resp, err := http.Get(srv.URL + "/v1/stream")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var st decide.StreamState
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		t.Fatal(err)
	}
	if st.Steps != 4 || !st.Halted {
		t.Fatalf("GET /v1/stream = %+v, want 4 steps and halted=true", st)
	}

	resetResp, err := http.Post(srv.URL+"/v1/stream/reset", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resetResp.Body.Close()
	var reset decide.StreamState
	if err := json.NewDecoder(resetResp.Body).Decode(&reset); err != nil {
		t.Fatal(err)
	}
	if reset.Steps != 0 || reset.Halted {
		t.Fatalf("POST /v1/stream/reset = %+v, want clean reset", reset)
	}
}
