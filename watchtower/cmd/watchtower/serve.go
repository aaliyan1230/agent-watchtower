package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/decide"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/ingest"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/report"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

// newServeCmd runs the ingest endpoint; every submitted envelope is
// reconstructed and verified immediately, the verdict is stored per
// trace id (GET /v1/reports/{traceId}) and, for JSON clients, returned
// in the response.
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
			cfg.Judge, err = buildJudge(cfg)
			if err != nil {
				return err
			}
			policy, stream, err := loadDecision(configFile)
			if err != nil {
				return err
			}
			log.Printf("watchtower listening on %s", addr)
			return http.ListenAndServe(addr, newServeMux(cfg, report.NewStore(5*time.Minute), policy, stream))
		},
	}
	cmd.Flags().StringVar(&addr, "addr", ":4318", "listen address")
	cmd.Flags().StringVar(&configFile, "config", "", "path to a verifier config JSON file (optional)")
	return cmd
}

// loadDecision reads the optional "decision": {"model": path, "stream": {...}}
// block of the serve config; a relative model path resolves against the
// config's directory so the server can be spawned from anywhere.
func loadDecision(configFile string) (*decide.Model, *decide.EProcess, error) {
	if configFile == "" {
		return nil, nil, nil
	}
	raw, err := os.ReadFile(configFile)
	if err != nil {
		return nil, nil, err
	}
	var cfg struct {
		Decision struct {
			Model  string                 `json:"model"`
			Stream *decide.EProcessConfig `json:"stream"`
		} `json:"decision"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, nil, err
	}
	var m *decide.Model
	if cfg.Decision.Model != "" {
		path := cfg.Decision.Model
		if !filepath.IsAbs(path) {
			path = filepath.Join(filepath.Dir(configFile), path)
		}
		mraw, err := os.ReadFile(path)
		if err != nil {
			return nil, nil, fmt.Errorf("decision model: %w", err)
		}
		var loaded decide.Model
		if err := json.Unmarshal(mraw, &loaded); err != nil {
			return nil, nil, fmt.Errorf("decision model %s: %w", path, err)
		}
		if loaded.Costs == (decide.Costs{}) {
			loaded.Costs = decide.DefaultCosts()
		}
		m = &loaded
	}
	var stream *decide.EProcess
	if cfg.Decision.Stream != nil {
		stream, err = decide.NewEProcess(*cfg.Decision.Stream)
		if err != nil {
			return nil, nil, fmt.Errorf("decision stream: %w", err)
		}
		if m == nil && stream.Config().Signal == decide.SignalRisk {
			return nil, nil, fmt.Errorf("decision stream signal %q requires a decision model", decide.SignalRisk)
		}
	}
	return m, stream, nil
}

func toReportStream(st decide.StreamState) *report.StreamState {
	return &report.StreamState{
		Signal:    string(st.Signal),
		NullRate:  st.NullRate,
		Alpha:     st.Alpha,
		Threshold: st.Threshold,
		Steps:     st.Steps,
		Mean:      st.Mean,
		Wealth:    st.Wealth,
		MaxWealth: st.MaxWealth,
		Halted:    st.Halted,
		HaltStep:  st.HaltStep,
	}
}

// verifyAndDecide verifies the run and, with a decision model or
// streaming e-process, picks its action. The judge is then a probe like
// any other: it runs only when the decision says its answer is worth
// its price (and the streaming circuit breaker has not already tripped).
func verifyAndDecide(run *graph.Run, cfg verify.Config, m *decide.Model, stream *decide.EProcess) (*report.Report, bool) {
	if m == nil && stream == nil {
		return report.Build(run, verify.Verify(run, cfg), cfg.Judge != nil), cfg.Judge != nil
	}
	if m == nil {
		findings := verify.Verify(run, cfg)
		sig, _ := decide.Signature(findings)
		_, _ = stream.Observe(decide.SignalValue(stream.Config().Signal, sig, 0))
		st := stream.State()
		action := string(decide.Accept)
		if st.Halted {
			action = string(decide.Halt)
		}
		r := report.Build(run, findings, cfg.Judge != nil)
		r.Decision = &report.Decision{
			Signature:    sig,
			First:        action,
			Action:       action,
			StreamHalted: st.Halted,
			Stream:       toReportStream(st),
		}
		return r, cfg.Judge != nil
	}
	det := cfg
	det.Judge = nil
	findings := verify.Verify(run, det)
	sig, gaps := decide.Signature(findings)
	preHalted := stream != nil && stream.Halted()
	d := m.Decide(decide.Observation{Signature: sig, CanReplay: gaps, CanJudge: !preHalted && cfg.Judge != nil}, m.Costs, !preHalted)
	rec := &report.Decision{
		Signature: sig,
		Risk:      d.Risk,
		First:     string(d.Action),
		Action:    string(d.Action),
		Expected:  map[string]float64{},
		Gated:     d.Gated,
		Threshold: m.Threshold,
	}
	for a, cost := range d.Expected {
		rec.Expected[string(a)] = cost
	}
	judged := false
	finalRisk := d.Risk
	if !preHalted {
		switch d.Action {
		case decide.Judge:
			js := verify.RunJudge(run, cfg.Judge)
			findings = append(findings, js...)
			judged = true
			posterior := d.Risk
			if !hasJudgeError(js) {
				posterior = m.JudgePosterior(sig, d.Risk, decide.JudgeFlag(js))
			}
			finalRisk = posterior
			s := m.Settle(posterior, m.Costs)
			rec.Posterior, rec.Action, rec.Gated = &posterior, string(s.Action), s.Gated
		case decide.Replay:
			rec.Fallback = string(m.Settle(d.Risk, m.Costs).Action)
		}
	}
	if stream != nil {
		_, _ = stream.Observe(decide.SignalValue(stream.Config().Signal, sig, finalRisk))
		st := stream.State()
		rec.Stream = toReportStream(st)
		if st.Halted {
			rec.Action = string(decide.Halt)
			rec.StreamHalted = true
		}
	}
	r := report.Build(run, findings, judged)
	r.Decision = rec
	return r, judged
}

func hasJudgeError(findings []verify.Finding) bool {
	for _, f := range findings {
		if f.Kind == verify.FindingJudgeError {
			return true
		}
	}
	return false
}

// newServeMux wires ingest -> verify -> decide -> report store ->
// retrieval endpoint. Split out so tests can mount it in an httptest
// server.
func newServeMux(cfg verify.Config, store *report.Store, m *decide.Model, stream *decide.EProcess) http.Handler {
	h := ingest.New(func(env *model.Envelope) ([]byte, error) {
		run, err := graph.Reconstruct(env.Spans)
		if err != nil {
			return nil, fmt.Errorf("reconstruct: %w", err)
		}
		r, judged := verifyAndDecide(run, cfg, m, stream)
		if judged {
			in, out := cfg.Judge.Usage()
			r.SetJudgeUsage(in, out)
			if jm, ok := cfg.Judge.(interface{ Model() string }); ok {
				r.JudgeModel = jm.Model()
			}
		}
		store.Put(r.TraceID, r)
		if r.Decision != nil {
			log.Printf("trace %s: %s -> %s (%d findings)", r.TraceID, r.Verdict, r.Decision.Action, len(r.Findings))
		} else {
			log.Printf("trace %s: %s (%d findings)", r.TraceID, r.Verdict, len(r.Findings))
		}
		return json.Marshal(r)
	})

	mux := http.NewServeMux()
	mux.Handle("/v1/traces", h)
	mux.HandleFunc("/v1/reports/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		traceID := r.URL.Path[len("/v1/reports/"):]
		rep, ok := store.Get(traceID)
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(rep)
	})
	mux.HandleFunc("/v1/stream", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if stream == nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(stream.State())
	})
	mux.HandleFunc("/v1/stream/reset", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if stream == nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(stream.Reset())
	})
	return mux
}
