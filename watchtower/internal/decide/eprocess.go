package decide

import (
	"fmt"
	"math"
	"sync"
)

// StreamSignal selects which bounded [0, 1] observation each run feeds
// into the anytime-valid e-process during streaming verification.
type StreamSignal string

const (
	// SignalRisk feeds the run's calibrated risk (posterior if judged,
	// prior otherwise). Tests H0: E[risk_t | F_{t-1}] <= NullRate.
	SignalRisk StreamSignal = "risk"
	// SignalViolation feeds 1 on a deterministic FAIL verdict, else 0.
	SignalViolation StreamSignal = "violation"
	// SignalNonPass feeds 0 on PASS, 1 on any other verdict.
	SignalNonPass StreamSignal = "non_pass"
)

// EProcessConfig configures a betting e-process for anytime-valid
// sequential halting under H0: E[X_t | F_{t-1}] <= NullRate.
type EProcessConfig struct {
	NullRate       float64      `json:"nullRate"`
	Alpha          float64      `json:"alpha"`
	MaxBetFraction float64      `json:"maxBetFraction,omitempty"`
	Signal         StreamSignal `json:"signal,omitempty"`
}

// StreamState is a serializable snapshot of an EProcess.
type StreamState struct {
	Signal    StreamSignal `json:"signal,omitempty"`
	NullRate  float64      `json:"nullRate"`
	Alpha     float64      `json:"alpha"`
	Threshold float64      `json:"threshold"`
	Steps     int          `json:"steps"`
	Mean      float64      `json:"mean"`
	Wealth    float64      `json:"wealth"`
	MaxWealth float64      `json:"maxWealth"`
	Halted    bool         `json:"halted"`
	HaltStep  int          `json:"haltStep,omitempty"`
}

// EStep is the evidence update from a single streaming observation.
type EStep struct {
	Step   int     `json:"step"`
	Value  float64 `json:"value"`
	Bet    float64 `json:"bet"`
	Factor float64 `json:"factor"`
	Wealth float64 `json:"wealth"`
	Halted bool    `json:"halted"`
}

// maxSafeLog caps math.Exp so serialized wealth never becomes +Inf.
const maxSafeLog = 700

// EProcess implements a predictable plug-in betting e-process
// (Waudby-Smith & Ramdas, 2024) for bounded observations X_t in [0, 1].
// At step t, lambda_t >= 0 is chosen solely from X_1..X_{t-1} and
// capped at MaxBetFraction/NullRate so every factor 1 + lambda_t*(X_t - p0)
// stays strictly positive. Under H0: E[X_t | F_{t-1}] <= NullRate, the
// wealth process is a nonnegative supermartingale with E_0 = 1, so by
// Ville's inequality P(exists t >= 1: E_t >= 1/Alpha) <= Alpha at any
// stopping time.
type EProcess struct {
	mu           sync.Mutex
	cfg          EProcessConfig
	logThreshold float64
	steps        int
	sum          float64
	logWealth    float64
	maxLogWealth float64
	halted       bool
	haltStep     int
}

func NewEProcess(cfg EProcessConfig) (*EProcess, error) {
	if math.IsNaN(cfg.NullRate) || cfg.NullRate <= 0 || cfg.NullRate >= 1 {
		return nil, fmt.Errorf("eprocess nullRate must be in (0, 1), got %v", cfg.NullRate)
	}
	if math.IsNaN(cfg.Alpha) || cfg.Alpha <= 0 || cfg.Alpha >= 1 {
		return nil, fmt.Errorf("eprocess alpha must be in (0, 1), got %v", cfg.Alpha)
	}
	if cfg.MaxBetFraction == 0 {
		cfg.MaxBetFraction = 0.75
	}
	if math.IsNaN(cfg.MaxBetFraction) || cfg.MaxBetFraction <= 0 || cfg.MaxBetFraction >= 1 {
		return nil, fmt.Errorf("eprocess maxBetFraction must be in (0, 1), got %v", cfg.MaxBetFraction)
	}
	if cfg.Signal == "" {
		cfg.Signal = SignalRisk
	}
	switch cfg.Signal {
	case SignalRisk, SignalViolation, SignalNonPass:
	default:
		return nil, fmt.Errorf("unknown eprocess signal %q", cfg.Signal)
	}
	return &EProcess{
		cfg:          cfg,
		logThreshold: -math.Log(cfg.Alpha),
	}, nil
}

// Config returns the normalized configuration.
func (e *EProcess) Config() EProcessConfig {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.cfg
}

// Observe incorporates one bounded observation x in [0, 1] and updates
// the e-value. The bet uses only prior steps so the supermartingale
// property holds under arbitrary sequential peeking.
func (e *EProcess) Observe(x float64) (EStep, error) {
	if math.IsNaN(x) || x < 0 || x > 1 {
		return EStep{}, fmt.Errorf("eprocess observation must be in [0, 1], got %v", x)
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	p0 := e.cfg.NullRate
	pHat := (e.sum + 0.5) / float64(e.steps+1)
	lambda := 0.0
	if pHat > p0 {
		lambda = (pHat - p0) / (p0 * (1 - p0))
		maxBet := e.cfg.MaxBetFraction / p0
		if lambda > maxBet {
			lambda = maxBet
		}
	}
	factor := 1 + lambda*(x-p0)
	e.logWealth += math.Log(factor)
	e.steps++
	e.sum += x
	if e.logWealth > e.maxLogWealth {
		e.maxLogWealth = e.logWealth
	}
	if !e.halted && e.logWealth >= e.logThreshold {
		e.halted = true
		e.haltStep = e.steps
	}
	return EStep{
		Step:   e.steps,
		Value:  x,
		Bet:    lambda,
		Factor: factor,
		Wealth: safeExp(e.logWealth),
		Halted: e.halted,
	}, nil
}

// Halted reports whether the e-process has crossed 1/Alpha.
func (e *EProcess) Halted() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.halted
}

// State returns a thread-safe snapshot of the current process state.
func (e *EProcess) State() StreamState {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.stateLocked()
}

func (e *EProcess) stateLocked() StreamState {
	mean := 0.0
	if e.steps > 0 {
		mean = e.sum / float64(e.steps)
	}
	return StreamState{
		Signal:    e.cfg.Signal,
		NullRate:  e.cfg.NullRate,
		Alpha:     e.cfg.Alpha,
		Threshold: 1 / e.cfg.Alpha,
		Steps:     e.steps,
		Mean:      mean,
		Wealth:    safeExp(e.logWealth),
		MaxWealth: safeExp(e.maxLogWealth),
		Halted:    e.halted,
		HaltStep:  e.haltStep,
	}
}

// Reset clears accumulated evidence so monitoring can resume after
// an operator intervenes.
func (e *EProcess) Reset() StreamState {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.steps = 0
	e.sum = 0
	e.logWealth = 0
	e.maxLogWealth = 0
	e.halted = false
	e.haltStep = 0
	return e.stateLocked()
}

// SignalValue extracts the configured [0, 1] observation from a run's
// deterministic signature and calibrated risk.
func SignalValue(signal StreamSignal, sig string, risk float64) float64 {
	switch signal {
	case SignalViolation:
		if verdictOf(sig) == "FAIL" {
			return 1
		}
		return 0
	case SignalNonPass:
		if verdictOf(sig) != "PASS" {
			return 1
		}
		return 0
	default:
		if risk < 0 {
			return 0
		}
		if risk > 1 {
			return 1
		}
		return risk
	}
}

func safeExp(logVal float64) float64 {
	if logVal > maxSafeLog {
		return math.Exp(maxSafeLog)
	}
	if logVal < -maxSafeLog {
		return 0
	}
	return math.Exp(logVal)
}
