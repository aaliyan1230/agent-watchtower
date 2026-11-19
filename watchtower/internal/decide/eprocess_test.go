package decide

import (
	"encoding/json"
	"math"
	"math/rand"
	"sync"
	"testing"
)

func TestNewEProcessValidation(t *testing.T) {
	tests := []struct {
		name    string
		cfg     EProcessConfig
		wantErr bool
	}{
		{"valid defaults", EProcessConfig{NullRate: 0.05, Alpha: 0.05}, false},
		{"valid explicit", EProcessConfig{NullRate: 0.1, Alpha: 0.01, MaxBetFraction: 0.5, Signal: SignalNonPass}, false},
		{"zero nullRate", EProcessConfig{NullRate: 0, Alpha: 0.05}, true},
		{"one nullRate", EProcessConfig{NullRate: 1, Alpha: 0.05}, true},
		{"nan nullRate", EProcessConfig{NullRate: math.NaN(), Alpha: 0.05}, true},
		{"zero alpha", EProcessConfig{NullRate: 0.05, Alpha: 0}, true},
		{"one alpha", EProcessConfig{NullRate: 0.05, Alpha: 1}, true},
		{"bad maxBetFraction", EProcessConfig{NullRate: 0.05, Alpha: 0.05, MaxBetFraction: 1}, true},
		{"unknown signal", EProcessConfig{NullRate: 0.05, Alpha: 0.05, Signal: "unknown"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ep, err := NewEProcess(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Fatalf("NewEProcess() err = %v, wantErr = %v", err, tt.wantErr)
			}
			if err == nil {
				if ep.Config().MaxBetFraction <= 0 || ep.Config().Signal == "" {
					t.Fatalf("defaults not populated: %+v", ep.Config())
				}
			}
		})
	}
}

func TestEProcessRejectsOutOfRangeObservation(t *testing.T) {
	ep, err := NewEProcess(EProcessConfig{NullRate: 0.05, Alpha: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []float64{-0.01, 1.01, math.NaN(), math.Inf(1)} {
		if _, err := ep.Observe(bad); err == nil {
			t.Errorf("Observe(%v) succeeded, want error", bad)
		}
	}
}

func TestEProcessAnytimeTypeIControlUnderNull(t *testing.T) {
	const (
		streams = 1000
		length  = 200
		p0      = 0.05
		alpha   = 0.05
	)
	tests := []struct {
		name   string
		sample func(rng *rand.Rand) float64
	}{
		{
			name: "bernoulli at boundary p=p0",
			sample: func(rng *rand.Rand) float64 {
				if rng.Float64() < p0 {
					return 1
				}
				return 0
			},
		},
		{
			name: "continuous bounded risk with mean p0",
			sample: func(rng *rand.Rand) float64 {
				return rng.Float64() * 2 * p0
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(20261119))
			halted := 0
			for s := 0; s < streams; s++ {
				ep, err := NewEProcess(EProcessConfig{NullRate: p0, Alpha: alpha})
				if err != nil {
					t.Fatal(err)
				}
				for step := 0; step < length; step++ {
					if _, err := ep.Observe(tt.sample(rng)); err != nil {
						t.Fatal(err)
					}
				}
				if ep.Halted() {
					halted++
				}
			}
			rate := float64(halted) / float64(streams)
			// Ville's inequality guarantees P(exists t: E_t >= 1/alpha) <= alpha.
			if rate > alpha {
				t.Fatalf("empirical false-halt rate = %.4f (%d/%d) > alpha %.2f", rate, halted, streams, alpha)
			}
		})
	}
}

func TestEProcessHaltsQuicklyUnderDrift(t *testing.T) {
	const (
		streams = 200
		length  = 100
		p0      = 0.05
		p1      = 0.30
		alpha   = 0.05
	)
	rng := rand.New(rand.NewSource(42))
	halted := 0
	totalStop := 0
	for s := 0; s < streams; s++ {
		ep, err := NewEProcess(EProcessConfig{NullRate: p0, Alpha: alpha})
		if err != nil {
			t.Fatal(err)
		}
		for step := 0; step < length; step++ {
			x := 0.0
			if rng.Float64() < p1 {
				x = 1
			}
			if _, err := ep.Observe(x); err != nil {
				t.Fatal(err)
			}
		}
		st := ep.State()
		if st.Halted {
			halted++
			totalStop += st.HaltStep
		}
	}
	if halted < 195 {
		t.Fatalf("halted only %d/%d drifted streams, want >= 195", halted, streams)
	}
	meanStop := float64(totalStop) / float64(halted)
	if meanStop > 30 {
		t.Fatalf("mean stopping time = %.1f steps, want <= 30", meanStop)
	}
}

func TestEProcessOverflowSafeAndReset(t *testing.T) {
	ep, err := NewEProcess(EProcessConfig{NullRate: 0.05, Alpha: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 500; i++ {
		if _, err := ep.Observe(1); err != nil {
			t.Fatal(err)
		}
	}
	st := ep.State()
	if !st.Halted || st.HaltStep == 0 || st.Steps != 500 {
		t.Fatalf("unexpected state after 500 faults: %+v", st)
	}
	if math.IsInf(st.Wealth, 0) || math.IsNaN(st.Wealth) {
		t.Fatalf("non-finite wealth: %v", st.Wealth)
	}
	if _, err := json.Marshal(st); err != nil {
		t.Fatalf("State() must marshal cleanly to JSON: %v", err)
	}

	reset := ep.Reset()
	if reset.Halted || reset.Steps != 0 || reset.HaltStep != 0 || !near(reset.Wealth, 1) {
		t.Fatalf("Reset() = %+v, want clean initial state", reset)
	}
}

func TestSignalValue(t *testing.T) {
	tests := []struct {
		signal StreamSignal
		sig    string
		risk   float64
		want   float64
	}{
		{SignalRisk, pass, 0.12, 0.12},
		{SignalRisk, pass, -0.5, 0},
		{SignalRisk, pass, 1.5, 1},
		{SignalViolation, fail, 0.9, 1},
		{SignalViolation, gap, 0.6, 0},
		{SignalNonPass, pass, 0.02, 0},
		{SignalNonPass, gap, 0.6, 1},
		{SignalNonPass, fail, 0.9, 1},
	}
	for _, tt := range tests {
		if got := SignalValue(tt.signal, tt.sig, tt.risk); !near(got, tt.want) {
			t.Errorf("SignalValue(%s, %q, %v) = %v, want %v", tt.signal, tt.sig, tt.risk, got, tt.want)
		}
	}
}

func TestEProcessConcurrent(t *testing.T) {
	ep, err := NewEProcess(EProcessConfig{NullRate: 0.1, Alpha: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				if _, err := ep.Observe(float64((i+id)%2) * 0.2); err != nil {
					t.Errorf("Observe: %v", err)
				}
				_ = ep.State()
				_ = ep.Halted()
			}
		}(g)
	}
	wg.Wait()
	if st := ep.State(); st.Steps != 400 {
		t.Fatalf("Steps = %d, want 400", st.Steps)
	}
}
