package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/decide"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/verify"
)

// artifactCell is the slice of an experiments results cell the decision
// layer needs; the injected behavior fault is the ground-truth label.
type artifactCell struct {
	Fault         *string `json:"fault"`
	EvidenceFault *string `json:"evidence_fault"`
	// Faulty overrides the label when the artifact knows better than the
	// injected fault, e.g. tau-bench's native task failure.
	Faulty *bool `json:"faulty"`
	// ReplayFindings is the report after an actual replay of this run;
	// null when it was not replayed.
	ReplayFindings *[]verify.Finding `json:"replayFindings"`
	Seed           int               `json:"seed"`
	Model          string            `json:"model"`
	Run            int               `json:"run"`
	Judged         bool              `json:"judged"`
	Findings       []verify.Finding  `json:"findings"`
}

// loadRuns labels every cell and attaches its replay outcome: the
// recorded replay when the artifact has one, else the signature of the
// same behavior cell recorded with clean telemetry, which is what
// recollecting the trace would have shown.
func loadRuns(path string) (map[int][]decide.Run, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var artifact struct {
		Cells []artifactCell `json:"cells"`
	}
	if err := json.Unmarshal(raw, &artifact); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	key := func(c artifactCell) string {
		fault := ""
		if c.Fault != nil {
			fault = *c.Fault
		}
		return fmt.Sprintf("%s|%d|%s|%d", fault, c.Seed, c.Model, c.Run)
	}
	clean := map[string]string{}
	for _, c := range artifact.Cells {
		if c.EvidenceFault == nil {
			clean[key(c)], _ = decide.Signature(c.Findings)
		}
	}
	bySeed := map[int][]decide.Run{}
	for _, c := range artifact.Cells {
		sig, gaps := decide.Signature(c.Findings)
		r := decide.Run{
			Signature: sig,
			Gaps:      gaps,
			Faulty:    c.Fault != nil,
			Judged:    c.Judged,
			JudgeFlag: decide.JudgeFlag(c.Findings),
		}
		if c.Faulty != nil {
			r.Faulty = *c.Faulty
		}
		switch {
		case gaps && c.ReplayFindings != nil:
			r.ReplaySignature, _ = decide.Signature(*c.ReplayFindings)
		case gaps:
			r.ReplaySignature = clean[key(c)]
		}
		bySeed[c.Seed] = append(bySeed[c.Seed], r)
	}
	return bySeed, nil
}

func pick(bySeed map[int][]decide.Run, seeds []int) []decide.Run {
	var out []decide.Run
	for _, s := range seeds {
		out = append(out, bySeed[s]...)
	}
	return out
}

func parseSeeds(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("seed list %q: %w", s, err)
		}
		out = append(out, n)
	}
	return out, nil
}

type decideResult struct {
	Costs     decide.Costs     `json:"costs"`
	Threshold float64          `json:"threshold"`
	Policies  []decide.Metrics `json:"policies"`
}

type streamEval struct {
	Name  string             `json:"name"`
	State decide.StreamState `json:"state"`
}

type decideOutput struct {
	Results   string         `json:"results"`
	Alpha     float64        `json:"alpha"`
	FitSeeds  []int          `json:"fitSeeds"`
	CalSeeds  []int          `json:"calSeeds"`
	TestSeeds []int          `json:"testSeeds"`
	FitRuns   int            `json:"fitRuns"`
	CalRuns   int            `json:"calRuns"`
	TestRuns  int            `json:"testRuns"`
	Main      decideResult   `json:"main"`
	Sweep     []decideResult `json:"sweep,omitempty"`
	Rotation  *rotation      `json:"rotation,omitempty"`
	Stream    []streamEval   `json:"stream,omitempty"`
}

// rotation averages every assignment of the seeds to fit, calibrate and
// test with the requested split sizes, so the alpha bound is checked
// across many calibration draws rather than one.
type rotation struct {
	Splits            int                `json:"splits"`
	Mean              []decide.Metrics   `json:"mean"`
	MaxAcceptedFaulty map[string]float64 `json:"maxAcceptedFaulty"`
}

func splits(seeds []int, sizes []int) [][3][]int {
	var out [][3][]int
	var rec func(rest []int, part int, cur [3][]int)
	rec = func(rest []int, part int, cur [3][]int) {
		if part == 3 {
			out = append(out, cur)
			return
		}
		choose(rest, sizes[part], func(picked, left []int) {
			next := cur
			next[part] = picked
			rec(left, part+1, next)
		})
	}
	rec(seeds, 0, [3][]int{})
	return out
}

func choose(items []int, k int, visit func(picked, left []int)) {
	var rec func(start int, picked []int)
	rec = func(start int, picked []int) {
		if len(picked) == k {
			var left []int
			for _, it := range items {
				if !contains(picked, it) {
					left = append(left, it)
				}
			}
			visit(append([]int(nil), picked...), left)
			return
		}
		for i := start; i < len(items); i++ {
			rec(i+1, append(picked, items[i]))
		}
	}
	rec(0, nil)
}

func contains(xs []int, x int) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

func rotate(bySeed map[int][]decide.Run, sizes []int, c decide.Costs, alpha float64) *rotation {
	seeds := make([]int, 0, len(bySeed))
	for s := range bySeed {
		seeds = append(seeds, s)
	}
	sort.Ints(seeds)
	rot := &rotation{MaxAcceptedFaulty: map[string]float64{}}
	sums := map[string]*decide.Metrics{}
	var order []string
	for _, sp := range splits(seeds, sizes) {
		res := comparePolicies(pick(bySeed, sp[0]), pick(bySeed, sp[1]), pick(bySeed, sp[2]), c, alpha)
		rot.Splits++
		for _, m := range res.Policies {
			acc, ok := sums[m.Policy]
			if !ok {
				acc = &decide.Metrics{Policy: m.Policy}
				sums[m.Policy] = acc
				order = append(order, m.Policy)
			}
			acc.Runs += m.Runs
			acc.CostPerRun += m.CostPerRun
			acc.AcceptedFaulty += m.AcceptedFaulty
			acc.AcceptPer100 += m.AcceptPer100
			acc.FalseHaltPer100 += m.FalseHaltPer100
			acc.EscalatePer100 += m.EscalatePer100
			acc.ProbePer100 += m.ProbePer100
			if m.AcceptedFaulty > rot.MaxAcceptedFaulty[m.Policy] {
				rot.MaxAcceptedFaulty[m.Policy] = m.AcceptedFaulty
			}
		}
	}
	n := float64(rot.Splits)
	for _, name := range order {
		m := *sums[name]
		m.Runs = int(float64(m.Runs) / n)
		m.CostPerRun /= n
		m.AcceptedFaulty /= n
		m.AcceptPer100 /= n
		m.FalseHaltPer100 /= n
		m.EscalatePer100 /= n
		m.ProbePer100 /= n
		rot.Mean = append(rot.Mean, m)
	}
	return rot
}

// comparePolicies fits on one split, calibrates the gate on a second,
// and scores every policy on the third, so neither the risk model nor
// the gate has seen a test run.
func comparePolicies(fit, cal, test []decide.Run, c decide.Costs, alpha float64) decideResult {
	gated := decide.Fit(fit)
	threshold := gated.CalibrateAccept(cal, c, true, alpha)
	noProbe := decide.Fit(fit)
	noProbe.CalibrateAccept(cal, c, false, alpha)
	ungated := decide.Fit(fit)
	policies := []decide.Policy{decide.OraclePolicy(), decide.RulesPolicy(c)}
	for _, r := range test {
		if r.Judged {
			policies = append(policies, decide.JudgeFirstPolicy(c))
			break
		}
	}
	policies = append(policies,
		decide.BayesPolicy("bayes-no-probe", noProbe, c, false),
		decide.BayesPolicy("bayes-ungated", ungated, c, true),
		decide.BayesPolicy("bayes", gated, c, true),
	)
	out := decideResult{Costs: c, Threshold: threshold}
	for _, p := range policies {
		out.Policies = append(out.Policies, decide.Evaluate(p, test))
	}
	return out
}

func evalStream(m *decide.Model, c decide.Costs, test []decide.Run, nullRate, alpha float64) ([]streamEval, error) {
	faultsEP, err := decide.NewEProcess(decide.EProcessConfig{NullRate: nullRate, Alpha: alpha})
	if err != nil {
		return nil, err
	}
	accFaultEP, err := decide.NewEProcess(decide.EProcessConfig{NullRate: alpha, Alpha: alpha})
	if err != nil {
		return nil, err
	}
	riskEP, err := decide.NewEProcess(decide.EProcessConfig{NullRate: nullRate, Alpha: alpha, Signal: decide.SignalRisk})
	if err != nil {
		return nil, err
	}
	policy := decide.BayesPolicy("bayes", m, c, true)
	for _, r := range test {
		o := policy.Act(r)
		f := 0.0
		if r.Faulty {
			f = 1
		}
		af := 0.0
		if o.Final == decide.Accept && r.Faulty {
			af = 1
		}
		if _, err := faultsEP.Observe(f); err != nil {
			return nil, err
		}
		if _, err := accFaultEP.Observe(af); err != nil {
			return nil, err
		}
		if _, err := riskEP.Observe(o.AcceptRisk); err != nil {
			return nil, err
		}
	}
	return []streamEval{
		{Name: "faults", State: faultsEP.State()},
		{Name: "accepted-faulty", State: accFaultEP.State()},
		{Name: "risk", State: riskEP.State()},
	}, nil
}

func newDecideCmd() *cobra.Command {
	var results, modelFile, fitSeeds, calSeeds, testSeeds, sweep, outFile, saveFile, dmnFile string
	var alpha, streamNull float64
	var rotateSplits bool
	c := decide.DefaultCosts()
	cmd := &cobra.Command{
		Use:   "decide",
		Short: "evaluate the decision layer offline over an experiment results artifact",
		RunE: func(cmd *cobra.Command, args []string) error {
			if results == "" && modelFile == "" {
				return fmt.Errorf("one of --results or --model is required")
			}
			if modelFile != "" && results == "" {
				raw, err := os.ReadFile(modelFile)
				if err != nil {
					return err
				}
				var m decide.Model
				if err := json.Unmarshal(raw, &m); err != nil {
					return fmt.Errorf("%s: %w", modelFile, err)
				}
				dmn, err := m.ExportDMN()
				if err != nil {
					return err
				}
				if dmnFile == "" || dmnFile == "-" {
					_, err = cmd.OutOrStdout().Write(dmn)
					return err
				}
				return os.WriteFile(dmnFile, dmn, 0o644)
			}
			bySeed, err := loadRuns(results)
			if err != nil {
				return err
			}
			out := decideOutput{Results: results, Alpha: alpha}
			for _, s := range []struct {
				flag string
				dst  *[]int
			}{{fitSeeds, &out.FitSeeds}, {calSeeds, &out.CalSeeds}, {testSeeds, &out.TestSeeds}} {
				if *s.dst, err = parseSeeds(s.flag); err != nil {
					return err
				}
			}
			fit, cal, test := pick(bySeed, out.FitSeeds), pick(bySeed, out.CalSeeds), pick(bySeed, out.TestSeeds)
			out.FitRuns, out.CalRuns, out.TestRuns = len(fit), len(cal), len(test)
			if len(fit) == 0 || len(cal) == 0 || len(test) == 0 {
				return fmt.Errorf("empty split: fit=%d cal=%d test=%d runs", len(fit), len(cal), len(test))
			}
			out.Main = comparePolicies(fit, cal, test, c, alpha)
			if saveFile != "" || dmnFile != "" || streamNull > 0 {
				m := decide.Fit(fit)
				m.CalibrateAccept(cal, c, true, alpha)
				if saveFile != "" {
					payload, err := json.MarshalIndent(m, "", "  ")
					if err != nil {
						return err
					}
					if err := os.WriteFile(saveFile, payload, 0o644); err != nil {
						return err
					}
				}
				if dmnFile != "" {
					dmn, err := m.ExportDMN()
					if err != nil {
						return err
					}
					if err := os.WriteFile(dmnFile, dmn, 0o644); err != nil {
						return err
					}
				}
				if streamNull > 0 {
					out.Stream, err = evalStream(m, c, test, streamNull, alpha)
					if err != nil {
						return err
					}
				}
			}
			if sweep != "" {
				for _, part := range strings.Split(sweep, ",") {
					miss, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
					if err != nil {
						return fmt.Errorf("--sweep-miss %q: %w", sweep, err)
					}
					sc := c
					sc.Miss = miss
					out.Sweep = append(out.Sweep, comparePolicies(fit, cal, test, sc, alpha))
				}
			}
			if rotateSplits {
				out.Rotation = rotate(bySeed, []int{len(out.FitSeeds), len(out.CalSeeds), len(out.TestSeeds)}, c, alpha)
			}
			if outFile != "" {
				payload, err := json.MarshalIndent(out, "", "  ")
				if err != nil {
					return err
				}
				if err := os.WriteFile(outFile, payload, 0o644); err != nil {
					return err
				}
			}
			return renderDecide(cmd.OutOrStdout(), out)
		},
	}
	f := cmd.Flags()
	f.StringVar(&results, "results", "", "experiments results artifact (required unless --model is given)")
	f.StringVar(&modelFile, "model", "", "saved decision model JSON to export as DMN without refitting (optional)")
	f.Float64Var(&alpha, "alpha", 0.05, "bound on the rate of accepted faulty runs")
	f.StringVar(&fitSeeds, "fit-seeds", "1,2", "seeds the risk model is fitted on")
	f.StringVar(&calSeeds, "cal-seeds", "3", "seeds the accept gate is calibrated on")
	f.StringVar(&testSeeds, "test-seeds", "4,5", "held-out seeds the policies are scored on")
	f.Float64Var(&c.Miss, "miss", c.Miss, "cost of accepting a faulty run")
	f.Float64Var(&c.FalseHalt, "false-halt", c.FalseHalt, "cost of halting a healthy run")
	f.Float64Var(&c.Human, "human", c.Human, "cost of one escalation to a human")
	f.Float64Var(&c.Replay, "replay-cost", c.Replay, "cost of recollecting or replaying a trace")
	f.Float64Var(&c.Judge, "judge-cost", c.Judge, "cost of one LLM judge call")
	f.StringVar(&sweep, "sweep-miss", "", "comma-separated miss costs to sweep (optional)")
	f.BoolVar(&rotateSplits, "rotate", false, "also average every assignment of seeds to fit/calibrate/test with the same split sizes")
	f.Float64Var(&streamNull, "stream-null", 0, "run anytime-valid e-process halting over the test stream under H0: mean <= p0 (optional)")
	f.StringVar(&saveFile, "save", "", "write the model fitted on the fit seeds and gated on the calibration seeds, for serve (optional)")
	f.StringVar(&dmnFile, "dmn", "", "export the fitted policy as an OMG DMN 1.3 XML decision table for audit (optional)")
	f.StringVar(&outFile, "output", "", "also write the full result as JSON to this file (optional)")
	return cmd
}

func renderDecide(w io.Writer, out decideOutput) error {
	c := out.Main.Costs
	var b strings.Builder
	fmt.Fprintf(&b, "decision layer over %s\n", out.Results)
	fmt.Fprintf(&b, "seeds: fit %v (%d runs) | calibrate %v (%d) | test %v (%d)\n",
		out.FitSeeds, out.FitRuns, out.CalSeeds, out.CalRuns, out.TestSeeds, out.TestRuns)
	fmt.Fprintf(&b, "costs: miss=%g falseHalt=%g human=%g replay=%g judge=%g | alpha=%g | %s\n\n",
		c.Miss, c.FalseHalt, c.Human, c.Replay, c.Judge, out.Alpha, gateText(out.Main.Threshold))
	fmt.Fprintf(&b, "%-16s%10s%16s%12s%15s%14s%11s\n",
		"policy", "cost/run", "acceptedFaulty", "accept/100", "falseHalt/100", "escalate/100", "probe/100")
	for _, m := range out.Main.Policies {
		fmt.Fprintf(&b, "%-16s%10.3f%15.1f%%%12.1f%15.1f%14.1f%11.1f\n",
			m.Policy, m.CostPerRun, 100*m.AcceptedFaulty, m.AcceptPer100, m.FalseHaltPer100, m.EscalatePer100, m.ProbePer100)
	}
	if len(out.Sweep) > 0 {
		fmt.Fprintf(&b, "\ncost/run as the miss cost varies\n%-8s%-22s", "miss", "gate")
		for _, m := range out.Sweep[0].Policies {
			fmt.Fprintf(&b, "%16s", m.Policy)
		}
		b.WriteString("\n")
		for _, row := range out.Sweep {
			fmt.Fprintf(&b, "%-8g%-22s", row.Costs.Miss, gateText(row.Threshold))
			for _, m := range row.Policies {
				fmt.Fprintf(&b, "%16.3f", m.CostPerRun)
			}
			b.WriteString("\n")
		}
	}
	if r := out.Rotation; r != nil {
		fmt.Fprintf(&b, "\nmean over %d seed rotations (same split sizes)\n", r.Splits)
		fmt.Fprintf(&b, "%-16s%10s%16s%20s%12s%14s%11s\n",
			"policy", "cost/run", "acceptedFaulty", "max acceptedFaulty", "accept/100", "escalate/100", "probe/100")
		for _, m := range r.Mean {
			fmt.Fprintf(&b, "%-16s%10.3f%15.2f%%%19.2f%%%12.1f%14.1f%11.1f\n",
				m.Policy, m.CostPerRun, 100*m.AcceptedFaulty, 100*r.MaxAcceptedFaulty[m.Policy], m.AcceptPer100, m.EscalatePer100, m.ProbePer100)
		}
	}
	if len(out.Stream) > 0 {
		fmt.Fprintf(&b, "\nanytime-valid streaming e-process (halt at wealth >= 1/alpha = %g)\n", 1/out.Alpha)
		fmt.Fprintf(&b, "%-16s%8s%8s%12s%10s%10s\n", "stream", "null", "mean", "maxWealth", "halted", "haltStep")
		for _, s := range out.Stream {
			step := "-"
			if s.State.Halted {
				step = strconv.Itoa(s.State.HaltStep)
			}
			fmt.Fprintf(&b, "%-16s%8.3f%8.3f%12.3g%10v%10s\n",
				s.Name, s.State.NullRate, s.State.Mean, s.State.MaxWealth, s.State.Halted, step)
		}
	}
	_, err := io.WriteString(w, b.String())
	return err
}

func gateText(threshold float64) string {
	if threshold < 0 {
		return "accept gate closed"
	}
	return fmt.Sprintf("accept gate: risk <= %.4f", threshold)
}
