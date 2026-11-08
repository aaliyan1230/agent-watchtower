package decide

// Outcome is what a policy did with one run and what it cost.
type Outcome struct {
	First, Final Action
	Cost         float64
	// AcceptRisk is the risk the run was accepted at; meaningful only
	// when Final is Accept.
	AcceptRisk float64
}

type Policy struct {
	Name string
	Act  func(Run) Outcome
}

func finalCost(a Action, r Run, c Costs) float64 {
	switch {
	case a == Accept && r.Faulty:
		return c.Miss
	case a == Halt && !r.Faulty:
		return c.FalseHalt
	case a == Escalate:
		return c.Human
	}
	return 0
}

// BayesPolicy acts by Model.Decide, reads the probe's answer off the
// labeled run, and settles.
func BayesPolicy(name string, m *Model, c Costs, probes bool) Policy {
	return Policy{Name: name, Act: func(r Run) Outcome {
		d := m.Decide(r.Observation(), c, probes)
		o := Outcome{First: d.Action, Final: d.Action, AcceptRisk: d.Risk}
		var s Decision
		switch d.Action {
		case Replay:
			o.Cost += c.Replay
			s = m.Settle(m.Risk(r.ReplaySignature), c)
		case Judge:
			o.Cost += c.Judge
			s = m.Settle(m.JudgePosterior(d.Risk, r.JudgeFlag), c)
		}
		if s.Action != "" {
			o.Final, o.AcceptRisk = s.Action, s.Risk
		}
		o.Cost += finalCost(o.Final, r, c)
		return o
	}}
}

// RulesPolicy is the fixed mapping an operator would write without a
// cost model: pass accepts, abstention escalates, anything else halts.
func RulesPolicy(c Costs) Policy {
	return Policy{Name: "rules", Act: func(r Run) Outcome {
		a := rulesAction(verdictOf(r.Signature))
		return Outcome{First: a, Final: a, Cost: finalCost(a, r, c)}
	}}
}

// JudgeFirstPolicy calls the judge on every judged run and lets any flag
// veto a pass, the usual "LLM-as-judge on everything" setup.
func JudgeFirstPolicy(c Costs) Policy {
	return Policy{Name: "judge-first", Act: func(r Run) Outcome {
		a := rulesAction(verdictOf(r.Signature))
		o := Outcome{First: a}
		if r.Judged {
			o.First, o.Cost = Judge, c.Judge
			if a == Accept && r.JudgeFlag {
				a = Halt
			}
		}
		o.Final = a
		o.Cost += finalCost(a, r, c)
		return o
	}}
}

// OraclePolicy knows the label: the zero-cost lower bound.
func OraclePolicy() Policy {
	return Policy{Name: "oracle", Act: func(r Run) Outcome {
		a := Accept
		if r.Faulty {
			a = Halt
		}
		return Outcome{First: a, Final: a}
	}}
}

func rulesAction(verdict string) Action {
	switch verdict {
	case "PASS":
		return Accept
	case "INCONCLUSIVE":
		return Escalate
	default:
		return Halt
	}
}

// Metrics summarizes a policy over test runs. AcceptedFaulty is a
// fraction of all runs, the quantity the accept gate bounds by alpha;
// the per-100 counts are what an operator staffs and pays for.
type Metrics struct {
	Policy          string  `json:"policy"`
	Runs            int     `json:"runs"`
	CostPerRun      float64 `json:"costPerRun"`
	AcceptedFaulty  float64 `json:"acceptedFaulty"`
	AcceptPer100    float64 `json:"acceptPer100"`
	FalseHaltPer100 float64 `json:"falseHaltPer100"`
	EscalatePer100  float64 `json:"escalatePer100"`
	ProbePer100     float64 `json:"probePer100"`
}

func Evaluate(p Policy, runs []Run) Metrics {
	m := Metrics{Policy: p.Name, Runs: len(runs)}
	if len(runs) == 0 {
		return m
	}
	var cost float64
	var acceptedFaulty, accepts, falseHalts, escalations, probes int
	for _, r := range runs {
		o := p.Act(r)
		cost += o.Cost
		if o.First == Replay || o.First == Judge {
			probes++
		}
		switch o.Final {
		case Accept:
			accepts++
			if r.Faulty {
				acceptedFaulty++
			}
		case Halt:
			if !r.Faulty {
				falseHalts++
			}
		case Escalate:
			escalations++
		}
	}
	n := float64(len(runs))
	m.CostPerRun = cost / n
	m.AcceptedFaulty = float64(acceptedFaulty) / n
	m.AcceptPer100 = 100 * float64(accepts) / n
	m.FalseHaltPer100 = 100 * float64(falseHalts) / n
	m.EscalatePer100 = 100 * float64(escalations) / n
	m.ProbePer100 = 100 * float64(probes) / n
	return m
}
