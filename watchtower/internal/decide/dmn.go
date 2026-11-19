package decide

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

const (
	dmnNamespace  = "https://www.omg.org/spec/DMN/20191111/MODEL/"
	feelNamespace = "https://www.omg.org/spec/DMN/20191111/FEEL/"
	modelNS       = "https://github.com/aaliyan1230/agent-watchtower/dmn"
)

// DMNDefinitions is the OMG DMN 1.3 XML root for an exported policy.
type DMNDefinitions struct {
	XMLName            xml.Name       `xml:"https://www.omg.org/spec/DMN/20191111/MODEL/ definitions"`
	ID                 string         `xml:"id,attr"`
	Name               string         `xml:"name,attr"`
	Namespace          string         `xml:"namespace,attr"`
	ExpressionLanguage string         `xml:"expressionLanguage,attr"`
	Description        string         `xml:"description,omitempty"`
	InputData          []DMNInputData `xml:"inputData"`
	Decisions          []DMNDecision  `xml:"decision"`
}

type DMNInputData struct {
	ID   string `xml:"id,attr"`
	Name string `xml:"name,attr"`
}

type DMNDecision struct {
	ID                      string                   `xml:"id,attr"`
	Name                    string                   `xml:"name,attr"`
	Description             string                   `xml:"description,omitempty"`
	InformationRequirements []DMNInfoRequirement     `xml:"informationRequirement,omitempty"`
	DecisionTable           DMNDecisionTable         `xml:"decisionTable"`
}

type DMNInfoRequirement struct {
	ID            string          `xml:"id,attr"`
	RequiredInput DMNElementRef   `xml:"requiredInput"`
}

type DMNElementRef struct {
	Href string `xml:"href,attr"`
}

type DMNDecisionTable struct {
	ID        string      `xml:"id,attr"`
	HitPolicy string      `xml:"hitPolicy,attr"`
	Inputs    []DMNInput  `xml:"input"`
	Outputs   []DMNOutput `xml:"output"`
	Rules     []DMNRule   `xml:"rule"`
}

type DMNInput struct {
	ID              string            `xml:"id,attr"`
	Label           string            `xml:"label,attr,omitempty"`
	InputExpression DMNInputExpr      `xml:"inputExpression"`
}

type DMNInputExpr struct {
	ID      string `xml:"id,attr"`
	TypeRef string `xml:"typeRef,attr"`
	Text    string `xml:"text"`
}

type DMNOutput struct {
	ID      string `xml:"id,attr"`
	Label   string `xml:"label,attr,omitempty"`
	Name    string `xml:"name,attr"`
	TypeRef string `xml:"typeRef,attr"`
}

type DMNRule struct {
	ID            string     `xml:"id,attr"`
	Description   string     `xml:"description,omitempty"`
	InputEntries  []DMNEntry `xml:"inputEntry"`
	OutputEntries []DMNEntry `xml:"outputEntry"`
}

type DMNEntry struct {
	ID   string `xml:"id,attr"`
	Text string `xml:"text"`
}

// ExportDMN renders the fitted and calibrated Model as OMG DMN 1.3 XML
// for external audit. The primary table ("Watchtower First Action")
// reproduces Model.Decide under hitPolicy=FIRST; when judge calibration
// data is present, a second table ("Watchtower Judge Settle") reproduces
// Model.Settle(Model.JudgePosterior(...)) after a judge probe answers.
func (m *Model) ExportDMN() ([]byte, error) {
	c := m.Costs
	if c == (Costs{}) {
		c = DefaultCosts()
	}
	sigs := sortedKeys(m.Signatures, m.Replays)
	verdicts := sortedVerdictKeys(m.Verdicts, m.JudgeByVerdict)

	defs := DMNDefinitions{
		ID:                 "watchtower_decision_policy",
		Name:               "Watchtower Decision Policy",
		Namespace:          modelNS,
		ExpressionLanguage: feelNamespace,
		Description: fmt.Sprintf(
			"Exported from Watchtower decision model: costs(miss=%g, falseHalt=%g, human=%g, replay=%g, judge=%g), alpha=%g, acceptThreshold=%s",
			c.Miss, c.FalseHalt, c.Human, c.Replay, c.Judge, m.Alpha, formatThreshold(m.Threshold),
		),
		InputData: []DMNInputData{
			{ID: "input_signature", Name: "signature"},
			{ID: "input_verdict", Name: "verdict"},
			{ID: "input_canReplay", Name: "canReplay"},
			{ID: "input_canJudge", Name: "canJudge"},
		},
	}

	defs.Decisions = append(defs.Decisions, buildFirstActionDecision(m, c, sigs, verdicts))
	if m.hasJudge() {
		defs.InputData = append(defs.InputData, DMNInputData{ID: "input_judgeFlag", Name: "judgeFlag"})
		defs.Decisions = append(defs.Decisions, buildJudgeSettleDecision(m, c, sigs, verdicts))
	}

	var buf bytes.Buffer
	buf.WriteString(xml.Header)
	enc := xml.NewEncoder(&buf)
	enc.Indent("", "  ")
	if err := enc.Encode(defs); err != nil {
		return nil, err
	}
	buf.WriteByte('\n')
	return buf.Bytes(), nil
}

type firstOutcome struct {
	action   Action
	risk     float64
	gated    bool
	fallback Action
	expected map[Action]float64
}

func (o firstOutcome) key() string {
	return fmt.Sprintf("%s|%t|%s", o.action, o.gated, o.fallback)
}

func evalFirst(m *Model, sig string, canReplay, canJudge bool, c Costs) firstOutcome {
	d := m.Decide(Observation{Signature: sig, CanReplay: canReplay, CanJudge: canJudge}, c, true)
	var fb Action
	if d.Action == Replay {
		fb = m.Settle(d.Risk, c).Action
	}
	return firstOutcome{
		action:   d.Action,
		risk:     d.Risk,
		gated:    d.Gated,
		fallback: fb,
		expected: d.Expected,
	}
}

func buildFirstActionDecision(m *Model, c Costs, sigs, verdicts []string) DMNDecision {
	table := DMNDecisionTable{
		ID:        "dt_first_action",
		HitPolicy: "FIRST",
		Inputs: []DMNInput{
			{ID: "in_fa_signature", Label: "Signature", InputExpression: DMNInputExpr{ID: "ie_fa_signature", TypeRef: "string", Text: "signature"}},
			{ID: "in_fa_verdict", Label: "Verdict", InputExpression: DMNInputExpr{ID: "ie_fa_verdict", TypeRef: "string", Text: "verdict"}},
			{ID: "in_fa_canReplay", Label: "Can Replay", InputExpression: DMNInputExpr{ID: "ie_fa_canReplay", TypeRef: "boolean", Text: "canReplay"}},
			{ID: "in_fa_canJudge", Label: "Can Judge", InputExpression: DMNInputExpr{ID: "ie_fa_canJudge", TypeRef: "boolean", Text: "canJudge"}},
		},
		Outputs: []DMNOutput{
			{ID: "out_fa_action", Label: "Action", Name: "action", TypeRef: "string"},
			{ID: "out_fa_risk", Label: "Risk", Name: "risk", TypeRef: "number"},
			{ID: "out_fa_gated", Label: "Gated", Name: "gated", TypeRef: "boolean"},
			{ID: "out_fa_fallback", Label: "Fallback", Name: "fallback", TypeRef: "string"},
		},
	}

	ruleIdx := 0
	addRule := func(sigExpr, verdictExpr, replayExpr, judgeExpr, desc string, o firstOutcome) {
		ruleIdx++
		id := fmt.Sprintf("rule_fa_%d", ruleIdx)
		table.Rules = append(table.Rules, DMNRule{
			ID:          id,
			Description: desc,
			InputEntries: []DMNEntry{
				{ID: id + "_in_1", Text: sigExpr},
				{ID: id + "_in_2", Text: verdictExpr},
				{ID: id + "_in_3", Text: replayExpr},
				{ID: id + "_in_4", Text: judgeExpr},
			},
			OutputEntries: []DMNEntry{
				{ID: id + "_out_1", Text: feelString(string(o.action))},
				{ID: id + "_out_2", Text: formatNum(o.risk)},
				{ID: id + "_out_3", Text: strconv.FormatBool(o.gated)},
				{ID: id + "_out_4", Text: feelString(string(o.fallback))},
			},
		})
	}

	emitCompressed := func(sigExpr, verdictExpr, evalSig, baseDesc string, hasReplay bool) {
		tt := evalFirst(m, evalSig, true, true, c)
		tf := evalFirst(m, evalSig, true, false, c)
		ft := evalFirst(m, evalSig, false, true, c)
		ff := evalFirst(m, evalSig, false, false, c)

		switch {
		case tt.key() == tf.key() && tt.key() == ft.key() && tt.key() == ff.key():
			addRule(sigExpr, verdictExpr, "-", "-", auditDesc(baseDesc, tt), tt)
		case !hasReplay || (tt.key() == ft.key() && tf.key() == ff.key()):
			addRule(sigExpr, verdictExpr, "-", "true", auditDesc(baseDesc, ft), ft)
			addRule(sigExpr, verdictExpr, "-", "false", auditDesc(baseDesc, ff), ff)
		case tt.key() == tf.key() && ft.key() == ff.key():
			addRule(sigExpr, verdictExpr, "true", "-", auditDesc(baseDesc, tt), tt)
			addRule(sigExpr, verdictExpr, "false", "-", auditDesc(baseDesc, ft), ft)
		default:
			addRule(sigExpr, verdictExpr, "true", "true", auditDesc(baseDesc, tt), tt)
			addRule(sigExpr, verdictExpr, "true", "false", auditDesc(baseDesc, tf), tf)
			addRule(sigExpr, verdictExpr, "false", "true", auditDesc(baseDesc, ft), ft)
			addRule(sigExpr, verdictExpr, "false", "false", auditDesc(baseDesc, ff), ff)
		}
	}

	for _, sig := range sigs {
		cnt := m.Signatures[sig]
		desc := fmt.Sprintf("observed signature (%d/%d faulty)", cnt.Faulty, cnt.Total)
		emitCompressed(feelString(sig), "-", sig, desc, len(m.Replays[sig]) > 0)
	}
	for _, v := range verdicts {
		cnt := m.Verdicts[v]
		desc := fmt.Sprintf("unseen signature fallback for verdict %s (%d/%d faulty)", v, cnt.Faulty, cnt.Total)
		emitCompressed("-", feelString(v), v+"|__unseen__", desc, false)
	}
	emitCompressed("-", "-", "__unseen__|", "default fallback for unseen verdict (prior 0.5)", false)

	return DMNDecision{
		ID:          "decision_first_action",
		Name:        "Watchtower First Action",
		Description: "Selects the minimum-expected-cost first action for an observed deterministic signature, subject to the conformal accept gate.",
		InformationRequirements: []DMNInfoRequirement{
			{ID: "ir_fa_1", RequiredInput: DMNElementRef{Href: "#input_signature"}},
			{ID: "ir_fa_2", RequiredInput: DMNElementRef{Href: "#input_verdict"}},
			{ID: "ir_fa_3", RequiredInput: DMNElementRef{Href: "#input_canReplay"}},
			{ID: "ir_fa_4", RequiredInput: DMNElementRef{Href: "#input_canJudge"}},
		},
		DecisionTable: table,
	}
}

func buildJudgeSettleDecision(m *Model, c Costs, sigs, verdicts []string) DMNDecision {
	table := DMNDecisionTable{
		ID:        "dt_judge_settle",
		HitPolicy: "FIRST",
		Inputs: []DMNInput{
			{ID: "in_js_signature", Label: "Signature", InputExpression: DMNInputExpr{ID: "ie_js_signature", TypeRef: "string", Text: "signature"}},
			{ID: "in_js_verdict", Label: "Verdict", InputExpression: DMNInputExpr{ID: "ie_js_verdict", TypeRef: "string", Text: "verdict"}},
			{ID: "in_js_judgeFlag", Label: "Judge Flag", InputExpression: DMNInputExpr{ID: "ie_js_judgeFlag", TypeRef: "boolean", Text: "judgeFlag"}},
		},
		Outputs: []DMNOutput{
			{ID: "out_js_action", Label: "Action", Name: "action", TypeRef: "string"},
			{ID: "out_js_posterior", Label: "Posterior", Name: "posterior", TypeRef: "number"},
			{ID: "out_js_gated", Label: "Gated", Name: "gated", TypeRef: "boolean"},
		},
	}

	ruleIdx := 0
	emitPair := func(sigExpr, verdictExpr, evalSig, label string) {
		prior := m.Risk(evalSig)
		tpr, fpr := m.JudgeRates(evalSig)
		for _, flagged := range []bool{true, false} {
			post := m.JudgePosterior(evalSig, prior, flagged)
			settled := m.Settle(post, c)
			ruleIdx++
			id := fmt.Sprintf("rule_js_%d", ruleIdx)
			desc := fmt.Sprintf("%s; prior=%.4f tpr=%.4f fpr=%.4f flagged=%t -> posterior=%.4f",
				label, prior, tpr, fpr, flagged, post)
			if settled.Gated {
				desc += " (gated)"
			}
			table.Rules = append(table.Rules, DMNRule{
				ID:          id,
				Description: desc,
				InputEntries: []DMNEntry{
					{ID: id + "_in_1", Text: sigExpr},
					{ID: id + "_in_2", Text: verdictExpr},
					{ID: id + "_in_3", Text: strconv.FormatBool(flagged)},
				},
				OutputEntries: []DMNEntry{
					{ID: id + "_out_1", Text: feelString(string(settled.Action))},
					{ID: id + "_out_2", Text: formatNum(post)},
					{ID: id + "_out_3", Text: strconv.FormatBool(settled.Gated)},
				},
			})
		}
	}

	for _, sig := range sigs {
		emitPair(feelString(sig), "-", sig, "signature "+sig)
	}
	for _, v := range verdicts {
		emitPair("-", feelString(v), v+"|__unseen__", "verdict "+v+" fallback")
	}
	emitPair("-", "-", "__unseen__|", "default unseen verdict fallback")

	return DMNDecision{
		ID:          "decision_judge_settle",
		Name:        "Watchtower Judge Settle",
		Description: "Settles the terminal action after a judge probe answers, using per-verdict Bayes posterior and the conformal accept gate.",
		InformationRequirements: []DMNInfoRequirement{
			{ID: "ir_js_1", RequiredInput: DMNElementRef{Href: "#input_signature"}},
			{ID: "ir_js_2", RequiredInput: DMNElementRef{Href: "#input_verdict"}},
			{ID: "ir_js_3", RequiredInput: DMNElementRef{Href: "#input_judgeFlag"}},
		},
		DecisionTable: table,
	}
}

func auditDesc(base string, o firstOutcome) string {
	var costs []string
	for _, a := range []Action{Accept, Halt, Escalate, Replay, Judge} {
		if v, ok := o.expected[a]; ok {
			costs = append(costs, fmt.Sprintf("%s=%.3f", a, v))
		}
	}
	desc := fmt.Sprintf("%s; risk=%.4f; expected[%s]", base, o.risk, strings.Join(costs, " "))
	if o.gated {
		desc += "; accept gated"
	}
	return desc
}

func sortedKeys(sigs map[string]Counts, replays map[string]map[string]int) []string {
	seen := map[string]bool{}
	for k := range sigs {
		seen[k] = true
	}
	for k := range replays {
		seen[k] = true
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedVerdictKeys(verdicts map[string]Counts, judgeByVerdict map[string]JudgeCounts) []string {
	seen := map[string]bool{}
	for k := range verdicts {
		seen[k] = true
	}
	for k := range judgeByVerdict {
		seen[k] = true
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func feelString(s string) string {
	return strconv.Quote(s)
}

func formatNum(x float64) string {
	return strconv.FormatFloat(x, 'f', 6, 64)
}

func formatThreshold(t float64) string {
	if t < 0 {
		return "closed"
	}
	return fmt.Sprintf("%.6f", t)
}
