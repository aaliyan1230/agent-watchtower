package verify

import (
	"encoding/json"
	"fmt"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// FieldType enumerates the JSON types a contract field may demand.
type FieldType string

const (
	TypeString FieldType = "string"
	TypeInt    FieldType = "int"
	TypeFloat  FieldType = "float"
	TypeBool   FieldType = "bool"
	TypeArray  FieldType = "array"
	TypeObject FieldType = "object"
)

// FieldSpec declares one field of a structured-output contract.
type FieldSpec struct {
	Required bool      `json:"required"`
	Type     FieldType `json:"type"`
}

// Contract is a structured-output contract a worker agent is supposed to
// honor: the harness stamps watchtower.contract with the contract name
// and watchtower.output with the raw model output; this verifier parses
// the output and checks every field.
type Contract struct {
	Name   string               `json:"name"`
	Fields map[string]FieldSpec `json:"fields"`
}

// CheckSchema verifies structured outputs against declared contracts.
// With no contracts configured the check is disabled entirely — a
// stamped contract name is only meaningful against a declared contract
// (opt-in, like every other check). Steps without a contract attribute
// are not checked; a step referencing an unknown contract is itself a
// finding (the harness stamped a contract it cannot honor).
func CheckSchema(run *graph.Run, contracts []Contract) []Finding {
	if len(contracts) == 0 {
		return nil
	}
	byName := make(map[string]Contract, len(contracts))
	for _, c := range contracts {
		byName[c.Name] = c
	}
	var out []Finding
	for _, step := range run.Steps {
		if step.Contract == "" {
			continue
		}
		contract, ok := byName[step.Contract]
		if !ok {
			out = append(out, Finding{
				Verifier: "schema", Severity: SeverityCritical,
				Message:    fmt.Sprintf("unknown contract %q", step.Contract),
				SpanIDs:    []string{step.SpanID},
				Timestamps: []string{step.StartTime.UTC().Format("2006-01-02T15:04:05Z")},
				Value:      step.Contract,
			})
			continue
		}
		if step.Output == "" {
			out = append(out, Finding{
				Verifier: "schema", Severity: SeverityCritical,
				Message:    fmt.Sprintf("contract %q declared but no output recorded", contract.Name),
				SpanIDs:    []string{step.SpanID},
				Timestamps: []string{step.StartTime.UTC().Format("2006-01-02T15:04:05Z")},
			})
			continue
		}
		out = append(out, checkOutput(step, contract)...)
	}
	return out
}

func checkOutput(step graph.Step, contract Contract) []Finding {
	var doc any
	if err := json.Unmarshal([]byte(step.Output), &doc); err != nil {
		return []Finding{{
			Verifier: "schema", Severity: SeverityCritical,
			Message:    fmt.Sprintf("output for contract %q is not valid JSON", contract.Name),
			SpanIDs:    []string{step.SpanID},
			Timestamps: []string{step.StartTime.UTC().Format("2006-01-02T15:04:05Z")},
			Value:      truncate(step.Output, 200),
		}}
	}
	obj, ok := doc.(map[string]any)
	if !ok {
		return []Finding{{
			Verifier: "schema", Severity: SeverityCritical,
			Message:    fmt.Sprintf("output for contract %q is not a JSON object", contract.Name),
			SpanIDs:    []string{step.SpanID},
			Timestamps: []string{step.StartTime.UTC().Format("2006-01-02T15:04:05Z")},
		}}
	}
	var out []Finding
	for name, spec := range contract.Fields {
		v, present := obj[name]
		if !present {
			if spec.Required {
				out = append(out, Finding{
					Verifier: "schema", Severity: SeverityCritical,
					Message:    fmt.Sprintf("required field %q missing in %q output", name, contract.Name),
					SpanIDs:    []string{step.SpanID},
					Timestamps: []string{step.StartTime.UTC().Format("2006-01-02T15:04:05Z")},
				})
			}
			continue
		}
		if !typeMatches(v, spec.Type) {
			out = append(out, Finding{
				Verifier: "schema", Severity: SeverityCritical,
				Message:    fmt.Sprintf("field %q has type %T, want %s (contract %q)", name, v, spec.Type, contract.Name),
				SpanIDs:    []string{step.SpanID},
				Timestamps: []string{step.StartTime.UTC().Format("2006-01-02T15:04:05Z")},
				Value:      truncate(fmt.Sprintf("%v", v), 200),
			})
		}
	}
	return out
}

// typeMatches checks a decoded JSON value against a field type. JSON
// numbers all decode as float64; TypeInt additionally requires a whole
// number.
func typeMatches(v any, t FieldType) bool {
	switch t {
	case TypeString:
		_, ok := v.(string)
		return ok
	case TypeInt:
		f, ok := v.(float64)
		return ok && f == float64(int64(f))
	case TypeFloat:
		_, ok := v.(float64)
		return ok
	case TypeBool:
		_, ok := v.(bool)
		return ok
	case TypeArray:
		_, ok := v.([]any)
		return ok
	case TypeObject:
		_, ok := v.(map[string]any)
		return ok
	}
	return false
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
