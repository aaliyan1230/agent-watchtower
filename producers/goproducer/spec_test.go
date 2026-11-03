package main

import (
	"testing"
)

func lookup(spec *spanSpec, name string) map[string]any {
	if spec.Name == name {
		return spec.Attrs
	}
	for _, child := range spec.Children {
		if attrs := lookup(child, name); attrs != nil {
			return attrs
		}
	}
	return nil
}

func TestCleanSpecCarriesFullContract(t *testing.T) {
	run, ok := buildSpec("clean")
	if !ok {
		t.Fatal("clean variant must be known")
	}
	root := run.Attrs
	for _, attr := range []string{agentName, wtCompleted, wtFinal, wtOutput, wtTraceClosed} {
		if _, present := root[attr]; !present {
			t.Errorf("root span missing %s", attr)
		}
	}
	if len(run.Events) != 1 || run.Events[0].Name != "supervisor.decision" {
		t.Errorf("root events = %+v, want supervisor.decision", run.Events)
	}
	chat := lookup(run, "chat")
	for _, attr := range []string{genAIOperationName, genAISystem, genAIRequestModel, genAIResponseModel} {
		if _, present := chat[attr]; !present {
			t.Errorf("chat span missing %s", attr)
		}
	}
	tool := lookup(run, "tool.call")
	for _, attr := range []string{toolName, toolCallID, toolResultOK} {
		if _, present := tool[attr]; !present {
			t.Errorf("tool span missing %s", attr)
		}
	}
}

func TestVariantsOmitExactlyOneRequiredField(t *testing.T) {
	tests := []struct {
		variant   string
		span      string
		attribute string
	}{
		{"omit-genai-system", "chat", genAISystem},
		{"omit-tool-result", "tool.call", toolResultOK},
		{"omit-tool-id", "tool.call", toolCallID},
		{"omit-final", "chat", wtFinal},
		{"omit-closed", "agent.run", wtTraceClosed},
	}
	for _, tt := range tests {
		t.Run(tt.variant, func(t *testing.T) {
			run, ok := buildSpec(tt.variant)
			if !ok {
				t.Fatalf("variant %q must be known", tt.variant)
			}
			if _, present := lookup(run, tt.span)[tt.attribute]; present {
				t.Errorf("%s must omit %s on %s spans", tt.variant, tt.attribute, tt.span)
			}
		})
	}
}

func TestUnknownVariantRejected(t *testing.T) {
	if _, ok := buildSpec("nonsense"); ok {
		t.Fatal("unknown variant must not build a spec")
	}
}

func TestAnswerTurnLinksToConsumedTool(t *testing.T) {
	answerTurn := func(run *spanSpec) *spanSpec { return run.Children[1] }
	clean, _ := buildSpec("clean")
	if got := answerTurn(clean).LinksTo; len(got) != 1 || got[0] != lookupSpec(clean, "tool.call").Key {
		t.Fatalf("clean answer turn links = %v, want the tool span", got)
	}
	omitted, _ := buildSpec("omit-link")
	if got := answerTurn(omitted).LinksTo; len(got) != 0 {
		t.Fatalf("omit-link answer turn links = %v, want none", got)
	}
}

func lookupSpec(spec *spanSpec, name string) *spanSpec {
	if spec.Name == name {
		return spec
	}
	for _, child := range spec.Children {
		if found := lookupSpec(child, name); found != nil {
			return found
		}
	}
	return nil
}
