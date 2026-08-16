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
	for _, attr := range []string{agentName, wtCompleted, wtFinal, wtOutput} {
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
