package verify

import (
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

// CheckStatus flags spans that ended with status=error: an errored
// span is direct evidence the run failed, whatever the cause (provider
// timeout, crash, refused tool). The one verifier that reads the OTLP
// status field rather than semconv attributes.
func CheckStatus(run *graph.Run) []Finding {
	var out []Finding
	for _, st := range run.Steps {
		if st.Status != model.StatusError {
			continue
		}
		msg := st.StatusMsg
		if msg == "" {
			msg = st.Name
		}
		out = append(out, Finding{
			Verifier:   "status",
			Severity:   SeverityCritical,
			Message:    "span ended with status=error: " + msg,
			SpanIDs:    []string{st.SpanID},
			Timestamps: []string{st.StartTime.UTC().Format("2006-01-02T15:04:05Z")},
			Value:      msg,
		})
	}
	return out
}
