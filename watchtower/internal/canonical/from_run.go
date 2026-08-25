package canonical

import (
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
)

// FromRun reconstructs the causal DAG of a reconstructed run into a
// canonical Graph. Parent-child edges are drawn from each step's ParentID
// (the tree already implicit in the run), and non-parent causal edges come
// from Run.CausalEdges (the OTLP link edges). Both kinds are added as
// directed From -> To dependencies, so the Graph carries the full
// happens-before relation the verifier reasons over — independent of the
// order the flat span list arrived in.
//
// Purpose labels: link edges keep their Purpose ("data" / "control");
// parent edges are stamped "" so they are distinguishable but still
// participate in serialization.
func FromRun(run *graph.Run) *Graph {
	g := &Graph{}
	for _, step := range run.Steps {
		g.AddNode(step.SpanID, step.Name, string(step.Kind))
	}
	for _, step := range run.Steps {
		if step.ParentID != "" {
			g.AddEdge(step.ParentID, step.SpanID, "")
		}
	}
	for _, edge := range run.CausalEdges {
		g.AddEdge(edge.From, edge.To, edge.Purpose)
	}
	return g
}
