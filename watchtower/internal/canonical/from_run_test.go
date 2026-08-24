package canonical

import (
	"testing"
	"time"

	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/graph"
	"github.com/aaliyan1230/agent-watchtower/watchtower/internal/model"
)

// runGraph builds a graph.Run mirroring the graph package's synthetic
// trace: a root with two concurrent children, one of them causally
// dependent on the other's tool result via a link edge.
func runGraph() *graph.Run {
	base := time.Unix(100, 0).UTC()
	mk := func(id, parent, name, kind string, links []model.Link) model.Span {
		s := model.Span{
			TraceID: "t", SpanID: id, ParentID: parent, Name: name,
			StartTime: base, EndTime: base.Add(time.Second),
			Status: model.StatusOK, Links: links,
		}
		return s
	}
	spans := []model.Span{
		mk("s0", "", "agent.run", "agent", nil),
		mk("s1", "s0", "chat", "llm", nil),
		mk("s2", "s0", "tool.call", "tool", nil),
		// s3 causally depends on s2 via a data link.
		mk("s3", "s0", "chat", "llm", []model.Link{{SpanID: "s2", Attributes: map[string]string{model.LinkPurpose: "data"}}}),
	}
	run, err := graph.Reconstruct(spans)
	if err != nil {
		panic(err)
	}
	return run
}

func TestFromRunBuildsParentAndCausalEdges(t *testing.T) {
	run := runGraph()
	g := FromRun(run)

	// Parent edges: s1->s0 is the child-parent direction, but our edges
	// point parent -> child, so expect s0->s1, s0->s2, s0->s3 and the
	// causal s2->s3.
	edges := g.Edges()
	have := map[string]bool{}
	for _, e := range edges {
		key := e.From + ">" + e.To
		have[key] = true
	}
	for _, want := range []string{"s0>s1", "s0>s2", "s0>s3", "s2>s3"} {
		if !have[want] {
			t.Fatalf("missing edge %s in %v", want, edges)
		}
	}
	// The link edge keeps its data purpose.
	for _, e := range edges {
		if e.From == "s2" && e.To == "s3" && e.Purpose != "data" {
			t.Fatalf("link edge purpose = %q, want data", e.Purpose)
		}
	}
}

func TestFromRunChecksumIndependentOfReconstructionOrder(t *testing.T) {
	// Reordering only the input span list must not change the canonical
	// checksum once reconstructed — the whole point of CausalTrace.
	base := time.Unix(100, 0).UTC()
	mk := func(id, parent, name string, links []model.Link) model.Span {
		return model.Span{
			TraceID: "t", SpanID: id, ParentID: parent, Name: name,
			StartTime: base, EndTime: base.Add(time.Second),
			Status: model.StatusOK, Links: links,
		}
	}
	normal := []model.Span{
		mk("s0", "", "agent.run", nil),
		mk("s1", "s0", "chat", nil),
		mk("s2", "s0", "chat", []model.Link{{SpanID: "s1", Attributes: map[string]string{model.LinkPurpose: "data"}}}),
	}
	reversed := []model.Span{normal[2], normal[1], normal[0]}

	r1, err1 := graph.Reconstruct(normal)
	r2, err2 := graph.Reconstruct(reversed)
	if err1 != nil || err2 != nil {
		t.Fatalf("reconstruct: %v / %v", err1, err2)
	}
	if FromRun(r1).Checksum() != FromRun(r2).Checksum() {
		t.Fatal("canonical checksum changed when input order changed")
	}
}

func TestFromRunTopoOrdersPerNodeCount(t *testing.T) {
	g := FromRun(runGraph())
	got := g.TopoOrders(20)
	if len(got) == 0 {
		t.Fatal("expected at least one topological ordering")
	}
	for _, order := range got {
		if len(order) != len(g.NodeIDs()) {
			t.Fatalf("ordering %v has %d nodes, want %d", order, len(order), len(g.NodeIDs()))
		}
		validTopo(t, g, order)
	}
	if len(got) > 20 {
		t.Fatalf("topo orders exceed bound: %d", len(got))
	}
}
