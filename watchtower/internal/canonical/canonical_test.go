package canonical

import (
	"strings"
	"testing"
)

// triangle builds the classic diamond DAG used across the package:
// a -> b, a -> c, b -> d, c -> d.
func diamond() *Graph {
	g := &Graph{}
	for _, id := range []string{"a", "b", "c", "d"} {
		g.AddNode(id, id, "tool")
	}
	g.AddEdge("a", "b", "")
	g.AddEdge("a", "c", "")
	g.AddEdge("b", "d", "")
	g.AddEdge("c", "d", "")
	return g
}

func TestRenderIsSortedAndIndependentOfInsertionOrder(t *testing.T) {
	g1 := &Graph{}
	g1.AddNode("b", "tool", "tool")
	g1.AddNode("a", "root", "agent")
	g1.AddEdge("a", "b", "data")

	// Same structure, different insertion order.
	g2 := &Graph{}
	g2.AddNode("a", "root", "agent")
	g2.AddNode("b", "tool", "tool")
	g2.AddEdge("a", "b", "data")

	if g1.Render() != g2.Render() {
		t.Fatalf("render depends on insertion order:\n%q\n!=\n%q", g1.Render(), g2.Render())
	}
	if g1.Checksum() != g2.Checksum() {
		t.Fatalf("checksum depends on insertion order: %s != %s", g1.Checksum(), g2.Checksum())
	}
	got := g1.Render()
	if !strings.Contains(got, "a\tagent\troot") || !strings.Contains(got, "b\ttool\ttool") {
		t.Fatalf("render missing nodes:\n%s", got)
	}
	if !strings.Contains(got, "a > b\tdata") {
		t.Fatalf("render missing edge:\n%s", got)
	}
}

func TestRenderDeduplicatesEdges(t *testing.T) {
	g := diamond()
	g.AddEdge("a", "b", "") // duplicate
	if n := len(g.Edges()); n != 4 {
		t.Fatalf("edges = %d, want 4 (dedup)", n)
	}
	out := g.Render()
	if strings.Count(out, "a > b\t") != 1 {
		t.Fatalf("duplicate edge leaked into render:\n%s", out)
	}
}

func TestChecksumStableAcrossSiblingReordering(t *testing.T) {
	// A sibling order (b,c) vs (c,b) with no cross-dependency must not
	// change the canonical checksum — the contract the experiment relies on.
	ab := &Graph{}
	for _, id := range []string{"a", "b", "c"} {
		ab.AddNode(id, id, "tool")
	}
	ab.AddEdge("a", "b", "")
	ab.AddEdge("a", "c", "")
	ac := &Graph{}
	for _, id := range []string{"a", "c", "b"} {
		ac.AddNode(id, id, "tool")
	}
	ac.AddEdge("a", "c", "")
	ac.AddEdge("a", "b", "")
	if ab.Checksum() != ac.Checksum() {
		t.Fatalf("sibling order changed checksum: %s != %s", ab.Checksum(), ac.Checksum())
	}
}

func validTopo(t *testing.T, g *Graph, order []string) {
	t.Helper()
	if len(order) != len(g.NodeIDs()) {
		t.Fatalf("order %v has wrong length", order)
	}
	seen := make(map[string]bool, len(order))
	pos := make(map[string]int, len(order))
	for i, id := range order {
		if seen[id] {
			t.Fatalf("id %q appears twice in %v", id, order)
		}
		seen[id] = true
		pos[id] = i
	}
	for _, e := range g.Edges() {
		if pos[e.From] > pos[e.To] {
			t.Fatalf("topo order %v violates edge %s->%s", order, e.From, e.To)
		}
	}
}

func TestTopoOrdersDiamond(t *testing.T) {
	g := diamond()
	got := g.TopoOrders(20)
	if len(got) < 2 {
		t.Fatalf("diamond should have 2 valid orderings, got %d", len(got))
	}
	for _, order := range got {
		validTopo(t, g, order)
	}
	// Diamond orderings are exactly [a b c d] and [a c b d].
	want := [][]string{{"a", "b", "c", "d"}, {"a", "c", "b", "d"}}
	for i, order := range got {
		if !eq(order, want[i]) {
			t.Fatalf("ordering %d = %v, want %v", i, order, want[i])
		}
	}
}

func TestTopoOrdersRespectsBoundAndIsDeterministic(t *testing.T) {
	// Disjoint pair of chains gives 4 orderings; bound should cap.
	g := &Graph{}
	for _, id := range []string{"a", "b", "x", "y"} {
		g.AddNode(id, id, "tool")
	}
	g.AddEdge("a", "b", "")
	g.AddEdge("x", "y", "")
	got := g.TopoOrders(3)
	if len(got) != 3 {
		t.Fatalf("bound 3 not respected: got %d", len(got))
	}
	again := g.TopoOrders(3)
	for i := range got {
		if !eq(got[i], again[i]) {
			t.Fatalf("topo enumeration not deterministic: %v != %v", got, again)
		}
	}
}

func TestTopoOrdersCycleReturnsNil(t *testing.T) {
	g := &Graph{}
	for _, id := range []string{"a", "b"} {
		g.AddNode(id, id, "tool")
	}
	g.AddEdge("a", "b", "")
	g.AddEdge("b", "a", "")
	if got := g.TopoOrders(10); got != nil {
		t.Fatalf("cyclic graph should yield no orderings, got %v", got)
	}
}

func TestTopoOrdersEmptyAndNonPositiveLimit(t *testing.T) {
	empty := &Graph{}
	if got := empty.TopoOrders(10); got != nil {
		t.Fatalf("empty graph should yield nil, got %v", got)
	}
	g := diamond()
	if got := g.TopoOrders(0); got != nil {
		t.Fatalf("limit<=0 should yield nil, got %v", got)
	}
	if got := g.TopoOrders(-1); got != nil {
		t.Fatalf("negative limit should yield nil, got %v", got)
	}
}

func eq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
