// Package canonical renders a reconstructed agent run as a canonical
// causal DAG and enumerates its valid topological serializations.
//
// The whole CausalTrace claim rests on order-invariance: two flat span
// logs of the same concurrent execution differ only in arrival order, so
// verification must not depend on that order. This package gives the two
// pieces the experiment needs to measure that:
//
//   - Render/Checksum: a deterministic byte-for-byte form of the causal
//     DAG (parent edges + non-parent link edges), sorted only by span and
//     edge identity — never by arrival order. Two orderings of the same
//     trace must produce the identical checksum.
//   - TopoOrders: all valid topological orderings of that DAG, bounded to
//     a small budget (the plan's 10–20 per trace), so a sweep can ask
//     "does the verdict flip across equivalent serializations?"
//
// Determinism is the contract: the same causal structure always renders
// to the same bytes, regardless of the order the flat span list arrived
// in. The experiment hashes this to test order-invariance end to end.
package canonical

import (
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"
)

// Edge is one directed causal dependency in the DAG: From must be
// verified before To. Parent-child spans and explicit link edges both
// contribute edges.
type Edge struct {
	From    string
	To      string
	Purpose string // "" for parent edges; link purpose (e.g. "data") otherwise
}

// Node is one span participating in the DAG, carrying enough identity
// for the canonical render to be self-describing.
type Node struct {
	ID   string
	Name string
	Kind string // "agent" | "llm" | "tool"
}

// Graph is the causal DAG as a set of nodes and directed edges. Edges
// are deduplicated on insert; the render and search funcs sort them, so
// construction order never matters.
type Graph struct {
	nodes map[string]Node
	edges map[string]Edge // key = from + "\x00" + to + purpose

	// adjacency + reverse index built lazily by topological searches.
	out  map[string][]string
	ind  map[string]int
	once bool
}

// AddNode registers a span in the graph. Duplicate ids collapse to the
// first definition (the caller owns deduplication concerns upstream).
func (g *Graph) AddNode(id, name, kind string) {
	if g.nodes == nil {
		g.nodes = make(map[string]Node)
		g.edges = make(map[string]Edge)
	}
	if _, ok := g.nodes[id]; !ok {
		g.nodes[id] = Node{ID: id, Name: name, Kind: kind}
	}
}

// AddEdge records a directed dependency From -> To (deduplicated).
func (g *Graph) AddEdge(from, to, purpose string) {
	if g.nodes == nil {
		g.nodes = make(map[string]Node)
		g.edges = make(map[string]Edge)
	}
	g.nodes[from] = g.nodes[from]
	if _, ok := g.nodes[from]; !ok {
		g.nodes[from] = Node{ID: from}
	}
	g.nodes[to] = g.nodes[to]
	if _, ok := g.nodes[to]; !ok {
		g.nodes[to] = Node{ID: to}
	}
	key := from + "\x00" + to + "\x00" + purpose
	if _, ok := g.edges[key]; !ok {
		g.edges[key] = Edge{From: from, To: to, Purpose: purpose}
		g.once = false
	}
}

// NodeIDs returns the node ids in sorted order.
func (g *Graph) NodeIDs() []string {
	ids := make([]string, 0, len(g.nodes))
	for id := range g.nodes {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Edges returns the distinct edges sorted by (from, to, purpose).
func (g *Graph) Edges() []Edge {
	out := make([]Edge, 0, len(g.edges))
	for _, e := range g.edges {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].From != out[j].From {
			return out[i].From < out[j].From
		}
		if out[i].To != out[j].To {
			return out[i].To < out[j].To
		}
		return out[i].Purpose < out[j].Purpose
	})
	return out
}

// build prepares the adjacency/indegree maps for topological searches.
// Because edges are deduplicated, rebuilding is idempotent.
func (g *Graph) build() {
	if g.once {
		return
	}
	g.out = make(map[string][]string)
	g.ind = make(map[string]int)
	for id := range g.nodes {
		g.ind[id] = 0
	}
	for _, e := range g.edges {
		g.out[e.From] = append(g.out[e.From], e.To)
		g.ind[e.To]++
	}
	for from := range g.out {
		sort.Strings(g.out[from])
	}
	g.once = true
}

// canTopo reports whether the DAG has at least one valid topological
// ordering (i.e. is acyclic). A cycle renders to no serialization.
func (g *Graph) canTopo() bool {
	g.build()
	indegree := make(map[string]int, len(g.ind))
	for id, d := range g.ind {
		indegree[id] = d
	}
	queue := make([]string, 0, len(g.nodes))
	for id, d := range indegree {
		if d == 0 {
			queue = append(queue, id)
		}
	}
	sort.Strings(queue)
	seen := 0
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		seen++
		for _, next := range g.out[cur] {
			indegree[next]--
			if indegree[next] == 0 {
				queue = append(queue, next)
			}
		}
		sort.Strings(queue)
	}
	return seen == len(g.nodes)
}

// Render builds the canonical text form of the DAG: one line per node
// (id, kind, name) followed by its lexicographically sorted outgoing
// edges, in sorted node order. Byte-for-byte identical for any
// construction order — the order-invariance contract.
func (g *Graph) Render() string {
	var b strings.Builder
	for _, id := range g.NodeIDs() {
		n := g.nodes[id]
		fmt.Fprintf(&b, "%s\t%s\t%s\n", id, n.Kind, n.Name)
	}
	for _, e := range g.Edges() {
		fmt.Fprintf(&b, "%s > %s\t%s\n", e.From, e.To, e.Purpose)
	}
	return b.String()
}

// Checksum returns the sha256 of the canonical render.
func (g *Graph) Checksum() string {
	sum := sha256.Sum256([]byte(g.Render()))
	return fmt.Sprintf("%x", sum)
}

// TopoOrders returns up to limit distinct valid topological orderings,
// each a deterministic comma-joined list of node ids. Generation is a
// depth-first search over the sorted zero-indegree frontier, so the
// same DAG always yields the same first `limit` orderings. Returns nil
// when the graph is cyclic or has no nodes.
func (g *Graph) TopoOrders(limit int) [][]string {
	if limit <= 0 || len(g.nodes) == 0 {
		return nil
	}
	g.build()
	if !g.canTopo() {
		return nil
	}

	type state struct {
		order []string
		ind   map[string]int
	}
	zeroIndegree := func(ind map[string]int) []string {
		var zero []string
		for id, d := range ind {
			if d == 0 {
				zero = append(zero, id)
			}
		}
		sort.Strings(zero)
		return zero
	}

	indegree := make(map[string]int, len(g.ind))
	for id, d := range g.ind {
		indegree[id] = d
	}

	var out [][]string
	stack := []state{{order: nil, ind: indegree}}
	for len(stack) > 0 && len(out) < limit {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if len(cur.order) == len(g.nodes) {
			out = append(out, cur.order)
			continue
		}
		frontier := zeroIndegree(cur.ind)
		// Push in reverse so the smallest id is expanded (and thus
		// completes) first, keeping the enumeration deterministic.
		for i := len(frontier) - 1; i >= 0; i-- {
			cand := frontier[i]
			next := make(map[string]int, len(cur.ind))
			for id, d := range cur.ind {
				next[id] = d
			}
			next[cand] = -1 // consumed; no longer available
			for _, child := range g.out[cand] {
				next[child]--
			}
			order := make([]string, len(cur.order)+1)
			copy(order, cur.order)
			order[len(cur.order)] = cand
			stack = append(stack, state{order: order, ind: next})
		}
	}
	return out
}
