package report

import (
	"testing"
	"time"
)

func TestStorePutGet(t *testing.T) {
	s := NewStore(time.Minute)
	r := &Report{TraceID: "t1", Verdict: VerdictFail, GeneratedAt: time.Now()}
	s.Put("t1", r)
	got, ok := s.Get("t1")
	if !ok || got.Verdict != VerdictFail {
		t.Fatalf("Get() = %+v, %v", got, ok)
	}
	if _, ok := s.Get("nope"); ok {
		t.Fatal("Get() of unknown trace should miss")
	}
}

func TestStoreTTLExpiry(t *testing.T) {
	s := NewStore(10 * time.Millisecond)
	s.Put("t1", &Report{TraceID: "t1", GeneratedAt: time.Now()})
	time.Sleep(25 * time.Millisecond) // outlive the TTL, then sweep lazily
	if _, ok := s.Get("t1"); ok {
		t.Fatal("expired report should be gone")
	}
}

func TestStoreOverwrite(t *testing.T) {
	s := NewStore(time.Minute)
	s.Put("t1", &Report{TraceID: "t1", Verdict: VerdictPass, GeneratedAt: time.Now()})
	s.Put("t1", &Report{TraceID: "t1", Verdict: VerdictFail, GeneratedAt: time.Now()})
	got, _ := s.Get("t1")
	if got.Verdict != VerdictFail {
		t.Fatalf("verdict = %s, want FAIL (latest wins)", got.Verdict)
	}
}
