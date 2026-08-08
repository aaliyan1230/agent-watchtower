package report

import (
	"sync"
	"time"
)

// Store is the in-memory report registry: serve keeps the latest report
// per trace id here so clients (the harness) can retrieve a verdict
// after submitting spans. Reports expire after a TTL — the store holds
// "recent verdicts", not an archive (that is a later-phase concern).
type Store struct {
	mu        sync.Mutex
	reports   map[string]*Report
	ttl       time.Duration
	lastSweep time.Time
}

func NewStore(ttl time.Duration) *Store {
	return &Store{reports: make(map[string]*Report), ttl: ttl}
}

// Put records the verdict for a trace id.
func (s *Store) Put(traceID string, r *Report) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweepLocked(time.Now())
	s.reports[traceID] = r
}

// Get retrieves the verdict for a trace id, reporting whether it is
// still retained.
func (s *Store) Get(traceID string) (*Report, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweepLocked(time.Now())
	r, ok := s.reports[traceID]
	return r, ok
}

// sweepLocked removes expired reports. Sweeping lazily on access keeps
// the store O(1) per operation on average; the cost is bounded by the
// number of reports written in one TTL window.
func (s *Store) sweepLocked(now time.Time) {
	if s.ttl <= 0 || now.Sub(s.lastSweep) < s.ttl {
		return
	}
	for id, r := range s.reports {
		if now.Sub(r.GeneratedAt) > s.ttl {
			delete(s.reports, id)
		}
	}
	s.lastSweep = now
}
