package decide

import "sort"

// CalibrateAccept sets the accept gate by conformal risk control
// (Angelopoulos et al., 2022). The loss of a calibration run at
// threshold t is 1 when the ungated policy accepts it at risk <= t and
// it is faulty; the gate is the largest t with (sum of losses + 1) /
// (n + 1) <= alpha. Then the expected rate of accepted faulty runs is at
// most alpha, provided the calibration runs are exchangeable with future
// runs and were not used to fit the model. With too few runs to certify
// even zero misses the gate stays closed.
func (m *Model) CalibrateAccept(cal []Run, c Costs, probes bool, alpha float64) float64 {
	open := *m
	open.Threshold = 1
	policy := BayesPolicy("", &open, c, probes)
	type accepted struct {
		risk   float64
		faulty bool
	}
	var acc []accepted
	for _, r := range cal {
		if o := policy.Act(r); o.Final == Accept {
			acc = append(acc, accepted{o.AcceptRisk, r.Faulty})
		}
	}
	sort.Slice(acc, func(i, j int) bool { return acc[i].risk < acc[j].risk })
	n := float64(len(cal))
	threshold, loss := -1.0, 0
	for i, a := range acc {
		if a.faulty {
			loss++
		}
		if i+1 < len(acc) && acc[i+1].risk == a.risk {
			continue
		}
		if (float64(loss)+1)/(n+1) > alpha {
			break
		}
		threshold = a.risk
	}
	m.Threshold = threshold
	return threshold
}
