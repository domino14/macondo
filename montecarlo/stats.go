package montecarlo

import "math"

// Statistic contains statistics per move
type Statistic struct {
	totalIterations int

	// For Welford's algorithm:
	oldM float64
	newM float64
	oldS float64
	newS float64
}

func (s *Statistic) Push(val float64) {
	s.totalIterations++
	if s.totalIterations == 1 {
		s.oldM = val
		s.newM = val
		s.oldS = 0
	} else {
		s.newM = s.oldM + (val-s.oldM)/float64(s.totalIterations)
		s.newS = s.oldS + (val-s.oldM)*(val-s.newM)
		s.oldM = s.newM
		s.oldS = s.newS
	}
}

func (s *Statistic) Mean() float64 {
	if s.totalIterations > 0 {
		return s.newM
	}
	return 0.0
}

func (s *Statistic) Variance() float64 {
	if s.totalIterations <= 1 {
		return 0.0
	}
	return s.newS / float64(s.totalIterations-1)
}

func (s *Statistic) Stdev() float64 {
	return math.Sqrt(s.Variance())
}
