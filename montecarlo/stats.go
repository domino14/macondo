package montecarlo

import "math"

// Statistic contains statistics per move
type Statistic struct {
	totalIterations int
	wins float64

	// For Welford's algorithm:
	oldM float64
	newM float64
	oldS float64
	newS float64
}

func (s *Statistic) Push(val float64) {
	s.totalIterations++
	if s.totalIterations == 1 {
		s.wins = 0
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

func (s *Statistic) PushEval(val float64) {
	s.Push(val)
	if val == 0 {
	    s.wins += 0.5
	}
	if val > 0 {
	    s.wins += 1
	}
}

func (s *Statistic) WinRate() float64 {
	return s.wins / float64(s.totalIterations)
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
