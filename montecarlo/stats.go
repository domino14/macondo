package montecarlo

import "math"

// Statistic contains statistics per move
type Statistic struct {
	totalIterations int
	losses int
	wins int

	// For Welford's algorithm:
	oldM float64
	newM float64
	oldS float64
	newS float64
}

func (s *Statistic) Push(val float64) {
	s.totalIterations++
	if s.totalIterations == 1 {
		s.losses = 0
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
	if val < 0 {
	    s.losses++
	}
	if val > 0 {
	    s.wins++
	}
}

func (s *Statistic) Scale() float64 {
	// Future work: endgame proven wins/losses can return 1.0 or 0.0
	return float64(1 + s.wins) / float64(2 + s.losses + s.wins)
}

func (s *Statistic) Equity() float64 {
	if s.totalIterations > 0 {
		mean := s.Mean()
		scale := s.Scale()
		if mean < 0 {
			return (1.0 - scale) * mean
		} else {
			return scale * mean
		}
	}
	return 0.0
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
