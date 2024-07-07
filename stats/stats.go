package stats

import "math"

const (
	Epsilon = 1e-6
)

func FuzzyEqual(a, b float64) bool {
	return math.Abs(a-b) < Epsilon
}

// Statistic contains statistics per move
type Statistic struct {
	totalIterations int
	last            float64

	// For Welford's algorithm:
	oldM float64
	newM float64
	oldS float64
	newS float64
}

func (s *Statistic) Push(val float64) {
	s.last = val
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

func (s *Statistic) Last() float64 {
	return s.last
}

// StandardError returns the standard error of the statistic.
func (s *Statistic) StandardError() float64 {
	return math.Sqrt(s.Variance() / float64(s.totalIterations))
}

func (s *Statistic) Iterations() int {
	return s.totalIterations
}
