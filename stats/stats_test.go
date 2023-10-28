package stats

import (
	"testing"

	"github.com/matryer/is"
)

func TestRunningStat(t *testing.T) {
	is := is.New(t)
	type tc struct {
		scores []int
		mean   float64
		stdev  float64
	}
	cases := []tc{
		{[]int{10, 12, 23, 23, 16, 23, 21, 16}, 18, 5.2372293656638},
		{[]int{14, 35, 71, 124, 10, 24, 55, 33, 87, 19}, 47.2, 36.937785531891},
		{[]int{1}, 1, 0},
		{[]int{}, 0, 0},
		{[]int{1, 1}, 1, 0},
	}
	for _, c := range cases {
		s := &Statistic{}
		for _, score := range c.scores {
			s.Push(float64(score))
		}
		is.True(FuzzyEqual(s.Mean(), c.mean))
		is.True(FuzzyEqual(s.Stdev(), c.stdev))

	}
}
