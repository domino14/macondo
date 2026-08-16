package montecarlo

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
)

// The precomputed CDF must sample racks proportionally to their weights and
// never select zero-weight entries.
func TestWeightedInferredDrawRacksCDF(t *testing.T) {
	is := is.New(t)
	s := &Simmer{}
	racks := []InferredRack{
		{Leave: []tilemapping.MachineLetter{1}, Weight: 0},
		{Leave: []tilemapping.MachineLetter{2}, Weight: 1},
		{Leave: []tilemapping.MachineLetter{3}, Weight: 3},
	}
	s.SetInferences(racks, 1, InferenceCompletePosterior)
	is.Equal(len(s.inferenceCDF), 3)

	counts := map[tilemapping.MachineLetter]int{}
	const draws = 40000
	for i := 0; i < draws; i++ {
		leave, err := s.weightedInferredDrawRacks()
		is.NoErr(err)
		is.Equal(len(leave), 1)
		counts[leave[0]]++
	}

	is.Equal(counts[1], 0) // zero-weight entry never selected
	ratio := float64(counts[3]) / float64(counts[2])
	if ratio < 2.7 || ratio > 3.3 {
		t.Fatalf("weight-3 vs weight-1 draw ratio %v, want ≈3", ratio)
	}
}

// All-zero weights must not install a CDF; drawing errors out instead of
// looping or selecting arbitrarily.
func TestWeightedInferredDrawRacksNoMass(t *testing.T) {
	is := is.New(t)
	s := &Simmer{}
	s.SetInferences([]InferredRack{
		{Leave: []tilemapping.MachineLetter{1}, Weight: 0},
	}, 1, InferenceCompletePosterior)
	is.Equal(len(s.inferenceCDF), 0)
	_, err := s.weightedInferredDrawRacks()
	is.True(err != nil)
}
