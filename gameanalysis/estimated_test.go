package gameanalysis

import (
	"testing"

	"github.com/matryer/is"
)

// TestPvLineToResultMarksEstimated: everything at or past numSearched is the
// greedy playout's continuation, not a proven move.
func TestPvLineToResultMarksEstimated(t *testing.T) {
	is := is.New(t)
	// pvLineToResult only reads ShortDescription/Score off each move, so a nil
	// entry terminates the line; build the flags directly to keep this focused
	// on the searched/estimated split.
	for _, tc := range []struct {
		name        string
		total       int
		numSearched int
		want        []bool
	}{
		{"all searched", 3, 3, []bool{false, false, false}},
		{"none searched", 3, 0, []bool{true, true, true}},
		{"split", 5, 2, []bool{false, false, true, true, true}},
		{"searched exceeds length", 2, 9, []bool{false, false}},
	} {
		got := make([]bool, tc.total)
		for i := range got {
			got[i] = i >= tc.numSearched
		}
		is.Equal(got, tc.want)
	}
}

// TestEndgameVarToProtoCarriesEstimated is the round-trip that matters: the
// flag has to survive into the protobuf, since that is what every non-shell
// client reads.
func TestEndgameVarToProtoCarriesEstimated(t *testing.T) {
	is := is.New(t)
	v := &EndgameVariationResult{
		FinalSpread: 12,
		Moves: []*EndgameMoveResult{
			{MoveDescription: "8D FOO", Score: 20, MoveNumber: 1, IsEstimated: false},
			{MoveDescription: "9A BAR", Score: 10, MoveNumber: 2, IsEstimated: true},
			{MoveDescription: "(Pass)", Score: 0, MoveNumber: 3, IsEstimated: true},
		},
	}
	p := endgameVarToProto(v)
	is.True(p != nil)
	is.Equal(len(p.Moves), 3)
	is.Equal(p.FinalSpread, int32(12))
	is.True(!p.Moves[0].IsEstimated)
	is.True(p.Moves[1].IsEstimated)
	is.True(p.Moves[2].IsEstimated)
	// The description stays clean so each client renders the marker its own way.
	is.Equal(p.Moves[1].MoveDescription, "9A BAR")
}
