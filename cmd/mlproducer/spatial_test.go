package main

import (
	"strings"
	"testing"

	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
)

// squaresOf is an independent oracle for the placement planes: the squares
// a logged play string covers, straight from its coordinates and tile
// string (a '.' is a played-through tile), transposed the way the producer
// replays this game. Exchanges and passes cover nothing.
func squaresOf(t *testing.T, row string, transpose bool) map[int]bool {
	t.Helper()
	fields := strings.Split(row, ",")
	play := stats.Normalize(fields[4])
	out := map[int]bool{}
	if play == "pass" || strings.HasPrefix(play, "exchange") {
		return out
	}
	parts := strings.Fields(play)
	if len(parts) != 2 {
		t.Fatalf("unexpected play %q", play)
	}
	r, c, vertical := move.FromBoardGameCoords(parts[0], transpose)
	for i, ch := range parts[1] {
		if ch == '.' {
			continue
		}
		if vertical {
			out[(r+i)*15+c] = true
		} else {
			out[r*15+c+i] = true
		}
	}
	return out
}

func planeSquares(plane []float32) map[int]bool {
	out := map[int]bool{}
	for i, v := range plane {
		if v != 0 {
			out[i] = true
		}
	}
	return out
}

func sameSquares(a, b map[int]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

// TestSpatialTargets checks, for several drawn turns of the sample game,
// that the opponent-next and self-next planes mark exactly the squares of
// the next two logged plays, and that the win planes copy them only for the
// player who won the game.
func TestSpatialTargets(t *testing.T) {
	gid := strings.Split(sampleGameTurns[0], ",")[1]
	transpose := shouldTranspose(gid)
	for _, turn := range []int{2, 3, 5, 12, 20, 22} { // the helper discards turn 1
		ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
		ga.valueFromResult = true
		ga.pickMax = 30
		got := feedGameWithPick(t, ga, turn)
		if len(got) != 1 {
			t.Fatalf("turn %d: emitted %d vectors, want 1", turn, len(got))
		}
		v := got[0]
		if len(v.predictions) != NumPredictions {
			t.Fatalf("turn %d: %d predictions, want %d", turn, len(v.predictions), NumPredictions)
		}
		wantOpp := squaresOf(t, sampleGameTurns[turn], transpose) // turn+1 (rows are 0-based)
		wantSelf := map[int]bool{}
		if turn+1 < len(sampleGameTurns) {
			wantSelf = squaresOf(t, sampleGameTurns[turn+1], transpose) // turn+2
		}
		if len(wantOpp) == 0 {
			t.Fatalf("turn %d: oracle found no squares for the reply", turn)
		}
		gotOpp := planeSquares(spatialPlane(v.predictions, SpatialOppNext))
		gotSelf := planeSquares(spatialPlane(v.predictions, SpatialSelfNext))
		if !sameSquares(gotOpp, wantOpp) {
			t.Errorf("turn %d: opp-next squares %v, want %v", turn, gotOpp, wantOpp)
		}
		if !sameSquares(gotSelf, wantSelf) {
			t.Errorf("turn %d: self-next squares %v, want %v", turn, gotSelf, wantSelf)
		}
		// Win conjunctions follow the final result from the mover's side.
		wdl := v.predictions[TargetWDL]
		gotOppWin := planeSquares(spatialPlane(v.predictions, SpatialOppWin))
		gotSelfWin := planeSquares(spatialPlane(v.predictions, SpatialSelfWin))
		switch {
		case wdl > 0:
			if !sameSquares(gotSelfWin, wantSelf) || len(gotOppWin) != 0 {
				t.Errorf("turn %d (mover won): self-win %v want %v, opp-win %v want none", turn, gotSelfWin, wantSelf, gotOppWin)
			}
		case wdl < 0:
			if !sameSquares(gotOppWin, wantOpp) || len(gotSelfWin) != 0 {
				t.Errorf("turn %d (mover lost): opp-win %v want %v, self-win %v want none", turn, gotOppWin, wantOpp, gotSelfWin)
			}
		default:
			if len(gotOppWin) != 0 || len(gotSelfWin) != 0 {
				t.Errorf("turn %d (draw): win planes must be empty", turn)
			}
		}
	}
}

// TestSpatialTargetsAtGameEnd: a position whose mover never moves again
// (the opponent's reply ends the game) has an empty self-next plane.
func TestSpatialTargetsAtGameEnd(t *testing.T) {
	gid := strings.Split(sampleGameTurns[0], ",")[1]
	transpose := shouldTranspose(gid)
	last := len(sampleGameTurns) // 24: turn 23's reply is the final move
	ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
	ga.valueFromResult = true
	ga.pickMax = 30
	ga.fixedPick = last - 1 // bypasses the empty-bag skip
	got := feedGame(t, ga)
	if len(got) != 1 {
		t.Fatalf("emitted %d vectors, want 1", len(got))
	}
	v := got[0]
	wantOpp := squaresOf(t, sampleGameTurns[last-1], transpose)
	if gotOpp := planeSquares(spatialPlane(v.predictions, SpatialOppNext)); !sameSquares(gotOpp, wantOpp) {
		t.Errorf("opp-next squares %v, want %v", gotOpp, wantOpp)
	}
	if n := len(planeSquares(spatialPlane(v.predictions, SpatialSelfNext))); n != 0 {
		t.Errorf("self-next plane has %d squares after the game ended, want 0", n)
	}
}
