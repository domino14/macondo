package preendgame

import (
	"math/rand"
	"testing"
)

// The final order is by points; among plays level on points, those with a
// tiebreak spread come first, by spread; and plays that tie on all of that
// keep the order they came in. The sort this replaced compared spreads
// only, so every play outside the tiebreak -- a winning pass, and all the
// plays below the winners -- came out in whatever order the sort left it:
// a play that wins 4 draws above one that wins 7, differently on each run.
func TestSortPlaysAfterTiebreak(t *testing.T) {
	type spec struct {
		points    float32
		spreadSet bool
		spread    int
	}
	// In the order they should come out.
	want := []spec{
		{8, true, 120}, {8, true, 95}, {8, true, 95}, {8, true, -3},
		{8, false, 0}, {8, false, 0}, // winners outside the tiebreak
		{7, false, 0}, {7, false, 0}, {7, false, 0},
		{4, false, 0}, {4, false, 0},
		{0.5, false, 0}, {0, false, 0}, {0, false, 0},
	}
	r := rand.New(rand.NewSource(7))
	for trial := 0; trial < 200; trial++ {
		plays := make([]*PreEndgamePlay, len(want))
		for i, w := range want {
			plays[i] = &PreEndgamePlay{Points: w.points, spreadSet: w.spreadSet, Spread: w.spread}
		}
		r.Shuffle(len(plays), func(i, j int) { plays[i], plays[j] = plays[j], plays[i] })
		before := append([]*PreEndgamePlay(nil), plays...)
		sortPlaysAfterTiebreak(plays)
		for i, w := range want {
			p := plays[i]
			if p.Points != w.points || p.spreadSet != w.spreadSet || p.Spread != w.spread {
				t.Fatalf("trial %d: place %d is {%v %v %v}, want %+v", trial, i, p.Points, p.spreadSet, p.Spread, w)
			}
		}
		// Plays that tie keep their order.
		pos := map[*PreEndgamePlay]int{}
		for i, p := range before {
			pos[p] = i
		}
		for i := 1; i < len(plays); i++ {
			a, b := plays[i-1], plays[i]
			if a.Points == b.Points && a.spreadSet == b.spreadSet && a.Spread == b.Spread && pos[a] > pos[b] {
				t.Fatalf("trial %d: tied plays at %d and %d changed order", trial, i-1, i)
			}
		}
	}
}
