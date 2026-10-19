package main

import (
	"strings"
	"testing"

	"github.com/domino14/macondo/game"
)

// One complete logged game (autoplay-softmax-v-hasty-5.txt), 24 turns.
var sampleGameTurns = []string{
	"p1,685417a6010f675458000001,1,ADEEIOT, 8C IODATE,16,16,6,E,15.884,86,0",
	"p2,685417a6010f675458000001,2,EIIMOSU, E6 OI.IUM,18,18,5,ES,29.190,80,16",
	"p1,685417a6010f675458000001,3,?EEIINO,D11 OI,8,24,2,?EEIN,37.981,75,18",
	"p2,685417a6010f675458000001,4,DELPRSZ, C7 Z.P,27,45,2,DELRS,43.085,73,24",
	"p1,685417a6010f675458000001,5,?AEEINW, 7G INWEAvE,70,94,7,,70.000,71,45",
	"p2,685417a6010f675458000001,6,ADEELRS, N1 LASERED,81,126,7,,81.000,64,94",
	"p1,685417a6010f675458000001,7,CGHLLNS, 6D L.CH,21,115,3,GLNS,25.490,57,126",
	"p2,685417a6010f675458000001,8,ACDENOT, 3G TACNODE.,80,206,7,,80.000,54,115",
	"p1,685417a6010f675458000001,9,GLNNOSW, H1 GN.WN,39,154,4,LOS,46.722,47,206",
	"p2,685417a6010f675458000001,10,DEKMNTY, 1L YE.K,45,251,3,DMNT,37.324,43,154",
	"p1,685417a6010f675458000001,11,ILOOQRS,C12 QI,33,187,2,LOORS,36.510,40,251",
	"p2,685417a6010f675458000001,12,ADEMNST, K6 M.NDATES,72,323,7,,72.000,38,187",
	"p1,685417a6010f675458000001,13,GLOORRS,L10 GOR,17,204,3,LORS,25.232,31,323",
	"p2,685417a6010f675458000001,14,AEEEIPU,M10 EPEE,24,347,4,AIU,12.015,28,204",
	"p1,685417a6010f675458000001,15,LORSSTV,10H LOV...S,21,225,4,RST,31.935,24,347",
	"p2,685417a6010f675458000001,16,AINRTTU, 4J URAT.,22,369,4,INT,24.359,20,225",
	"p1,685417a6010f675458000001,17,BBGLRST, 6M B.G,10,235,2,BLRST,11.210,16,369",
	"p2,685417a6010f675458000001,18,AFIIJNT, 2F JI.,28,397,2,AFINT,30.821,14,235",
	"p1,685417a6010f675458000001,19,BELRSTV, I9 V.LT,11,246,3,BERS,26.111,12,397",
	"p2,685417a6010f675458000001,20,AFINRTU,N13 FUR,17,414,3,AINT,22.921,9,246",
	"p1,685417a6010f675458000001,21,?BEORSX,15J BOXE.S,69,315,5,?R,98.778,6,414",
	"p2,685417a6010f675458000001,22,AFHINRT,F10 HAFT,36,450,4,INR,36.944,1,315",
	"p1,685417a6010f675458000001,23,?AIORUY,H12 AIRY,27,342,4,?OU,13.000,0,450",
	"p2,685417a6010f675458000001,24,AINR,15D RAIN.,9,463,4,,13.000,0,342",
}

func feedGame(t *testing.T, ga *GameAssembler) []outputVector {
	t.Helper()
	hdr := "playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n"
	sc := NewTurnScanner(strings.NewReader(hdr + strings.Join(sampleGameTurns, "\n") + "\n"))
	var out []outputVector
	for sc.Scan() {
		out = append(out, ga.FeedTurn(sc.Turn())...)
	}
	if sc.Err() != nil {
		t.Fatal(sc.Err())
	}
	return out
}

// -labeler result -per-game: one position per game, its value target is
// the mover's real result and its spread target the spread change to the
// end of the game; the features are the table-mode features for that turn.
func TestResultLabelerOnePositionPerGame(t *testing.T) {
	table := NewGameAssembler(NPlies)
	want := feedGame(t, table)
	byTurn := map[int]outputVector{}
	for _, v := range want {
		byTurn[v.turn] = v
	}
	// p2 won the sample game 463-342 (459 + 4 for p1's unplayed ?OU), so
	// the final spread is +121 for p2 and -121 for p1.
	const finalP2 = 121.0

	seen := 0
	for trial := 0; trial < 40; trial++ {
		ga := NewGameAssembler(NPlies)
		ga.valueFromResult = true
		ga.pickMax = 30
		got := feedGame(t, ga)
		if len(got) > 1 {
			t.Fatalf("trial %d: %d vectors from one game, want at most 1", trial, len(got))
		}
		if len(got) == 0 {
			continue // the draw was past the end of the game
		}
		seen++
		v := got[0]
		ref, ok := byTurn[v.turn]
		if !ok {
			t.Fatalf("trial %d: emitted turn %d, which table mode never emits", trial, v.turn)
		}
		for j := range *ref.features {
			if (*ref.features)[j] != (*v.features)[j] {
				t.Fatalf("trial %d turn %d: feature %d differs from table mode", trial, v.turn, j)
			}
		}
		final := float32(finalP2)
		if v.mover == 0 {
			final = -finalP2
		}
		wdl := float32(1)
		if final < 0 {
			wdl = -1
		}
		if v.predictions[TargetValue] != wdl || v.predictions[TargetWDL] != wdl {
			t.Fatalf("trial %d turn %d: value %v wdl %v, want %v", trial, v.turn, v.predictions[TargetValue], v.predictions[TargetWDL], wdl)
		}
		wantSpread := game.NormalizeSpreadForML(final - v.spreadNow)
		if v.predictions[TargetSpread] != wantSpread {
			t.Fatalf("trial %d turn %d: spread %v, want %v (final %v, now %v)", trial, v.turn, v.predictions[TargetSpread], wantSpread, final, v.spreadNow)
		}
		for _, k := range []int{TargetOppBingo, TargetOppScore} {
			if v.predictions[k] != ref.predictions[k] {
				t.Fatalf("trial %d turn %d: target %d differs from table mode", trial, v.turn, k)
			}
		}
	}
	if seen == 0 {
		t.Fatal("no trial emitted a position")
	}
}

// In per-game mode a drawn endgame turn emits nothing; the net is never
// consulted with an empty bag.
func TestPerGameSkipsEndgamePositions(t *testing.T) {
	// Every turn of the sample game with the bag already empty must be skipped,
	// and every other turn emitted, when drawn.
	for turn := 20; turn <= 24; turn++ {
		ga := NewGameAssembler(NPlies)
		ga.valueFromResult = true
		ga.pickMax = 30
		// Draw exactly `turn`: pickMax marks per-game mode, fixedPick is
		// overridden below to keep the skip active.
		ga.fixedPick = 0
		ga.games = map[string]*gameWindow{}
		got := feedGameWithPick(t, ga, turn)
		emptyBag := turn >= 23 // sample game: bag empty before turns 23 and 24
		if emptyBag && len(got) != 0 {
			t.Fatalf("turn %d: emitted %d vectors from an endgame draw", turn, len(got))
		}
		if !emptyBag && len(got) != 1 {
			t.Fatalf("turn %d: emitted %d vectors, want 1", turn, len(got))
		}
	}
}

// feedGameWithPick feeds the sample game with the per-game draw forced to
// `turn` without using the fixedPick test hook (which bypasses the skip).
func feedGameWithPick(t *testing.T, ga *GameAssembler, turn int) []outputVector {
	t.Helper()
	hdr := "playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n"
	sc := NewTurnScanner(strings.NewReader(hdr + strings.Join(sampleGameTurns, "\n") + "\n"))
	var out []outputVector
	first := true
	for sc.Scan() {
		tn := sc.Turn()
		if first {
			// Create the window with a throwaway draw of turn 1, discard
			// that position's state, then set the real draw.
			ga.pickMax = 1
			out = append(out, ga.FeedTurn(tn)...)
			gw := ga.games[tn.GameID]
			gw.pick, gw.picks = turn, []int{turn}
			game.MLVectorPool.Put(gw.plies[0].state)
			gw.plies[0].state = nil
			ga.pickMax = 30
			first = false
			continue
		}
		out = append(out, ga.FeedTurn(tn)...)
	}
	return out
}
