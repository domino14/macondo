package main

import (
	"fmt"
	"strings"
	"testing"

	"github.com/domino14/word-golib/kwg"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
)

// -endgame-plies: a position whose bag was already empty is labeled by a
// quick search from the opponent's reply. In the sample game p1's turn 23
// (AIRY, down 108) leaves p2 to play out RAIN for 9 plus 4 for p1's ?OU,
// so the search must find p2 winning by 13: value -1, spread -13 for p1.
// Turn 22 empties the bag with the move itself, so the mover's new tiles
// are not known and the position keeps its logged-result label. The
// replay must survive the search: features identical to table mode.
func TestEndgameSearchLabels(t *testing.T) {
	table := NewGameAssembler(NPlies)
	want := feedGame(t, table)
	byTurn := map[int]outputVector{}
	for _, v := range want {
		byTurn[v.turn] = v
	}
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL23")
	if err != nil {
		t.Fatal(err)
	}
	negamax.GlobalTranspositionTable.Reset(0.01, 15)

	for _, tc := range []struct {
		turn   int
		solved bool
		spread float32 // expected mover's spread change
	}{{23, true, -13}, {22, false, 121 - 135}} {
		ga := NewGameAssembler(NPlies)
		ga.valueFromResult = true
		ga.fixedPick = tc.turn
		ga.endgamePlies = 2
		ga.kwg = gd
		got := feedGame(t, ga)
		if len(got) != 1 || got[0].turn != tc.turn {
			t.Fatalf("turn %d: got %d vectors", tc.turn, len(got))
		}
		v := got[0]
		ref := byTurn[tc.turn]
		for j := range *ref.features {
			if (*ref.features)[j] != (*v.features)[j] {
				t.Fatalf("turn %d: feature %d differs from table mode", tc.turn, j)
			}
		}
		if (v.solved != nil) != tc.solved || int(ga.solved) != map[bool]int{true: 1, false: 0}[tc.solved] {
			t.Fatalf("turn %d: solved=%v (%d solved), want %v", tc.turn, v.solved != nil, ga.solved, tc.solved)
		}
		wantValue := float32(-1) // p1 lost the game either way
		if tc.turn == 22 {
			wantValue = 1 // p2's position: p2 won
		}
		if v.predictions[TargetValue] != wantValue || v.predictions[TargetWDL] != wantValue {
			t.Fatalf("turn %d: value %v wdl %v, want %v", tc.turn, v.predictions[TargetValue], v.predictions[TargetWDL], wantValue)
		}
		if tc.solved && v.solved.spread != tc.spread {
			t.Fatalf("turn %d: solved spread %v, want %v", tc.turn, v.solved.spread, tc.spread)
		}
		if v.predictions[TargetSpread] != game.NormalizeSpreadForML(tc.spread) {
			t.Fatalf("turn %d: spread target %v, want %v", tc.turn, v.predictions[TargetSpread], game.NormalizeSpreadForML(tc.spread))
		}
	}
}

// The search returns the opponent's spread change, not the result: a mover
// who is ahead and gives back a few points still wins.
func TestEndgameLabelArithmetic(t *testing.T) {
	for _, tc := range []struct {
		now    float32
		change int16 // opponent's gain
		value  float32
		spread float32
	}{
		{-108, 13, -1, -13}, // behind, loses more: loss
		{50, 13, 1, -13},    // ahead, gives back 13: still a win
		{10, 10, 0, -10},    // exactly tied at the end: draw
		{-5, -20, 1, 20},    // behind, but the opponent is stuck: win
	} {
		got := endgameLabel(tc.now, tc.change)
		if got.value != tc.value || got.spread != tc.spread {
			t.Fatalf("now %v change %v: got %+v, want value %v spread %v", tc.now, tc.change, got, tc.value, tc.spread)
		}
	}
}

// A second logged game with a five-position endgame (bag empty before
// turns 19-23). From the log: after turn 22 p2 leads 466-378 holding LN and
// p1 goes out with R for 6 plus 4 (2x LN), so p2's change is -10; after
// turn 21 p1 trails 378-456 holding R, p2 (LNT) plays T for 10, then p1
// goes out for 6+4, so p1's change is 0.
var endgameGameTurns = []string{
	"p1,685417a6010f67545800000f,1,AADLUVW, 8G VAW,18,18,3,ADLU,14.113,86,0",
	"p2,685417a6010f67545800000f,2,NNOSTVY, H7 N.VY,10,10,3,NOST,20.645,83,18",
	"p1,685417a6010f67545800000f,3,?ADLOPU, J2 UPLOADs,72,90,7,,72.000,80,10",
	"p2,685417a6010f67545800000f,4,AEJNOST,10F JO.,29,39,2,AENST,52.473,73,90",
	"p1,685417a6010f67545800000f,5,AILOSUU, 3J .OILU,14,104,4,ASU,16.279,71,39",
	"p2,685417a6010f67545800000f,6,AAENRST, O3 SANTERA,82,121,7,,82.000,67,104",
	"p1,685417a6010f67545800000f,7,AAEIMSU, N9 AMUSIA,30,134,6,E,30.584,60,121",
	"p2,685417a6010f67545800000f,8,BEEEEGN, L1 BE.GNE,24,145,5,EE,19.820,54,134",
	"p1,685417a6010f67545800000f,9,?AAEELS,12H ELAStA.E,68,202,7,,68.000,49,145",
	"p2,685417a6010f67545800000f,10,ACEEKMR, 4C MACKERE.,84,229,7,,84.000,42,202",
	"p1,685417a6010f67545800000f,11,CDIOOTY, C2 CO.ITY,34,236,5,DO,33.049,35,229",
	"p2,685417a6010f67545800000f,12,DIINOSX,15H DIOXINS,107,336,7,,107.000,30,236",
	"p1,685417a6010f67545800000f,13,DEEFOOU,11E OOF,26,262,3,DEEU,18.918,23,336",
	"p2,685417a6010f67545800000f,14,EGHIIOT,12A HOGTIE,40,376,6,I,37.708,20,262",
	"p1,685417a6010f67545800000f,15,DEEENPU,14J PE,29,291,2,DEENU,26.272,14,376",
	"p2,685417a6010f67545800000f,16,BHIIRRT, A8 BIRT.,30,406,4,HIR,30.478,12,291",
	"p1,685417a6010f67545800000f,17,DEEFNQU, H1 QUE.N,42,333,4,DEF,44.052,8,406",
	"p2,685417a6010f67545800000f,18,GHILRTW, C9 WRI.HT,34,440,5,GL,29.985,4,333",
	"p1,685417a6010f67545800000f,19,DEEFIRR, 1L .EEF,27,360,3,DIRR,7.000,0,440",
	"p2,685417a6010f67545800000f,20,DGLNTZ, 6I G.Z.D,16,456,3,LNT,0.000,0,360",
	"p1,685417a6010f67545800000f,21,DIRR,13L IR.D,18,378,3,R,6.000,0,456",
	"p2,685417a6010f67545800000f,22,LNT, L1 ......T,10,466,1,LN,-4.000,0,378",
	"p1,685417a6010f67545800000f,23,R, 3G R.,6,388,1,,10.000,0,466",
}

func feedTurns(t *testing.T, ga *GameAssembler, rows []string) []outputVector {
	t.Helper()
	hdr := "playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n"
	sc := NewTurnScanner(strings.NewReader(hdr + strings.Join(rows, "\n") + "\n"))
	var out []outputVector
	for sc.Scan() {
		out = append(out, ga.FeedTurn(sc.Turn())...)
	}
	if sc.Err() != nil {
		t.Fatal(sc.Err())
	}
	return out
}

// After an endgame label the producer's game must be exactly what table
// mode has at the same point, since the rest of the replay (and any other
// labeler) continues from it.
func TestEndgameSearchRestoresState(t *testing.T) {
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL23")
	if err != nil {
		t.Fatal(err)
	}
	negamax.GlobalTranspositionTable.Reset(0.01, 15)
	gid := "685417a6010f67545800000f"

	snapshot := func(g *game.Game) string {
		return fmt.Sprintf("turn=%d playing=%v racks=%q/%q bag=%d spread=%d scoreless=%d",
			g.PlayerOnTurn(), g.Playing(), g.RackLettersFor(0), g.RackLettersFor(1),
			g.Bag().TilesRemaining(), g.SpreadFor(0), g.ScorelessTurns())
	}

	for pick := 19; pick <= 22; pick++ {
		table := NewGameAssembler(NPlies)
		solved := NewGameAssembler(NPlies)
		solved.valueFromResult = true
		solved.fixedPick = pick
		solved.endgamePlies = 2
		solved.kwg = gd

		var want, got []outputVector
		hdr := "playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n"
		scT := NewTurnScanner(strings.NewReader(hdr + strings.Join(endgameGameTurns, "\n") + "\n"))
		scS := NewTurnScanner(strings.NewReader(hdr + strings.Join(endgameGameTurns, "\n") + "\n"))
		for scT.Scan() && scS.Scan() {
			want = append(want, table.FeedTurn(scT.Turn())...)
			got = append(got, solved.FeedTurn(scS.Turn())...)
			if scT.Turn().TurnNumber == pick {
				if solved.solved != 1 {
					t.Fatalf("pick %d: solved %d positions after the pick, want 1", pick, solved.solved)
				}
				a, b := snapshot(table.games[gid].game.Game), snapshot(solved.games[gid].game.Game)
				if a != b {
					t.Fatalf("pick %d: game state differs after the endgame label\n table:  %s\n solved: %s", pick, a, b)
				}
			}
		}
		if len(got) != 1 || got[0].turn != pick {
			t.Fatalf("pick %d: got %d vectors", pick, len(got))
		}
		if solved.games[gid] != nil || table.games[gid] != nil {
			t.Fatalf("pick %d: game did not finish", pick)
		}
		var ref outputVector
		for _, v := range want {
			if v.turn == pick {
				ref = v
			}
		}
		v := got[0]
		for j := range *ref.features {
			if (*ref.features)[j] != (*v.features)[j] {
				t.Fatalf("pick %d: feature %d differs from table mode", pick, j)
			}
		}
		if v.predictions[TargetWDL] != ref.predictions[TargetWDL] {
			t.Fatalf("pick %d: wdl %v, table mode %v", pick, v.predictions[TargetWDL], ref.predictions[TargetWDL])
		}
		switch pick {
		case 22:
			if v.solved.spread != -10 || v.predictions[TargetValue] != 1 {
				t.Fatalf("turn 22: solved %+v value %v, want spread -10 value +1", v.solved, v.predictions[TargetValue])
			}
		case 21:
			if v.solved.spread != 0 || v.predictions[TargetValue] != -1 {
				t.Fatalf("turn 21: solved %+v value %v, want spread 0 value -1", v.solved, v.predictions[TargetValue])
			}
		default:
			// Earlier in the endgame the search's own answer is not an oracle;
			// the result must at least agree with how the game went, since
			// p2 is far ahead throughout.
			if v.predictions[TargetValue] != ref.predictions[TargetWDL] {
				t.Fatalf("turn %d: value %v, want %v", pick, v.predictions[TargetValue], ref.predictions[TargetWDL])
			}
		}
	}
}

// A search that exceeds the timeout is abandoned and the position keeps
// its logged label.
func TestEndgameSearchTimeoutFallsBack(t *testing.T) {
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL23")
	if err != nil {
		t.Fatal(err)
	}
	negamax.GlobalTranspositionTable.Reset(0.01, 15)
	ga := NewGameAssembler(NPlies)
	ga.valueFromResult = true
	ga.fixedPick = 23
	ga.endgamePlies = 2
	ga.endgameTimeout = 1 // nanosecond: every search times out
	ga.kwg = gd
	got := feedGame(t, ga)
	if len(got) != 1 || got[0].solved != nil || ga.solved != 0 || ga.endgameTimeouts != 1 {
		t.Fatalf("got %d vectors, solved=%v, solved count %d, timeouts %d", len(got), got[0].solved != nil, ga.solved, ga.endgameTimeouts)
	}
	if got[0].predictions[TargetValue] != -1 || got[0].predictions[TargetSpread] != game.NormalizeSpreadForML(-13) {
		t.Fatalf("logged label expected: value %v spread %v", got[0].predictions[TargetValue], got[0].predictions[TargetSpread])
	}
}
