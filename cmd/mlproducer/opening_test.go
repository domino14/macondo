package main

import (
	"fmt"
	"strings"
	"testing"

	"github.com/cespare/xxhash"
)

const baseHeader = "playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore"

func scanAll(t *testing.T, text string) []Turn {
	t.Helper()
	sc := NewTurnScanner(strings.NewReader(text))
	var out []Turn
	for sc.Scan() {
		out = append(out, sc.Turn())
	}
	if sc.Err() != nil {
		t.Fatal(sc.Err())
	}
	return out
}

func TestTurnScannerOpeningColumn(t *testing.T) {
	row := "p1,g1,1,ADEEIOT, 8C IODATE,16,16,6,E,15.884,86,0"
	// Without the column: zero.
	turns := scanAll(t, baseHeader+"\n"+row+"\n")
	if len(turns) != 1 || turns[0].OpeningPlies != 0 || turns[0].TilesRemaining != 86 {
		t.Fatalf("plain log: %+v", turns)
	}
	// With it: parsed on every row.
	turns = scanAll(t, baseHeader+",openingplies\n"+row+",3\n"+row+",0\n")
	if len(turns) != 2 || turns[0].OpeningPlies != 3 || turns[1].OpeningPlies != 0 {
		t.Fatalf("opening log: %+v", turns)
	}
	// After the inference columns, with rows of differing width (the
	// non-inferring bot's rows carry empty inference fields).
	hdr := baseHeader + ",inferCount,trueLeave,truePost,truePrior,liftBits,trueRank,inferLeaves,trueMeasured,openingplies"
	turns = scanAll(t, hdr+"\n"+row+",,,,,,,,,2\n"+row+",5,AB,0.1,0.05,1,3,40,40,2\n")
	if len(turns) != 2 || turns[0].OpeningPlies != 2 || turns[1].OpeningPlies != 2 {
		t.Fatalf("inference+opening log: %+v", turns)
	}
	// A short row is an error, not a silent skip.
	sc := NewTurnScanner(strings.NewReader(baseHeader + "\np1,g1,1\n"))
	if sc.Scan() || sc.Err() == nil {
		t.Fatalf("short row accepted")
	}
}

func TestPerGamePickStartsAtOpening(t *testing.T) {
	ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
	ga.pickMax = 30
	for _, k := range []int{0, 1, 3, 8, 30, 31} {
		lo := max(1, k)
		seen := map[int]bool{}
		for i := 0; i < 300; i++ {
			gw := ga.newGameWindow(Turn{PlayerID: "p1", GameID: "g", OpeningPlies: k})
			if k > ga.pickMax {
				if gw.pick != ga.pickMax+1 {
					t.Fatalf("K=%d: pick %d, want %d (never emits)", k, gw.pick, ga.pickMax+1)
				}
				continue
			}
			if gw.pick < lo || gw.pick > ga.pickMax {
				t.Fatalf("K=%d: pick %d outside [%d, %d]", k, gw.pick, lo, ga.pickMax)
			}
			seen[gw.pick] = true
		}
		if k <= ga.pickMax && !seen[lo] {
			t.Errorf("K=%d: the first eligible turn %d was never drawn in 300 tries", k, lo)
		}
	}
}

func TestPerGameSeveralPicks(t *testing.T) {
	// Two picks per game: two positions at distinct turns, both with the
	// game's result label; an empty-bag draw still emits nothing.
	for trial := 0; trial < 20; trial++ {
		ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
		ga.valueFromResult = true
		ga.pickMax = 22 // the sample game's last non-empty-bag turn
		ga.picks = 2
		got := feedGame(t, ga)
		if len(got) != 2 {
			t.Fatalf("trial %d: emitted %d vectors, want 2", trial, len(got))
		}
		if got[0].turn == got[1].turn {
			t.Fatalf("trial %d: both picks at turn %d", trial, got[0].turn)
		}
		for _, v := range got {
			if v.predictions[TargetWDL] == 0 || v.predictions[TargetValue] != v.predictions[TargetWDL] {
				t.Fatalf("trial %d turn %d: labels %v", trial, v.turn, v.predictions[:3])
			}
		}
	}
	// More picks than eligible turns: every turn in the range, once.
	ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
	ga.valueFromResult = true
	ga.pickMax = 5
	ga.picks = 9
	if got := feedGame(t, ga); len(got) != 5 {
		t.Fatalf("emitted %d vectors, want 5 (turns 1..5)", len(got))
	}
	if d := drawTurns(3, 30, 2); len(d) != 2 || d[0] >= d[1] || d[0] < 3 || d[1] > 30 {
		t.Fatalf("drawTurns: %v", d)
	}
}

func TestHoldoutSplitPartitionsGames(t *testing.T) {
	// Every game is on exactly one side, about 1/mod of them on the val side,
	// and no split means every game is emitted.
	held := 0
	for i := 0; i < 20000; i++ {
		h := xxhash.Sum64String(fmt.Sprintf("seed:game-%d", i))
		tr, va := inSplit(h, 20, "train"), inSplit(h, 20, "val")
		if tr == va {
			t.Fatalf("game %d: train=%v val=%v", i, tr, va)
		}
		if va {
			held++
		}
		if !inSplit(h, 0, "train") || !inSplit(h, 0, "val") {
			t.Fatalf("game %d: dropped without a split", i)
		}
	}
	if held < 800 || held > 1200 {
		t.Fatalf("held out %d of 20000 with mod 20, want ~1000", held)
	}
}

func TestPicksDifferAcrossScans(t *testing.T) {
	// A second scan of the same game draws its own turn, so streamed passes
	// see different positions of a game (20 scans, one draw each: not all equal).
	seen := map[int]bool{}
	for i := 0; i < 20; i++ {
		ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
		ga.valueFromResult = true
		ga.pickMax = 22
		for _, v := range feedGame(t, ga) {
			seen[v.turn] = true
		}
	}
	if len(seen) < 3 {
		t.Fatalf("20 scans drew only turns %v", seen)
	}
}

func TestTurnScannerSkipsRepeatedHeaders(t *testing.T) {
	// Concatenated logs carry a header row per file; each is skipped.
	row := "p1,g1,1,ADEEIOT, 8C IODATE,16,16,6,E,15.884,86,0"
	turns := scanAll(t, baseHeader+"\n"+row+"\n"+baseHeader+"\n"+row+"\n"+baseHeader+",openingplies\n"+row+",0\n")
	if len(turns) != 3 {
		t.Fatalf("got %d turns, want 3: %+v", len(turns), turns)
	}
}
