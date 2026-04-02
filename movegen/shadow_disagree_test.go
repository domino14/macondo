package movegen_test

import (
	"fmt"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
)

func replayToTurn(t *testing.T, gidx, targetTurn int) (*game.Game, *kwg.KWG) {
	t.Helper()
	is := is.New(t)

	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL23",
		board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	playerInfos := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "Player1"},
		{Nickname: "p2", RealName: "Player2"},
	}

	g, err := game.NewGame(rules, playerInfos)
	is.NoErr(err)

	seed := [32]byte{}
	seed[0] = byte(gidx)
	seed[1] = byte(gidx >> 8)
	seed[2] = byte(gidx >> 16)
	g.SeedBag(seed)
	g.StartGame()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL23")
	is.NoErr(err)

	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())

	for turn := 0; turn < targetTurn; turn++ {
		if g.Playing() != pb.PlayState_PLAYING {
			t.Fatalf("game ended at turn %d, needed %d", turn, targetTurn)
		}
		rack := g.RackFor(g.PlayerOnTurn())
		gen.SetPlayRecorderTopPlay()
		plays := gen.GenAll(rack, false)
		g.PlayMove(plays[0], false, 0)
	}

	return g, gd
}

// TestShadowDisagree137741 reproduces game 137741 turn 25 where shadow found
// score 10 but the actual best was 11.
func TestShadowDisagree137741(t *testing.T) {
	g, gd := replayToTurn(t, 137741, 25)

	rack := g.RackFor(g.PlayerOnTurn())
	fmt.Printf("Game 137741 turn 25: rack=%s\n", rack.String())
	fmt.Printf("Board: %s\n", g.Board().ToFEN(gd.GetAlphabet()))

	ld := g.Bag().LetterDistribution()

	// No shadow - find all moves
	genNS := movegen.NewGordonGenerator(gd, g.Board(), ld)
	
	genNS.SetPlayRecorder(movegen.AllPlaysRecorder)
	playsNS := genNS.GenAll(rack, false)
	topNS := bestByScore(playsNS)
	fmt.Printf("No-shadow top: %s (%d) [%d moves]\n", topNS.ShortDescription(), topNS.Score(), len(playsNS))

	// Shadow - TopPlayOnly
	genS := movegen.NewGordonGenerator(gd, g.Board(), ld)
	genS.SetPlayRecorderTopPlay()
	playsS := genS.GenAll(rack, false)
	fmt.Printf("Shadow top: %s (%d)\n", playsS[0].ShortDescription(), playsS[0].Score())

	// Run shadow only to examine anchors
	genS2 := movegen.NewGordonGenerator(gd, g.Board(), ld)
	anchors := genS2.RunShadowOnly(rack)
	fmt.Printf("Shadow anchors (%d):\n", len(anchors))
	for i, a := range anchors {
		dir := "H"
		if a.Dir == board.VerticalDirection {
			dir = "V"
		}
		fmt.Printf("  [%d] (%d,%d) %s score=%d\n", i, a.Row, a.Col, dir, a.HighestPossibleScore)
	}

	// Find which anchor the winning move belongs to
	fmt.Printf("\nExpected winning move: %s (%d)\n", topNS.ShortDescription(), topNS.Score())

	if topNS.Score() != playsS[0].Score() {
		t.Errorf("DISAGREEMENT: noshadow=%d shadow=%d", topNS.Score(), playsS[0].Score())
	}
}

// TestShadowDisagree550396 reproduces game 550396 turn 12 where shadow found
// score 26 but the actual best was 30 (4-point difference).
func TestShadowDisagree550396(t *testing.T) {
	g, gd := replayToTurn(t, 550396, 12)

	rack := g.RackFor(g.PlayerOnTurn())
	fmt.Printf("Game 550396 turn 12: rack=%s\n", rack.String())
	fmt.Printf("Board: %s\n", g.Board().ToFEN(gd.GetAlphabet()))

	ld := g.Bag().LetterDistribution()

	// No shadow - find all moves
	genNS := movegen.NewGordonGenerator(gd, g.Board(), ld)
	
	genNS.SetPlayRecorder(movegen.AllPlaysRecorder)
	playsNS := genNS.GenAll(rack, false)
	topNS := bestByScore(playsNS)
	fmt.Printf("No-shadow top: %s (%d) [%d moves]\n", topNS.ShortDescription(), topNS.Score(), len(playsNS))

	// Shadow - TopPlayOnly
	genS := movegen.NewGordonGenerator(gd, g.Board(), ld)
	genS.SetPlayRecorderTopPlay()
	playsS := genS.GenAll(rack, false)
	fmt.Printf("Shadow top: %s (%d)\n", playsS[0].ShortDescription(), playsS[0].Score())

	// Run shadow only to examine anchors
	genS2 := movegen.NewGordonGenerator(gd, g.Board(), ld)
	anchors := genS2.RunShadowOnly(rack)
	fmt.Printf("Shadow anchors (%d):\n", len(anchors))
	for i, a := range anchors {
		dir := "H"
		if a.Dir == board.VerticalDirection {
			dir = "V"
		}
		fmt.Printf("  [%d] (%d,%d) %s score=%d\n", i, a.Row, a.Col, dir, a.HighestPossibleScore)
	}

	fmt.Printf("\nExpected winning move: %s (%d)\n", topNS.ShortDescription(), topNS.Score())

	if topNS.Score() != playsS[0].Score() {
		t.Errorf("DISAGREEMENT: noshadow=%d shadow=%d", topNS.Score(), playsS[0].Score())
	}
}

// TestShadowDisagree3064 reproduces game 3064 turn 19 where shadow found
// score 27 but the actual best was 28 (1-point difference).
func TestShadowDisagree3064(t *testing.T) {
	g, gd := replayToTurn(t, 3064, 19)

	rack := g.RackFor(g.PlayerOnTurn())
	fmt.Printf("Game 3064 turn 19: rack=%s\n", rack.String())
	fmt.Printf("Board: %s\n", g.Board().ToFEN(gd.GetAlphabet()))

	ld := g.Bag().LetterDistribution()

	// No shadow - find all moves
	genNS := movegen.NewGordonGenerator(gd, g.Board(), ld)
	
	genNS.SetPlayRecorder(movegen.AllPlaysRecorder)
	playsNS := genNS.GenAll(rack, false)
	topNS := bestByScore(playsNS)
	fmt.Printf("No-shadow top: %s (%d) [%d moves]\n", topNS.ShortDescription(), topNS.Score(), len(playsNS))

	// Shadow - TopPlayOnly
	genS := movegen.NewGordonGenerator(gd, g.Board(), ld)
	genS.SetPlayRecorderTopPlay()
	playsS := genS.GenAll(rack, false)
	fmt.Printf("Shadow top: %s (%d)\n", playsS[0].ShortDescription(), playsS[0].Score())

	// Run shadow only to examine anchors
	genS2 := movegen.NewGordonGenerator(gd, g.Board(), ld)
	anchors := genS2.RunShadowOnly(rack)
	fmt.Printf("Shadow anchors (%d):\n", len(anchors))
	for i, a := range anchors {
		dir := "H"
		if a.Dir == board.VerticalDirection {
			dir = "V"
		}
		fmt.Printf("  [%d] (%d,%d) %s score=%d\n", i, a.Row, a.Col, dir, a.HighestPossibleScore)
	}

	fmt.Printf("\nExpected winning move: %s (%d)\n", topNS.ShortDescription(), topNS.Score())

	if topNS.Score() != playsS[0].Score() {
		t.Errorf("DISAGREEMENT: noshadow=%d shadow=%d", topNS.Score(), playsS[0].Score())
	}
}
