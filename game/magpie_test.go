package game

import (
	"math/rand/v2"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tinymove"
)

// snapshot is everything about a game's position a move changes.
type snapshot struct {
	board  *board.GameBoard
	racks  [2]string
	points [2]int
	bag    int
	onturn int
}

func takeSnapshot(g *Game) snapshot {
	return snapshot{board: g.Board().Copy(), racks: [2]string{g.RackLettersFor(0), g.RackLettersFor(1)},
		points: [2]int{g.PointsFor(0), g.PointsFor(1)}, bag: g.Bag().TilesRemaining(), onturn: g.PlayerOnTurn()}
}

func checkSnapshot(t *testing.T, g *Game, want snapshot, where string) {
	t.Helper()
	got := takeSnapshot(g)
	dim := want.board.Dim()
	for r := 0; r < dim; r++ {
		for c := 0; c < dim; c++ {
			if got.board.GetLetter(r, c) != want.board.GetLetter(r, c) {
				t.Fatalf("%s: square %d,%d differs", where, r, c)
			}
			for _, d := range []board.BoardDirection{board.HorizontalDirection, board.VerticalDirection} {
				if got.board.GetCrossSet(r, c, d) != want.board.GetCrossSet(r, c, d) ||
					got.board.GetCrossScore(r, c, d) != want.board.GetCrossScore(r, c, d) {
					t.Fatalf("%s: cross-set or cross-score at %d,%d differs", where, r, c)
				}
			}
		}
	}
	if got.racks != want.racks || got.points != want.points || got.bag != want.bag || got.onturn != want.onturn {
		t.Fatalf("%s: racks/points/bag/turn %+v, want %+v", where, got, want)
	}
}

func nwl23Rules(t *testing.T) *GameRules {
	rules, err := NewBasicGameRules(DefaultConfig, "NWL23", board.CrosswordGameLayout,
		"english", CrossScoreAndSet, VarClassic)
	if err != nil {
		t.Fatal(err)
	}
	return rules
}

func bestMove(t *testing.T, g *Game) *move.Move {
	pos, err := g.MagpiePosition()
	if err != nil {
		t.Fatal(err)
	}
	return g.MoveFromMagpie(&pos.Generate(magpie.RecordBest, magpie.SortEquity, 0)[0])
}

// Unplaying a move puts the board, racks, bag and scores back exactly, in
// simulation mode, through random games of MAGPIE's own moves.
func TestUnplayRestores(t *testing.T) {
	is := is.New(t)
	rules := nwl23Rules(t)
	rng := rand.New(rand.NewPCG(1, 2))
	for gi := 0; gi < 20; gi++ {
		g, err := NewGame(rules, []*pb.PlayerInfo{{Nickname: "a"}, {Nickname: "b"}})
		is.NoErr(err)
		g.StartGame()
		g.SetStateStackLength(40)
		g.SetBackupMode(SimulationMode)
		var history []snapshot
		for g.Playing() == pb.PlayState_PLAYING && len(history) < 38 {
			before := takeSnapshot(g)
			pos, err := g.MagpiePosition()
			is.NoErr(err)
			moves := pos.GenerateN(magpie.RecordAll, magpie.SortEquity, 0, 5)
			m := g.MoveFromMagpie(&moves[rng.IntN(len(moves))])
			is.NoErr(g.PlayMove(m, false, 0))
			history = append(history, before)
			if len(history) > 1 && rng.IntN(3) == 0 {
				g.UnplayLastMove()
				checkSnapshot(t, g, history[len(history)-1], "after unplay")
				history = history[:len(history)-1]
			}
		}
		// Unwind the whole game.
		for i := len(history) - 1; i >= 0; i-- {
			g.UnplayLastMove()
			checkSnapshot(t, g, history[i], "unwinding")
		}
	}
}

// The simmer's pattern: back up the candidate, play further plies without
// backups, then reset to the first state.
func TestResetToFirstStateRestores(t *testing.T) {
	is := is.New(t)
	rules := nwl23Rules(t)
	rng := rand.New(rand.NewPCG(3, 4))
	g, err := NewGame(rules, []*pb.PlayerInfo{{Nickname: "a"}, {Nickname: "b"}})
	is.NoErr(err)
	g.StartGame()
	g.SetStateStackLength(1)
	for i := 0; i < 6; i++ {
		is.NoErr(g.PlayMove(bestMove(t, g), false, 0))
	}
	pos, err := g.MagpiePosition()
	is.NoErr(err)
	candidates := pos.GenerateN(magpie.RecordAll, magpie.SortEquity, 0, 10)
	plays := make([]*move.Move, len(candidates))
	for i := range candidates {
		plays[i] = g.MoveFromMagpie(&candidates[i])
	}
	start := takeSnapshot(g)
	for iter := 0; iter < 50; iter++ {
		for _, c := range plays {
			g.SetBackupMode(SimulationMode)
			is.NoErr(g.PlayMove(c, false, 0))
			g.SetBackupMode(NoBackup)
			for ply := 0; ply < 2+rng.IntN(3) && g.Playing() == pb.PlayState_PLAYING; ply++ {
				is.NoErr(g.PlayMove(bestMove(t, g), false, 0))
			}
			g.ResetToFirstState()
			checkSnapshot(t, g, start, "after reset")
		}
	}
}

// Endgame small moves play and unplay cleanly, trees a few plies deep.
func TestSmallMovesUnplay(t *testing.T) {
	is := is.New(t)
	rules := nwl23Rules(t)
	buf := make([]magpie.SmallMove, 50000)
	for gi := 0; gi < 10; gi++ {
		g, err := NewGame(rules, []*pb.PlayerInfo{{Nickname: "a"}, {Nickname: "b"}})
		is.NoErr(err)
		g.StartGame()
		for g.Bag().TilesRemaining() > 0 && g.Playing() == pb.PlayState_PLAYING {
			is.NoErr(g.PlayMove(bestMove(t, g), false, 0))
		}
		if g.Playing() != pb.PlayState_PLAYING {
			continue
		}
		g.SetStateStackLength(10)
		g.SetBackupMode(SimulationMode)
		g.SetEndgameMode(true)
		var walk func(depth int)
		walk = func(depth int) {
			if depth == 0 || g.Playing() != pb.PlayState_PLAYING {
				return
			}
			pos, err := g.MagpiePosition()
			is.NoErr(err)
			moves := append([]magpie.SmallMove(nil), pos.GenerateSmall(true, buf)...)
			for i := 0; i < len(moves); i += 1 + len(moves)/6 {
				sm := tinymove.PassMove()
				if moves[i].TM != 0 {
					sm = tinymove.TilePlayMove(tinymove.TinyMove(moves[i].TM), moves[i].Score,
						moves[i].TilesPlayed, moves[i].PlayLength)
				}
				before := takeSnapshot(g)
				_, err := g.PlaySmallMove(&sm)
				is.NoErr(err)
				walk(depth - 1)
				g.UnplayLastMove()
				checkSnapshot(t, g, before, "after small unplay")
			}
		}
		walk(3)
	}
}
