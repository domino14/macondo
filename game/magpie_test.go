package game

import (
	"math/rand/v2"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/move"
)

// checkPosMatches checks MAGPIE's copy of the position against the Go board,
// racks and bag, without letting MagpiePosition rebuild it.
func checkPosMatches(t *testing.T, g *Game, where string) {
	t.Helper()
	if g.posStale {
		t.Fatalf("%s: MAGPIE's copy fell out of step", where)
	}
	g.pushRacks()
	d, _ := g.rules.MagpieData()
	mb := d.NewBoard()
	g.pos.ReadBoard(mb)
	dim := g.board.Dim()
	letters := uint64(1)<<len(g.letterDistribution.Distribution()) - 2
	for r := 0; r < dim; r++ {
		for c := 0; c < dim; c++ {
			i := r*dim + c
			if mb.Letters[i] != g.board.GetLetter(r, c) {
				t.Fatalf("%s: square %d,%d differs", where, r, c)
			}
			if g.board.GetLetter(r, c) != 0 {
				continue
			}
			for dir, md := range []board.BoardDirection{board.VerticalDirection, board.HorizontalDirection} {
				if uint64(g.board.GetCrossSet(r, c, md))&letters != mb.CrossSets[dir*dim*dim+i]&letters {
					t.Fatalf("%s: cross-set %d,%d dir %d differs", where, r, c, dir)
				}
			}
		}
	}
	r0, r1, nbag := g.pos.RackCounts()
	for i := range r0 {
		if int(r0[i]) != g.players[0].rack.LetArr[i] || int(r1[i]) != g.players[1].rack.LetArr[i] {
			t.Fatalf("%s: racks differ", where)
		}
	}
	if nbag != g.bag.TilesRemaining() {
		t.Fatalf("%s: bag has %d in MAGPIE, %d in Macondo", where, nbag, g.bag.TilesRemaining())
	}
}

// MAGPIE's copy follows a game played through the Game API -- moves picked
// from MAGPIE's own generation, simulation-mode backups and unplays -- without
// ever having to be rebuilt.
func TestMagpiePositionFollowsGame(t *testing.T) {
	is := is.New(t)
	rules, err := NewBasicGameRules(DefaultConfig, "NWL23", board.CrosswordGameLayout,
		"english", CrossScoreAndSet, VarClassic)
	is.NoErr(err)
	rng := rand.New(rand.NewPCG(1, 2))
	for gi := 0; gi < 30; gi++ {
		g, err := NewGame(rules, []*pb.PlayerInfo{{Nickname: "a"}, {Nickname: "b"}})
		is.NoErr(err)
		g.StartGame()
		g.SetStateStackLength(40)
		g.SetBackupMode(SimulationMode)
		_, err = g.MagpiePosition()
		is.NoErr(err)
		checkPosMatches(t, g, "start")
		played := 0
		for g.Playing() == pb.PlayState_PLAYING && played < 38 {
			pos, err := g.MagpiePosition()
			is.NoErr(err)
			moves := pos.GenerateN(magpie.RecordAll, magpie.SortEquity, 0, 5)
			m := g.MoveFromMagpie(&moves[rng.IntN(len(moves))])
			is.NoErr(g.PlayMove(m, false, 0))
			played++
			checkPosMatches(t, g, "after play")
			if played > 1 && rng.IntN(3) == 0 {
				g.UnplayLastMove()
				played--
				checkPosMatches(t, g, "after unplay")
			}
		}
	}
}

// The simmer's pattern: back up the candidate, play further plies without
// backups, then reset to the first state. MAGPIE's copy must follow without
// a rebuild.
func TestMagpiePositionFollowsSimPattern(t *testing.T) {
	is := is.New(t)
	rules, err := NewBasicGameRules(DefaultConfig, "NWL23", board.CrosswordGameLayout,
		"english", CrossScoreAndSet, VarClassic)
	is.NoErr(err)
	rng := rand.New(rand.NewPCG(3, 4))
	g, err := NewGame(rules, []*pb.PlayerInfo{{Nickname: "a"}, {Nickname: "b"}})
	is.NoErr(err)
	g.StartGame()
	g.SetStateStackLength(1)
	// Get a few moves in first.
	for i := 0; i < 6; i++ {
		pos, err := g.MagpiePosition()
		is.NoErr(err)
		m := g.MoveFromMagpie(&pos.Generate(magpie.RecordBest, magpie.SortEquity, 0)[0])
		is.NoErr(g.PlayMove(m, false, 0))
	}
	pos, err := g.MagpiePosition()
	is.NoErr(err)
	candidates := pos.GenerateN(magpie.RecordAll, magpie.SortEquity, 0, 10)
	plays := make([]*move.Move, len(candidates))
	for i := range candidates {
		plays[i] = g.MoveFromMagpie(&candidates[i])
	}
	checkPosMatches(t, g, "before sim")
	for iter := 0; iter < 50; iter++ {
		for _, c := range plays {
			g.SetBackupMode(SimulationMode)
			is.NoErr(g.PlayMove(c, false, 0))
			g.SetBackupMode(NoBackup)
			for ply := 0; ply < 2+rng.IntN(3) && g.Playing() == pb.PlayState_PLAYING; ply++ {
				pos, err := g.MagpiePosition()
				is.NoErr(err)
				m := g.MoveFromMagpie(&pos.Generate(magpie.RecordBest, magpie.SortEquity, 0)[0])
				is.NoErr(g.PlayMove(m, false, 0))
			}
			g.ResetToFirstState()
			checkPosMatches(t, g, "after reset")
		}
	}
}
