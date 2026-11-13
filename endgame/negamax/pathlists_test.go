package negamax

import (
	"context"
	"os"
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/zobrist"
)

// ENDGAME_CHECK_LISTS=1 makes every solve in this package's tests verify each
// incremental move list against a full generation.
func init() {
	if os.Getenv("ENDGAME_CHECK_LISTS") != "" {
		checkMoveLists = true
	}
}

func TestTileCounts(t *testing.T) {
	is := is.New(t)
	rack := tileCounts{tileCountHighBits, tileCountHighBits, tileCountHighBits}
	rack[0] += 2 << (5 * 4) // two of tile 5
	rack[1] += 1 << (3 * 4) // one of tile 19
	var one, three, other tileCounts
	one[0] = 1 << (5 * 4)
	three[0] = 3 << (5 * 4)
	other[1] = 1 << (4 * 4)
	is.True(rack.holds(&tileCounts{}))
	is.True(rack.holds(&one))
	is.True(!rack.holds(&three))
	is.True(!rack.holds(&other))
	other[1] = 1 << (3 * 4)
	is.True(rack.holds(&other))
}

// The key of a position reached by a tile play, found without playing it,
// must be the key AddMove gives after playing it.
func TestUnplayedChildKey(t *testing.T) {
	is := is.New(t)
	s, err := setUpSolver("NWL20", "english", board.VsCanik, 4, "DEHILOR", "BGIV", 389, 384, 1)
	is.NoErr(err)
	g := s.game
	g.SetStateStackLength(4)
	z := &zobrist.Zobrist{}
	z.Initialize(g.Board().Dim())
	onTurn := g.PlayerOnTurn()
	rack := g.RackFor(onTurn)
	rootKey := z.Hash(g.Board().GetSquares(), rack, g.RackFor(1-onTurn), false, g.ScorelessTurns())
	s.stmMovegen.GenAll(rack, false)
	plays := append([]tinymove.SmallMove{}, s.stmMovegen.SmallPlays()...)
	checked := 0
	for i := range plays {
		if plays[i].IsPass() {
			continue
		}
		var tiles [board.MaxBoardDim]tilemapping.MachineLetter
		smallMoveTiles(g.Board(), &plays[i], &tiles)
		unplayed := z.AddTilePlayUnplayed(rootKey, &plays[i], rack, &tiles, true, g.ScorelessTurns())
		moveTiles, err := g.PlaySmallMove(&plays[i])
		is.NoErr(err)
		played := z.AddMove(rootKey, &plays[i], rack, moveTiles, true,
			g.ScorelessTurns(), g.LastScorelessTurns())
		g.UnplayLastMove()
		is.Equal(unplayed, played)
		checked++
	}
	is.True(checked > 10)
}

// A solve gives the same value with the incremental lists, the child
// shortcuts, both, and neither.
func TestListsAndShortcutsKeepValue(t *testing.T) {
	is := is.New(t)
	for _, tc := range []struct{ lists, shortcuts bool }{
		{false, false}, {true, false}, {false, true}, {true, true},
	} {
		s, err := setUpSolver("NWL20", "english", board.VsCanik, 4, "DEHILOR", "BGIV", 389, 384, 1)
		is.NoErr(err)
		s.SetThreads(1)
		s.SetIncrementalMoveLists(tc.lists)
		s.SetChildShortcuts(tc.shortcuts)
		v, _, err := s.Solve(context.Background(), 4)
		is.NoErr(err)
		is.Equal(v, int16(11))
	}
}

// boardsIdentical compares everything a move can change on a board,
// extension sets included (GameBoard.Equals leaves those out).
func boardsIdentical(a, b *board.GameBoard) bool {
	if !a.Equals(b) {
		return false
	}
	n := a.Dim() * a.Dim()
	for _, dir := range []board.BoardDirection{board.HorizontalDirection, board.VerticalDirection} {
		for pos := 0; pos < n; pos++ {
			if a.GetLeftExtSetIdx(pos, dir) != b.GetLeftExtSetIdx(pos, dir) ||
				a.GetRightExtSetIdx(pos, dir) != b.GetRightExtSetIdx(pos, dir) {
				return false
			}
		}
	}
	return true
}

// TestIncrementalUndo plays every move and every reply to it with the board's
// undo journal on, and checks that unplaying leaves the board exactly as a
// whole-board copy has it, and that the board after a move is the same
// either way.
func TestIncrementalUndo(t *testing.T) {
	is := is.New(t)
	s, err := setUpSolver("NWL20", "english", board.VsCanik, 4, "DEHILOR", "BGIV", 389, 384, 1)
	is.NoErr(err)
	g := s.game
	g.SetStateStackLength(4)
	ref := g.Copy()
	ref.SetBackupMode(game.SimulationMode)
	g.SetIncrementalUndo(true)
	defer g.SetIncrementalUndo(false)

	root := g.Board().Copy()
	s.stmMovegen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
	plays := append([]tinymove.SmallMove{}, s.stmMovegen.SmallPlays()...)
	replies := 0
	for i := range plays {
		_, err := g.PlaySmallMove(&plays[i])
		is.NoErr(err)
		_, err = ref.PlaySmallMove(&plays[i])
		is.NoErr(err)
		is.True(boardsIdentical(g.Board(), ref.Board()))
		if i%7 == 0 && g.Playing() == pb.PlayState_PLAYING {
			after := g.Board().Copy()
			s.stmMovegen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
			second := append([]tinymove.SmallMove{}, s.stmMovegen.SmallPlays()...)
			for j := range second {
				_, err := g.PlaySmallMove(&second[j])
				is.NoErr(err)
				g.UnplayLastMove()
				is.True(boardsIdentical(g.Board(), after))
				replies++
			}
		}
		g.UnplayLastMove()
		ref.UnplayLastMove()
		is.True(boardsIdentical(g.Board(), root))
		is.Equal(g.Board().UndoMark(), 0)
	}
	is.True(replies > 10)
}

// TestKeepOneBoundKeepsValue solves with the keep-one bound off and on and
// expects the same value. With ENDGAME_CHECK_LISTS set the solver also
// searches every move the bound would skip and panics if one beats alpha.
func TestKeepOneBoundKeepsValue(t *testing.T) {
	is := is.New(t)
	for _, plies := range []int{3, 4, 5} {
		var values []int16
		for _, on := range []bool{false, true} {
			s, err := setUpSolver("NWL20", "english", board.VsCanik, plies, "DEHILOR", "BGIV", 389, 384, 1)
			is.NoErr(err)
			s.SetThreads(1)
			s.SetKeepOneBound(on)
			v, _, err := s.Solve(context.Background(), plies)
			is.NoErr(err)
			values = append(values, v)
		}
		is.Equal(values[0], values[1])
	}
}

// TestOutReplyBoundKeepsValue solves with the out-reply bound off and on,
// with either player to move, and expects the same values.
func TestOutReplyBoundKeepsValue(t *testing.T) {
	is := is.New(t)
	for _, tc := range []struct {
		rack1, rack2 string
		pts1, pts2   int
		onTurn       int
	}{
		{"DEHILOR", "BGIV", 389, 384, 1},
		{"DEHILOR", "BGIV", 389, 384, 0},
	} {
		for _, plies := range []int{3, 4} {
			var values []int16
			for _, on := range []bool{false, true} {
				s, err := setUpSolver("NWL20", "english", board.VsCanik, plies, tc.rack1, tc.rack2, tc.pts1, tc.pts2, tc.onTurn)
				is.NoErr(err)
				s.SetThreads(1)
				s.SetOutReplyBound(on)
				v, _, err := s.Solve(context.Background(), plies)
				is.NoErr(err)
				values = append(values, v)
			}
			is.Equal(values[0], values[1])
		}
	}
}

// TestSelfCheckedSolve solves with the solver's self-checks on, whatever the
// environment says: every incremental move list is compared with a full
// generation of the position, every derived one-tile play with a scan of the
// board, and every move the keep-one or out-reply bound would skip is
// searched to see that it fails low. Any difference panics.
func TestSelfCheckedSolve(t *testing.T) {
	is := is.New(t)
	was := checkMoveLists
	checkMoveLists = true
	defer func() { checkMoveLists = was }()
	for _, threads := range []int{1, 3} {
		s, err := setUpSolver("NWL20", "english", board.VsCanik, 5, "DEHILOR", "BGIV", 389, 384, 1)
		is.NoErr(err)
		s.SetThreads(threads)
		v, _, err := s.Solve(context.Background(), 5)
		is.NoErr(err)
		is.Equal(v, int16(11))
	}
}

// The cross-set caches change only the speed of a solve: the value is the
// same with them off, they are actually consulted, each thread has its own,
// and the game gets its own generator back afterwards.
func TestCrossSetCacheKeepsValue(t *testing.T) {
	is := is.New(t)
	for _, threads := range []int{1, 3} {
		values := []int16{}
		for _, on := range []bool{false, true} {
			s, err := setUpSolver("NWL20", "english", board.VsCanik, 5, "DEHILOR", "BGIV", 389, 384, 1)
			is.NoErr(err)
			s.SetThreads(threads)
			s.SetCrossSetCache(on)
			orig := s.game.CrossSetGen()
			v, _, err := s.Solve(context.Background(), 5)
			is.NoErr(err)
			values = append(values, v)
			is.Equal(s.game.CrossSetGen(), orig)
			if !on {
				is.Equal(len(s.crossSetCaches), 0)
				continue
			}
			is.Equal(len(s.crossSetCaches), threads)
			for i, c := range s.crossSetCaches {
				is.True(c.Hits > c.Misses)
				for j := 0; j < i; j++ {
					is.True(c != s.crossSetCaches[j])
				}
			}
		}
		is.Equal(values[0], int16(11))
		is.Equal(values[1], int16(11))
	}
}
