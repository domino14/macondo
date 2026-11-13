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
