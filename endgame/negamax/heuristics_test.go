package negamax

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/tinymove/conversions"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

// Eldar vs Nigel. The player on turn holds AEEIRUW; the opponent is stuck with
// a lone V that cannot be played anywhere. See TestStuckPruning for the optimal
// sequences; the true value is +72, and reaching it by search alone takes 11
// plies and a long time.
const stuckEndgameCGP = "4EXODE6/1DOFF1KERATIN1U/1OHO8YEN/1POOJA1B3MEWS/5SQUINTY2A/4RHINO1e3V/2B4C2R3E/GOAT1D1E2ZIN1d/1URACILS2E4/1PIG1S4T4/2L2R4T4/2L2A1GENII3/2A2T1L7/5E1A7/5D1M7 AEEIRUW/V 410/409 0 lex CSW19;"

// A position where nothing is stuck: both players have common tiles and the
// board is open.
const openEndgameCGP = "GATELEGs1POGOED/R4MOOLI3X1/AA10U2/YU4BREDRIN2/1TITULE3E1IN1/1E4N3c1BOK/1C2O4CHARD1/QI1FLAWN2E1OE1/IS2E1HIN1A1W2/1MOTIVATE1T1S2/1S2N5S4/3PERJURY5/15/15/15 FV/AADIZ 442/388 0 lex CSW19;"

// solverFromCGP builds a solver for a CGP position, ready for Solve.
func solverFromCGP(t *testing.T, pos, lex string) *Solver {
	t.Helper()
	g, err := cgp.ParseCGP(DefaultConfig, pos)
	if err != nil {
		t.Fatal(err)
	}
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), lex)
	if err != nil {
		t.Fatal(err)
	}
	g.SetBackupMode(game.SimulationMode)
	g.RecalculateBoard()
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
	s := new(Solver)
	if err := s.Init(gen, g.Game); err != nil {
		t.Fatal(err)
	}
	s.SetThreads(1)
	return s
}

// prepForLeafCall sets up a solver so that greedyPlayout / oppStuckFraction /
// assignEstimates can be called directly on the root position, the way negamax
// would call them at a leaf.
func prepForLeafCall(t *testing.T, s *Solver, stackLen int) {
	t.Helper()
	s.solvingPlayer = s.game.PlayerOnTurn()
	s.requestedPlies = 0
	s.ensureArenas()
	s.game.SetEndgameMode(true)
	s.game.SetBackupMode(game.SimulationMode)
	s.game.SetStateStackLength(stackLen)
	s.stmMovegen.SetSortingParameter(movegen.SortByNone)
}

// TestStuckEndgameSolvedShallow is the headline claim: on a position that takes
// 11 plies to solve exactly, the greedy playout finds the true value at a
// trivial depth, essentially instantly, while the static leaf does not.
func TestStuckEndgameSolvedShallow(t *testing.T) {
	is := is.New(t)
	const trueValue = int16(72)

	for _, plies := range []int{2, 3} {
		s := solverFromCGP(t, stuckEndgameCGP, "CSW19")
		v, seq, err := s.Solve(context.Background(), plies)
		is.NoErr(err)
		is.Equal(v, trueValue)
		// The playout carries the variation all the way to the end of the
		// game, past what the search itself proved.
		is.True(len(seq) > plies)
		is.Equal(s.NumSearchedMoves(), plies)

		off := solverFromCGP(t, stuckEndgameCGP, "CSW19")
		off.SetUseHeuristics(false)
		vOff, _, err := off.Solve(context.Background(), plies)
		is.NoErr(err)
		is.True(vOff != trueValue)
	}
}

func TestOppStuckFraction(t *testing.T) {
	is := is.New(t)

	// Opponent holds a lone V with nowhere to put it: every point of their rack
	// is stuck.
	s := solverFromCGP(t, stuckEndgameCGP, "CSW19")
	prepForLeafCall(t, s, 30)
	is.Equal(s.oppStuckFraction(s.game, s.stmMovegen, 0, 0), float32(1.0))

	// Nothing is stuck on an open board with common tiles.
	open := solverFromCGP(t, openEndgameCGP, "CSW19")
	prepForLeafCall(t, open, 30)
	is.Equal(open.oppStuckFraction(open.game, open.stmMovegen, 0, 0), float32(0.0))

	// Now the multi-tile cases. The single-tile cross-set scan cannot settle
	// these on its own -- a tile with no single-tile play may still play inside
	// a longer word -- so a partial result falls through to move generation.

	// Add a B: it plays, the V still does not, so the stuck share is the V's 4
	// points out of the rack's 7.
	withB := solverFromCGP(t, stuckEndgameCGP, "CSW19")
	withB.game.RackFor(1).Add(tilemapping.MachineLetter(2)) // B
	prepForLeafCall(t, withB, 30)
	is.Equal(withB.game.RackLettersFor(1), "BV")
	is.Equal(withB.oppStuckFraction(withB.game, withB.stmMovegen, 0, 0), float32(4)/float32(7))

	// Add an E instead and the V stops being stuck entirely, because it can now
	// go down as part of a longer word. This is exactly the case the fast scan
	// gets wrong on its own, and why the fallback exists.
	withE := solverFromCGP(t, stuckEndgameCGP, "CSW19")
	withE.game.RackFor(1).Add(tilemapping.MachineLetter(5)) // E
	prepForLeafCall(t, withE, 30)
	is.Equal(withE.game.RackLettersFor(1), "EV")
	is.Equal(withE.oppStuckFraction(withE.game, withE.stmMovegen, 0, 0), float32(0.0))
}

// TestGreedyPlayoutRestoresState checks that the playout leaves the position
// exactly as it found it. If it did not, leaf values would depend on the path
// the search took to get there rather than on the position.
func TestGreedyPlayoutRestoresState(t *testing.T) {
	is := is.New(t)
	s := solverFromCGP(t, openEndgameCGP, "CSW19")
	prepForLeafCall(t, s, 40)
	g := s.game
	bd := g.Board()

	squares := slices.Clone(bd.SquaresSlice())
	hCross := slices.Clone(bd.CrossSetsForDir(board.HorizontalDirection))
	vCross := slices.Clone(bd.CrossSetsForDir(board.VerticalDirection))
	rack0 := g.RackLettersFor(0)
	rack1 := g.RackLettersFor(1)
	pts0, pts1 := g.PointsFor(0), g.PointsFor(1)
	onTurn := g.PlayerOnTurn()
	scoreless := g.ScorelessTurns()
	stackRemaining := g.StateStackRemaining()

	var pv PVLine
	v1 := s.greedyPlayout(g, s.stmMovegen, &pv, 0, 0)

	is.Equal(bd.SquaresSlice(), squares)
	is.Equal(bd.CrossSetsForDir(board.HorizontalDirection), hCross)
	is.Equal(bd.CrossSetsForDir(board.VerticalDirection), vCross)
	is.Equal(g.RackLettersFor(0), rack0)
	is.Equal(g.RackLettersFor(1), rack1)
	is.Equal(g.PointsFor(0), pts0)
	is.Equal(g.PointsFor(1), pts1)
	is.Equal(g.PlayerOnTurn(), onTurn)
	is.Equal(g.ScorelessTurns(), scoreless)
	is.Equal(g.StateStackRemaining(), stackRemaining)
	is.Equal(g.Playing(), pb.PlayState_PLAYING)

	// Same position, same answer.
	v2 := s.greedyPlayout(g, s.stmMovegen, &pv, 0, 0)
	is.Equal(v1, v2)

	// And playing a move, evaluating, and taking it back must not disturb the
	// value at the original position either.
	s.stmMovegen.GenAll(g.RackFor(onTurn), false)
	plays := slices.Clone(s.stmMovegen.SmallPlays())
	for i := 0; i < len(plays) && i < 10; i++ {
		_, err := g.PlaySmallMove(&plays[i])
		is.NoErr(err)
		if g.Playing() == pb.PlayState_PLAYING {
			s.greedyPlayout(g, s.stmMovegen, &pv, 0, 0)
		}
		g.UnplayLastMove()
	}
	is.Equal(s.greedyPlayout(g, s.stmMovegen, &pv, 0, 0), v1)
}

// TestGreedyPlayoutStackBudget checks that the playout respects the backup
// stack it was given. Callers of QuickAndDirtySolve size the stack themselves,
// so overrunning it would panic.
func TestGreedyPlayoutStackBudget(t *testing.T) {
	is := is.New(t)
	s := solverFromCGP(t, openEndgameCGP, "CSW19")
	prepForLeafCall(t, s, 40)
	g := s.game

	var pv PVLine
	full := s.greedyPlayout(g, s.stmMovegen, &pv, 0, 0)
	is.True(pv.NumMoves() > 0)

	// With no room at all, fall back to the static spread and touch nothing.
	g.SetStateStackLength(0)
	is.Equal(g.StateStackRemaining(), 0)
	static := s.greedyPlayout(g, s.stmMovegen, &pv, 0, 0)
	is.Equal(static, int16(g.SpreadFor(g.PlayerOnTurn())))
	is.Equal(pv.NumMoves(), 0)
	is.True(static != full)

	// A short stack should still play what it can rather than panic.
	g.SetStateStackLength(2)
	short := s.greedyPlayout(g, s.stmMovegen, &pv, 0, 0)
	is.True(pv.NumMoves() <= 2)
	is.Equal(g.StateStackRemaining(), 2)
	_ = short
}

// TestPassPenaltyPreventsDoublePass guards the deliberate divergence from
// MAGPIE. In conserve mode every real play takes a conservation penalty; if a
// pass did not take one too, passing would win the greedy pick and the playout
// would end the game immediately with a pessimistic value.
func TestPassPenaltyPreventsDoublePass(t *testing.T) {
	is := is.New(t)
	s := solverFromCGP(t, stuckEndgameCGP, "CSW19")
	prepForLeafCall(t, s, 40)
	g := s.game

	frac := s.oppStuckFraction(g, s.stmMovegen, 0, 0)
	is.Equal(frac, float32(1.0))

	// Establish that the penalty is load-bearing here: with the conservation
	// bonus applied, no real play's adjusted score beats zero, so an
	// unpenalized pass would win the greedy pick. The penalty is what puts the
	// pass behind them.
	ld := g.Bag().LetterDistribution()
	onTurn := g.PlayerOnTurn()
	s.stmMovegen.GenAll(g.RackFor(onTurn), false)
	plays := slices.Clone(s.stmMovegen.SmallPlays())
	bestUnpenalized := math.MinInt32
	for i := range plays {
		if plays[i].IsPass() {
			continue
		}
		bestUnpenalized = max(bestUnpenalized, plays[i].Score()-conservationBonus(&plays[i], ld, frac))
	}
	passPenalty := int(float32(g.RackFor(onTurn).ScoreOn(ld)+g.RackFor(1-onTurn).ScoreOn(ld)) * frac)
	t.Logf("best adjusted real play: %d, unpenalized pass: 0, penalized pass: %d",
		bestUnpenalized, -passPenalty)
	is.True(bestUnpenalized <= 0)
	is.True(bestUnpenalized > -passPenalty)

	var pv PVLine
	v := s.greedyPlayout(g, s.stmMovegen, &pv, 0, 0)
	t.Logf("greedy playout value %d over %d moves", v, pv.NumMoves())
	// It plays the rack off instead of passing out immediately.
	is.True(!pv.tinyMoves[0].IsPass())
	is.True(pv.NumMoves() > 4)
	is.Equal(pv.NumSearchedMoves(), 0)
	// Passing out immediately would end the game on two scoreless turns, with
	// each player docked their own rack: (410-10) - (409-4) = -5.
	is.True(v > 0)
}

// TestBuildChainValues checks the build-chain estimate: a short play contained
// in a longer one is worth its own score plus the best containing play's value,
// so that the search tries extendable plays first when there is time to build.
func TestBuildChainValues(t *testing.T) {
	is := is.New(t)
	s := solverFromCGP(t, stuckEndgameCGP, "CSW19")
	prepForLeafCall(t, s, 30)
	g := s.game

	s.stmMovegen.GenAll(g.RackFor(g.PlayerOnTurn()), false)
	moves := slices.Clone(s.stmMovegen.SmallPlays())
	is.True(len(moves) > 1)

	// No build chain unless the opponent is stuck.
	is.Equal(s.computeBuildChainValues(moves, g.Board(), 0, 0), nil)

	got := s.computeBuildChainValues(moves, g.Board(), 1.0, 0)
	is.True(got != nil)
	is.Equal(len(got), len(moves))

	want := referenceBuildChainValues(moves, g.Board())
	is.Equal(got, want)

	// The test would be vacuous if this position had no nested plays at all.
	boosted := 0
	for i := range moves {
		is.True(got[i] >= moves[i].Score())
		if got[i] > moves[i].Score() {
			boosted++
		}
	}
	is.True(boosted > 0)
}

// referenceBuildChainValues recomputes the build chain the slow, obvious way:
// it expands every move to a full Move first and works from real board
// coordinates and tiles, rather than from the SmallMove bit layout.
func referenceBuildChainValues(moves []tinymove.SmallMove, bd *board.GameBoard) []int {
	type expanded struct {
		row, col int
		vertical bool
		length   int
		tiles    tilemapping.MachineWord
		pass     bool
	}
	exp := make([]expanded, len(moves))
	for i := range moves {
		if moves[i].IsPass() {
			exp[i].pass = true
			continue
		}
		var m move.Move
		conversions.TinyMoveToMove(moves[i].TinyMove(), bd, &m)
		r, c, v := m.CoordsAndVertical()
		exp[i] = expanded{row: r, col: c, vertical: v,
			length: moves[i].PlayLength(), tiles: slices.Clone(m.Tiles())}
	}

	order := make([]int, len(moves))
	for i := range order {
		order[i] = i
	}
	// Longest first, so a move's containers are already final when we reach it.
	slices.SortStableFunc(order, func(a, b int) int {
		return moves[b].TilesPlayed() - moves[a].TilesPlayed()
	})

	values := make([]int, len(moves))
	for oi, i := range order {
		values[i] = moves[i].Score()
		if exp[i].pass {
			continue
		}
		best := 0
		for _, j := range order[:oi] {
			if exp[j].pass || moves[j].TilesPlayed() <= moves[i].TilesPlayed() {
				continue
			}
			if exp[j].vertical != exp[i].vertical {
				continue
			}
			var contained bool
			var offset int
			if exp[i].vertical {
				contained = exp[i].col == exp[j].col && exp[i].row >= exp[j].row &&
					exp[i].row+exp[i].length <= exp[j].row+exp[j].length
				offset = exp[i].row - exp[j].row
			} else {
				contained = exp[i].row == exp[j].row && exp[i].col >= exp[j].col &&
					exp[i].col+exp[i].length <= exp[j].col+exp[j].length
				offset = exp[i].col - exp[j].col
			}
			if !contained {
				continue
			}
			match := true
			for ti := 0; ti < exp[i].length && ti < len(exp[i].tiles); ti++ {
				if exp[i].tiles[ti] == 0 {
					continue // play-through: comes from the board, not this move
				}
				if ti+offset >= len(exp[j].tiles) || exp[i].tiles[ti] != exp[j].tiles[ti+offset] {
					match = false
					break
				}
			}
			if match && values[j] > best {
				best = values[j]
			}
		}
		values[i] += best
	}
	return values
}

// TestMagpieOrdering keeps the ported MAGPIE move ordering exercised even
// though it is off by default (macondo's existing ordering measures better --
// see the magpieOrdering field comment). Ordering cannot change the value, so
// both settings must agree.
func TestMagpieOrdering(t *testing.T) {
	is := is.New(t)
	for _, plies := range []int{2, 3} {
		def := solverFromCGP(t, stuckEndgameCGP, "CSW19")
		vDefault, _, err := def.Solve(context.Background(), plies)
		is.NoErr(err)

		magpie := solverFromCGP(t, stuckEndgameCGP, "CSW19")
		magpie.SetMagpieOrdering(true)
		vMagpie, _, err := magpie.Solve(context.Background(), plies)
		is.NoErr(err)

		is.Equal(vDefault, int16(72))
		is.Equal(vMagpie, vDefault)
	}
}

// TestHeuristicToggleEquivalence: the heuristics only change leaf estimates and
// move ordering, so once the search is deep enough to play the endgame out both
// settings must agree exactly.
func TestHeuristicToggleEquivalence(t *testing.T) {
	is := is.New(t)
	// FV vs AADIZ is 7 tiles, so 9 plies covers every line; use 10 for margin.
	for _, plies := range []int{10, 11} {
		on := solverFromCGP(t, openEndgameCGP, "CSW19")
		vOn, _, err := on.Solve(context.Background(), plies)
		is.NoErr(err)

		off := solverFromCGP(t, openEndgameCGP, "CSW19")
		off.SetUseHeuristics(false)
		vOff, _, err := off.Solve(context.Background(), plies)
		is.NoErr(err)

		is.Equal(vOn, vOff)
	}
}

// TestHeuristicShallowAccuracy: across a spread of positions and shallow ply
// counts, the greedy leaf should get closer to the true value in aggregate than
// the static leaf does. It is not required to win every single cell -- a rollout
// can overshoot at a given depth -- but it should win overall.
func TestHeuristicShallowAccuracy(t *testing.T) {
	is := is.New(t)
	type fixture struct {
		name  string
		setup func() (*Solver, error)
		truth int16
	}
	fixtures := []fixture{
		{"VsCanik", func() (*Solver, error) {
			return setUpSolver("NWL20", "english", board.VsCanik, 0, "DEHILOR", "BGIV", 389, 384, 1)
		}, 11},
		{"VsJoel", func() (*Solver, error) {
			return setUpSolver("NWL18", "english", board.VsJoel, 0, "EIQSS", "AAFIRTW", 393, 373, 1)
		}, 25},
		{"VsRoy", func() (*Solver, error) {
			return setUpSolver("America", "english", board.VsRoy, 0, "WZ", "EFHIKOQ", 427, 331, 1)
		}, 116},
	}

	totalOn, totalOff := 0, 0
	for _, f := range fixtures {
		for _, plies := range []int{2, 3, 4} {
			on, err := f.setup()
			is.NoErr(err)
			on.SetThreads(1)
			vOn, _, err := on.Solve(context.Background(), plies)
			is.NoErr(err)

			off, err := f.setup()
			is.NoErr(err)
			off.SetThreads(1)
			off.SetUseHeuristics(false)
			vOff, _, err := off.Solve(context.Background(), plies)
			is.NoErr(err)

			errOn := absInt(int(vOn) - int(f.truth))
			errOff := absInt(int(vOff) - int(f.truth))
			totalOn += errOn
			totalOff += errOff
			t.Logf("%s plies=%d truth=%d on=%d (err %d) off=%d (err %d)",
				f.name, plies, f.truth, vOn, errOn, vOff, errOff)
		}
	}
	t.Logf("total error: heuristics on=%d off=%d", totalOn, totalOff)
	is.True(totalOn < totalOff)
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
