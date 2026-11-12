package montecarlo

import (
	"context"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
)

// Endgame leaves: in a sim of a position with at most EndgameLeafMaxBag tiles
// in the bag, a rollout that empties the bag stops there and scores the
// endgame with a quick result-only search instead of playing it out with
// static moves. Once the bag is empty both racks are known, so a search is
// the right judge of who wins; static moves get the winner wrong in about one
// close endgame in twelve. Only endgames within EndgameLeafMargin points are
// searched (the others are decided anyway); a search that runs past
// EndgameLeafCap leaves the rollout to play on with static moves, as before.
//
// With one or two tiles in the bag every rollout reaches the endgame within a
// ply or two, and the opponent holds one of at most 36 racks, so a sim meets
// the same few endgames over and over: each is searched once and its result
// kept for the rest of the sim.
const (
	EndgameLeafMaxBag = 2
	EndgameLeafPlies  = 2
	EndgameLeafMargin = 60
	EndgameLeafCap    = 100 * time.Millisecond
)

var (
	egTableOnce sync.Once
	// egLeafTable is one shared, lock-free transposition table for every leaf
	// search in the process, as the quick-endgame bots share theirs.
	egLeafTable = &negamax.TranspositionTable{}
)

// EndgameLeafSolves, EndgameLeafFallbacks and EndgameLeafCacheHits count the
// endgame leaves searched, the ones left to static play instead (a search
// that ran out of time) and the ones answered from the sim's cache.
var EndgameLeafSolves, EndgameLeafFallbacks, EndgameLeafCacheHits atomic.Int64

// egLeafKey identifies an endgame leaf: the position's hash (board, both
// racks, scoreless turns) and the spread for the player on turn, which the
// result-only search's answer depends on.
type egLeafKey struct {
	hash   uint64
	spread int
}

// egLeafFailed marks a cached endgame whose search ran out of time, so the
// rollouts that reach it again play on without searching.
const egLeafFailed = math.MinInt32

// SetEndgameLeaves turns endgame-leaf search on or off for this sim (see
// EndgameLeafMaxBag). It is on unless turned off.
func (s *Simmer) SetEndgameLeaves(on bool) {
	s.noEndgameLeaves = !on
}

// EndgameLeaves reports whether this sim searches the endgames its rollouts
// reach, when the bag is low enough.
func (s *Simmer) EndgameLeaves() bool {
	return !s.noEndgameLeaves
}

// prepareEndgameLeaves gives each thread a solver on its game copy when the
// position qualifies, and sizes the copy's state stack for a search on top of
// the rollout's one backed-up play.
func (s *Simmer) prepareEndgameLeaves() error {
	s.egSolvers = nil
	s.egCache = sync.Map{}
	if s.noEndgameLeaves || s.origGame.Bag().TilesRemaining() > EndgameLeafMaxBag ||
		game.IsWordSmog(s.origGame.Rules().Variant()) {
		return nil
	}
	egTableOnce.Do(func() {
		egLeafTable.Reset(0.02, s.origGame.Board().Dim())
	})
	solvers := make([]*negamax.Solver, len(s.gameCopies))
	for i, g := range s.gameCopies {
		g.SetStateStackLength(1 + EndgameLeafPlies + negamax.MaxGreedyPlayoutPlies + 5)
		es := &negamax.Solver{}
		if err := es.Init(g); err != nil {
			return err
		}
		es.SetTTable(egLeafTable)
		es.SetThreads(1)
		es.SetFirstWinOptim(true)
		es.SetSkipMaterialize(true)
		// A pruned word graph per search costs more than a 2-ply search
		// saves with it.
		es.SetPrunedKWGOptim(false)
		solvers[i] = es
	}
	s.egSolvers = solvers
	return nil
}

// endgameLeafClose reports whether the empty-bag position g is close enough
// to search.
func endgameLeafClose(g *game.Game) bool {
	sp := g.CurrentSpread()
	return sp <= EndgameLeafMargin && sp >= -EndgameLeafMargin
}

// endgameLeaf scores the empty-bag position g for the sim's player: 1 for a
// win, 0 for a loss, 0.5 for a tie. ok is false when the search failed or ran
// out of time; the rollout then plays on with static moves. A search leaves g
// as it found it.
func (s *Simmer) endgameLeaf(g *game.Game, thread int) (winProb float64, ok bool) {
	onTurn := g.PlayerOnTurn()
	if g.RackFor(onTurn).NumTiles() == 0 || g.RackFor(1-onTurn).NumTiles() == 0 {
		return 0, false
	}
	onTurnSpread := g.CurrentSpread()
	key := egLeafKey{spread: onTurnSpread, hash: egLeafTable.Zobrist().Hash(g.Board().GetSquares(),
		g.RackFor(onTurn), g.RackFor(1-onTurn), false, g.ScorelessTurns())}
	if v, hit := s.egCache.Load(key); hit {
		EndgameLeafCacheHits.Add(1)
		if v.(int) == egLeafFailed {
			// This endgame ran out of time before; it would again.
			EndgameLeafFallbacks.Add(1)
			return 0, false
		}
		return s.egLeafResult(v.(int), onTurn), true
	}
	ctx, cancel := context.WithTimeout(context.Background(), EndgameLeafCap)
	defer cancel()
	g.SetBackupMode(game.SimulationMode)
	g.SetEndgameMode(true)
	val, _, err := s.egSolvers[thread].QuickAndDirtySolve(ctx, EndgameLeafPlies, thread)
	g.SetEndgameMode(false)
	g.SetBackupMode(game.NoBackup)
	if err != nil || ctx.Err() != nil {
		s.egCache.Store(key, egLeafFailed)
		EndgameLeafFallbacks.Add(1)
		return 0, false
	}
	EndgameLeafSolves.Add(1)
	final := int(val) + onTurnSpread
	s.egCache.Store(key, final)
	return s.egLeafResult(final, onTurn), true
}

// egLeafResult scores a searched endgame's final spread for the player on
// turn as a win, loss or tie for the sim's player.
func (s *Simmer) egLeafResult(final, onTurn int) float64 {
	winProb := 0.5
	switch {
	case final > 0:
		winProb = 1
	case final < 0:
		winProb = 0
	}
	if onTurn != s.initialPlayer {
		winProb = 1 - winProb
	}
	return winProb
}
