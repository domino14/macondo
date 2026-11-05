package montecarlo

import (
	"context"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
)

// Endgame leaves (lab knob): MACONDO_SIM_EGPLIES=N > 0 makes a rollout that
// empties the bag stop there and score the endgame with a quick N-ply
// result-only search (greedy playout at its leaves, the (-1, 1) root window
// the quick-endgame bots use) instead of playing it out with static moves.
// Once the bag is empty both racks are known, so a search is the right judge
// of who wins; static moves are not. MACONDO_SIM_EGCAPMS caps each solve
// (default 100 ms); a solve that runs out falls back to the win % table.
// Only a sim whose owner asks for it (SetEndgameLeaves: the bot's own sim,
// not the inference mini-sims) solves; MACONDO_SIM_EGMINI=1 lets every sim.
// MACONDO_SIM_EGMARGIN=M > 0 solves only endgames within M points, as the
// quick-endgame bots do; the others play out with static moves as before.
// MACONDO_SIM_EGMAXBAG=B > 0 solves only in sims of positions with at most B
// tiles in the bag.
var (
	egLeafOnce   sync.Once
	egLeafPlies  int
	egLeafMini   bool
	egLeafMarg   int
	egLeafMaxBag int
	egLeafCap    time.Duration
	egTableOnce  sync.Once
	// egLeafTable is one shared, lock-free transposition table for every
	// leaf solve in the process, as the quick-endgame bots share theirs.
	egLeafTable = &negamax.TranspositionTable{}
)

// EndgameLeafSolves, EndgameLeafFallbacks and EndgameLeafNanos count the
// endgame leaves solved, the ones that fell back to static play, and the time
// spent solving, for the lab tools' logs.
var EndgameLeafSolves, EndgameLeafFallbacks, EndgameLeafNanos atomic.Int64

func endgameLeafPlies() int {
	egLeafOnce.Do(func() {
		egLeafPlies, _ = strconv.Atoi(os.Getenv("MACONDO_SIM_EGPLIES"))
		egLeafMini = os.Getenv("MACONDO_SIM_EGMINI") == "1"
		egLeafMarg, _ = strconv.Atoi(os.Getenv("MACONDO_SIM_EGMARGIN"))
		egLeafMaxBag, _ = strconv.Atoi(os.Getenv("MACONDO_SIM_EGMAXBAG"))
		ms, _ := strconv.Atoi(os.Getenv("MACONDO_SIM_EGCAPMS"))
		if ms <= 0 {
			ms = 100
		}
		egLeafCap = time.Duration(ms) * time.Millisecond
	})
	return egLeafPlies
}

// SetEndgameLeaves marks the sim as one that solves the endgames its
// rollouts reach, when MACONDO_SIM_EGPLIES is set.
func (s *Simmer) SetEndgameLeaves(on bool) {
	s.endgameLeaves = on
}

// prepareEndgameLeaves sizes each thread's state stack for a solve on top of
// the rollout's one backed-up play and gives each thread a solver on its game.
func (s *Simmer) prepareEndgameLeaves() error {
	plies := endgameLeafPlies()
	if plies <= 0 || !(s.endgameLeaves || egLeafMini) ||
		(egLeafMaxBag > 0 && s.origGame.Bag().TilesRemaining() > egLeafMaxBag) {
		s.egSolvers = nil
		return nil
	}
	egTableOnce.Do(func() {
		egLeafTable.Reset(0.02, s.origGame.Board().Dim())
	})
	// One pruned word graph for the whole sim, as the pre-endgame builds:
	// every rollout deals from the same pool (our rack, the opponent's rack
	// and the bag), and the pruning draws on that union only.
	if s.egKWG != nil {
		s.egKWG.Close()
		s.egKWG = nil
	}
	if os.Getenv("MACONDO_SIM_EGNOPRUNE") != "1" {
		if pos, err := s.origGame.MagpiePosition(); err == nil {
			s.egKWG = pos.PrunedKWG()
		}
	}
	if len(s.egSolvers) != len(s.gameCopies) {
		// Solvers keep their search arenas, so they outlive a sim.
		s.egSolvers = make([]*negamax.Solver, len(s.gameCopies))
	}
	for i, g := range s.gameCopies {
		g.SetStateStackLength(1 + plies + negamax.MaxGreedyPlayoutPlies + 5)
		es := s.egSolvers[i]
		if es == nil {
			es = &negamax.Solver{}
		}
		if err := es.Init(g); err != nil {
			return err
		}
		es.SetTTable(egLeafTable)
		es.SetThreads(1)
		es.SetFirstWinOptim(true)
		es.SetSkipMaterialize(true)
		if s.egKWG != nil {
			es.SetPrunedKWG(s.egKWG)
		} else {
			// A pruned graph per solve costs more than a 2-ply search saves.
			es.SetPrunedKWGOptim(false)
		}
		s.egSolvers[i] = es
	}
	return nil
}

// endgameLeafClose reports whether the empty-bag position g is close enough
// to solve (see MACONDO_SIM_EGMARGIN).
func (s *Simmer) endgameLeafClose(g *game.Game) bool {
	if egLeafMarg <= 0 {
		return true
	}
	sp := g.CurrentSpread()
	return sp <= egLeafMarg && sp >= -egLeafMarg
}

// endgameLeaf scores the empty-bag position g for the sim's player: 1 for a
// win, 0 for a loss, 0.5 for a tie, with the sim's spread there. A solve that
// fails or runs out of time scores the position from the win % table instead;
// it may leave g mid-search, which the rollout's reset puts right.
func (s *Simmer) endgameLeaf(g *game.Game, thread int) (winProb float64, spread int) {
	onTurn := g.PlayerOnTurn()
	spread = g.SpreadFor(s.initialPlayer)
	fallback := func() (float64, int) {
		EndgameLeafFallbacks.Add(1)
		return winProbability(spread, 0, false, s.winPcts, 0,
			int(g.RackFor(onTurn).NumTiles()), int(g.RackFor(1-onTurn).NumTiles()),
			onTurn != s.initialPlayer), spread
	}
	if g.RackFor(0).NumTiles() == 0 || g.RackFor(1).NumTiles() == 0 {
		return fallback()
	}
	onTurnSpread := g.CurrentSpread()
	onRack, oppRack := int(g.RackFor(onTurn).NumTiles()), int(g.RackFor(1-onTurn).NumTiles())
	ctx, cancel := context.WithTimeout(context.Background(), egLeafCap)
	defer cancel()
	g.SetBackupMode(game.SimulationMode)
	g.SetEndgameMode(true)
	t0 := time.Now()
	val, _, err := s.egSolvers[thread].QuickAndDirtySolve(ctx, endgameLeafPlies(), thread)
	EndgameLeafNanos.Add(int64(time.Since(t0)))
	g.SetEndgameMode(false)
	g.SetBackupMode(game.NoBackup)
	if s.egKWG != nil {
		// The solver points the game at the pruned graph and leaves it
		// there; take it off so nothing outlives the graph.
		if pos, err := g.MagpiePosition(); err == nil {
			pos.SetOverrideKWG(nil)
		}
	}
	if err != nil || ctx.Err() != nil {
		EndgameLeafFallbacks.Add(1)
		return winProbability(spread, 0, false, s.winPcts, 0, onRack, oppRack,
			onTurn != s.initialPlayer), spread
	}
	EndgameLeafSolves.Add(1)
	final := int(val) + onTurnSpread
	switch {
	case final > 0:
		winProb = 1
	case final < 0:
		winProb = 0
	default:
		winProb = 0.5
	}
	if onTurn != s.initialPlayer {
		winProb = 1 - winProb
	}
	return winProb, spread
}
