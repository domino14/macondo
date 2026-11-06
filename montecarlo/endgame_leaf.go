package montecarlo

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
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

// EndgameLeafSolves, EndgameLeafFallbacks, EndgameLeafNanos and
// EndgameLeafCacheHits count the endgame leaves solved, the ones that fell
// back to static play, the time spent solving and the leaves answered from
// the sim's cache, for the lab tools' logs.
var EndgameLeafSolves, EndgameLeafFallbacks, EndgameLeafNanos, EndgameLeafCacheHits atomic.Int64

// MACONDO_SIM_EGCHECK=1 solves cache hits anyway and counts in
// EndgameLeafCacheMismatches the ones whose result differs from the cache's.
var EndgameLeafCacheMismatches atomic.Int64
var egLeafCheck = os.Getenv("MACONDO_SIM_EGCHECK") == "1"

// egLeafKey identifies an endgame leaf: the position's hash (board, both
// racks, scoreless turns) and the spread for the player on turn, which the
// result-only search's answer depends on. Late in the game a sim's rollouts
// reach the same few endgames over and over (with two tiles in the bag the
// opponent holds one of at most 36 racks, and static rollouts are
// deterministic), so each sim keeps the results it has solved.
type egLeafKey struct {
	hash   uint64
	spread int
}

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
	s.egCache = sync.Map{}
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
func (s *Simmer) endgameLeaf(g *game.Game, thread, pliesLeft int) (winProb float64, spread int) {
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
	key := egLeafKey{spread: onTurnSpread, hash: egLeafTable.Zobrist().Hash(g.Board().GetSquares(),
		g.RackFor(onTurn), g.RackFor(1-onTurn), false, g.ScorelessTurns())}
	cached, hit := 0, false
	if v, ok := s.egCache.Load(key); ok {
		EndgameLeafCacheHits.Add(1)
		cached, hit = v.(int), true
		if !egLeafCheck {
			return s.egLeafResult(cached, onTurn), spread
		}
	}
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
	if hit && s.egLeafResult(final, onTurn) != s.egLeafResult(cached, onTurn) {
		EndgameLeafCacheMismatches.Add(1)
	}
	s.egCache.Store(key, final)
	if egAudit > 0 && egAuditN.Add(1)%int64(egAudit) == 0 {
		s.auditLeaf(g, thread, pliesLeft, onTurn, onTurnSpread, final)
	}
	return s.egLeafResult(final, onTurn), spread
}

// MACONDO_SIM_EGAUDIT=N (lab check): every Nth solved leaf also gets a deeper
// solve (8 plies, 3 s) and the static playout the plain sim would have used,
// both to the end of the game and cut at the sim's remaining plies, and logs
// all of them to stderr as the sim's player's results. It leaves g mid-game;
// the rollout's reset puts it right.
var (
	egAudit, _ = strconv.Atoi(os.Getenv("MACONDO_SIM_EGAUDIT"))
	egAuditN   atomic.Int64
	egAuditMu  sync.Mutex
)

func (s *Simmer) auditLeaf(g *game.Game, thread, pliesLeft, onTurn, onTurnSpread, quick int) {
	rec := map[string]any{"onRack": g.RackFor(onTurn).NumTiles(), "oppRack": g.RackFor(1 - onTurn).NumTiles(),
		"spread": onTurnSpread, "pliesLeft": pliesLeft, "quick": s.egLeafResult(quick, onTurn),
		"sim": fmt.Sprintf("%p", s), "rootBag": s.origGame.Bag().TilesRemaining(),
		"leafOnTurnIsSimmer": onTurn == s.initialPlayer}
	// A full-window solve: the exact final spread and its first move.
	es := s.egSolvers[thread]
	es.SetFirstWinOptim(false)
	es.SetSkipMaterialize(false)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	g.SetBackupMode(game.SimulationMode)
	g.SetEndgameMode(true)
	val, seq, err := es.QuickAndDirtySolve(ctx, 8, thread)
	g.SetEndgameMode(false)
	g.SetBackupMode(game.NoBackup)
	es.SetFirstWinOptim(true)
	es.SetSkipMaterialize(true)
	rec["deepDone"] = err == nil && ctx.Err() == nil
	cancel()
	if err == nil && ctx.Err() == nil {
		exact := int(val) + onTurnSpread
		rec["deep"] = s.egLeafResult(exact, onTurn)
		rec["exact"] = exact // on-turn player's final spread
		if len(seq) > 0 {
			rec["solverFirst"] = seq[0].ShortDescription()
		}
	}
	if s.egKWG != nil {
		if pos, err := g.MagpiePosition(); err == nil {
			pos.SetOverrideKWG(nil)
		}
	}
	for ply := 0; ply < 30 && g.Playing() == pb.PlayState_PLAYING; ply++ {
		if ply == pliesLeft {
			p := g.PlayerOnTurn()
			rec["plain"] = winProbability(g.SpreadFor(s.initialPlayer), 0, false, s.winPcts, 0,
				int(g.RackFor(p).NumTiles()), int(g.RackFor(1-p).NumTiles()), p != s.initialPlayer)
		}
		m := s.bestStaticTurn(g.PlayerOnTurn(), thread)
		if ply == 0 {
			rec["staticFirst"] = m.ShortDescription()
		}
		g.PlayMove(m, false, 0)
	}
	rec["staticFinal"] = g.SpreadFor(onTurn)
	sp := g.SpreadFor(s.initialPlayer)
	end := 0.5
	if sp > 0 {
		end = 1
	} else if sp < 0 {
		end = 0
	}
	rec["static"] = end
	if _, ok := rec["plain"]; !ok {
		rec["plain"] = end
	}
	b, _ := json.Marshal(rec)
	egAuditMu.Lock()
	fmt.Fprintf(os.Stderr, "EGAUDIT %s\n", b)
	egAuditMu.Unlock()
}

// egLeafResult scores a solved endgame's final spread for the player on
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
