package negamax

import (
	"context"
	"sync"
	"sync/atomic"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/tinymove"
)

// PV-splitting, after Marsland and Campbell, "Parallel Search of Strongly
// Ordered Game Trees" (ACM Computing Surveys 14(4), 1982), figure 14.
//
// The main thread walks down the principal variation. At each of its nodes it
// first searches the first move, alone, to get a value; only then are the
// node's other moves searched, one thread to a move, each with a null window
// at that value. So every thread works with the best bound there is, and the
// threads are spent on the part of the tree that has to be searched anyway:
// with good ordering the siblings of a PV move are refuted, and their subtrees
// are all needed for that.
//
// A sibling that beats the value is not searched again by the thread that
// found it. It goes back to the main thread, which searches it as a new
// principal variation, splitting again along it.
//
// The paper's threads wait once a node's moves are all taken. Here a thread
// with no move left joins the search of one still in progress: the two share
// the transposition table and, as in ABDADA, each puts off the moves the other
// is searching. Whichever finishes first ends the search for both.

// pvSplitMinDepth is the smallest depth at which a node's moves are handed to
// the helpers. Below it a subtree is cheaper than the handoff.
const pvSplitMinDepth = 2

// pvSplitHelp is whether a thread with no move left joins another's search.
var pvSplitHelp = true

type pvSplitState struct {
	jobs []chan *splitPoint
	done sync.WaitGroup
	// path is the main thread's line from the root to the node it is at.
	path []tinymove.SmallMove
}

// splitPoint is one round of null-window searches at a PV node.
type splitPoint struct {
	path     []tinymove.SmallMove
	nodeKey  uint64
	depth    int
	children []tinymove.SmallMove
	// pending holds the indices in children to search this round.
	pending []int
	// alpha is the best lower bound on the node's value. A move that fails
	// high raises it for the moves still being searched. Written under mu.
	alpha  atomic.Int32
	beta   int16
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu sync.Mutex
	// next is the position in pending of the first move nobody has taken.
	next int
	// moves is indexed like pending.
	moves []splitMove
	// vals[i] bounds child i's value: from below if failHigh[i], else from above.
	vals     []int16
	failHigh []bool
	impure   bool
	maxReach uint64
	err      error
}

// splitMove is the state of one move being searched at a split point.
type splitMove struct {
	// ctx ends when the move has a result or the split point is abandoned.
	ctx       context.Context
	cancel    context.CancelFunc
	searchers int
	done      bool
}

// claim picks a move for a thread to search: the next one nobody has taken
// or, when there is none, the unfinished one with the fewest threads on it.
// It returns the move's position in pending, or -1 if there is nothing to do.
func (sp *splitPoint) claim() int {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	n := -1
	if sp.next < len(sp.pending) {
		n = sp.next
		sp.next++
		sp.moves[n].ctx, sp.moves[n].cancel = context.WithCancel(sp.ctx)
	} else if pvSplitHelp {
		for i := range sp.moves {
			if !sp.moves[i].done && (n < 0 || sp.moves[i].searchers < sp.moves[n].searchers) {
				n = i
			}
		}
	}
	if n >= 0 {
		sp.moves[n].searchers++
	}
	return n
}

// finish records the result of searching a move with a null window at a,
// unless another thread got there first. It returns false if the move has to
// be searched again: it beat a, but another move has raised the bound past
// its value since.
func (sp *splitPoint) finish(n int, v, a int16, impure bool, reach uint64) bool {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	mv := &sp.moves[n]
	if mv.done {
		return true
	}
	failHigh := v > a
	if failHigh && v < sp.beta {
		if int32(v) <= sp.alpha.Load() {
			return false
		}
		sp.alpha.Store(int32(v))
	}
	mv.done = true
	mv.cancel()
	idx := sp.pending[n]
	sp.vals[idx] = v
	sp.failHigh[idx] = failHigh
	if impure {
		sp.impure = true
	}
	sp.maxReach = max(sp.maxReach, reach+1)
	if v >= sp.beta {
		// A cutoff: the node's other moves no longer matter.
		sp.cancel()
	}
	return true
}

// leave takes a thread off a move another thread finished first.
func (sp *splitPoint) leave(n int) {
	sp.mu.Lock()
	sp.moves[n].searchers--
	sp.mu.Unlock()
}

func (st *pvSplitState) start(s *Solver) {
	st.jobs = make([]chan *splitPoint, s.threads-1)
	for t := 1; t < s.threads; t++ {
		ch := make(chan *splitPoint, 1)
		st.jobs[t-1] = ch
		st.done.Add(1)
		go func() {
			defer st.done.Done()
			s.pvSplitHelper(t, ch)
		}()
	}
}

func (st *pvSplitState) stop() {
	for _, ch := range st.jobs {
		close(ch)
	}
	st.done.Wait()
}

// pvSplitHelper brings its game to each split point it is sent and searches
// moves there until none are left.
func (s *Solver) pvSplitHelper(thread int, jobs <-chan *splitPoint) {
	g := s.gameCopies[thread-1]
	// The moves this thread's game has played from the root. Split points
	// follow one another down and back up a line, so most of it is kept.
	var cur []tinymove.SmallMove
	for sp := range jobs {
		common := 0
		for common < len(cur) && common < len(sp.path) &&
			cur[common].TinyMove() == sp.path[common].TinyMove() {
			common++
		}
		for len(cur) > common {
			g.UnplayLastMove()
			cur = cur[:len(cur)-1]
		}
		ok := true
		for i := common; i < len(sp.path); i++ {
			m := sp.path[i]
			if _, err := g.PlaySmallMove(&m); err != nil {
				sp.fail(err)
				ok = false
				break
			}
			cur = append(cur, m)
		}
		if ok {
			s.pvSplitWork(sp, thread)
		}
		sp.wg.Done()
	}
	for range cur {
		g.UnplayLastMove()
	}
}

func (sp *splitPoint) fail(err error) {
	sp.mu.Lock()
	if sp.err == nil {
		sp.err = err
	}
	sp.mu.Unlock()
	sp.cancel()
}

// pvSplitWork takes moves from the split point and searches each with a null
// window at the shared bound. The thread's game must be at the split point.
func (s *Solver) pvSplitWork(sp *splitPoint, thread int) {
	g := s.game
	if thread > 0 {
		g = s.gameCopies[thread-1]
	}
	onTurn := g.PlayerOnTurn()
	stmRack := g.RackFor(onTurn)
	childPV := PVLine{g: g}
	counter := &s.impureEvents[thread]

	for sp.ctx.Err() == nil {
		n := sp.claim()
		if n < 0 {
			return
		}
		ctx := sp.moves[n].ctx
		child := sp.children[sp.pending[n]]
		moveTiles, err := g.PlaySmallMove(&child)
		if err != nil {
			sp.fail(err)
			return
		}
		s.nodes.Add(1)
		childKey := s.ttable.Zobrist().AddMove(sp.nodeKey, &child, stmRack, moveTiles,
			onTurn == s.solvingPlayer, g.ScorelessTurns(), g.LastScorelessTurns())

		impureBefore := counter.n
		reach := uint64(0)
		for {
			a := int16(sp.alpha.Load())
			var val int16
			val, err = s.negamax(ctx, childKey, sp.depth-1, -a-1, -a, &childPV, thread, false)
			if err != nil {
				break
			}
			reach = max(reach, counter.reach)
			if sp.finish(n, -val, a, counter.n != impureBefore, reach) {
				break
			}
		}
		g.UnplayLastMove()
		if err != nil {
			if sp.ctx.Err() != nil {
				return
			}
			if ctx.Err() != nil {
				// Another thread finished this move.
				sp.leave(n)
				continue
			}
			sp.fail(err)
			return
		}
	}
}

// pvSplit is negamax for a node of the principal variation, run by the main
// thread. It returns what negamax would.
func (s *Solver) pvSplit(ctx context.Context, nodeKey uint64, depth int, α, β int16, pv *PVLine) (int16, error) {
	g := s.game
	if depth < pvSplitMinDepth || g.Playing() != pb.PlayState_PLAYING {
		return s.negamax(ctx, nodeKey, depth, α, β, pv, 0, true)
	}
	if ctx.Err() != nil {
		return 0, ctx.Err()
	}
	st := s.pvs
	onTurn := g.PlayerOnTurn()
	ourSpread := g.SpreadFor(onTurn)
	stmRack := g.RackFor(onTurn)
	isRoot := depth == s.currentIDDepths[0]

	alphaOrig := α
	impure := &s.impureEvents[0].n
	impureAtEntry := *impure
	reach := &s.impureEvents[0].reach
	maxReach := uint64(0)

	// The table, as in negamax at a PV node: an entry never ends the search
	// unless it closes the window, but it can narrow it.
	ttMove := tinymove.InvalidTinyMove
	if ttEntry := s.ttable.lookup(nodeKey); ttEntry.valid() {
		// Whatever depth the entry is from, its move is the best guess at this
		// node's: the last iteration's principal variation is found this way.
		ttMove = ttEntry.getPlay()
		entryDepth := ttEntry.depth()
		if entryDepth == uint8(depth) || (ttEntry.pure() && entryDepth <= uint8(depth)) {
			if ttEntry.pure() {
				maxReach = uint64(entryDepth)
			} else {
				*impure++
			}
			score := ttEntry.getScore() + int16(ourSpread)
			switch ttEntry.flag() {
			case TTLower:
				if !isRoot {
					α = max(α, score)
				} else if score > α && score < β {
					α = score
				}
			case TTUpper:
				if !isRoot {
					β = min(β, score)
				} else if score > α && score < β {
					β = score
				}
			}
			if α >= β {
				*reach = maxReach
				return score, nil
			}
		}
	}

	var children []tinymove.SmallMove
	if isRoot {
		children = s.initialMoves[0]
	} else {
		var fromArena bool
		children, fromArena = s.generateSTMPlays(depth, 0)
		if fromArena {
			defer s.arenas[0].Dealloc(len(children))
		}
		s.assignEstimates(children, depth, 0, ttMove, nodeKey)
	}

	bestValue := -HugeNumber
	var bestMove tinymove.SmallMove
	childPV := PVLine{g: g}

	// searchPV searches child idx as a principal variation, splitting along it.
	searchPV := func(idx int) (int16, error) {
		childPV.Clear()
		moveTiles, err := g.PlaySmallMove(&children[idx])
		if err != nil {
			return 0, err
		}
		s.nodes.Add(1)
		childKey := s.ttable.Zobrist().AddMove(nodeKey, &children[idx], stmRack, moveTiles,
			onTurn == s.solvingPlayer, g.ScorelessTurns(), g.LastScorelessTurns())
		st.path = append(st.path, children[idx])
		value, err := s.pvSplit(ctx, childKey, depth-1, -β, -α, &childPV)
		st.path = st.path[:len(st.path)-1]
		g.UnplayLastMove()
		if err != nil {
			return 0, err
		}
		maxReach = max(maxReach, *reach+1)
		value = -value
		if value > bestValue {
			bestValue = value
			bestMove = children[idx]
			pv.Update(bestMove, &childPV, bestValue-int16(s.initialSpread))
		}
		if isRoot {
			children[idx].SetEstimatedValue(value)
		}
		α = max(α, bestValue)
		return value, nil
	}

	if _, err := searchPV(0); err != nil {
		return 0, err
	}

	pending := make([]int, 0, len(children)-1)
	for i := 1; i < len(children); i++ {
		pending = append(pending, i)
	}
	vals := make([]int16, len(children))
	failHigh := make([]bool, len(children))

	for bestValue < β && len(pending) > 0 {
		sp := &splitPoint{
			path:     st.path,
			nodeKey:  nodeKey,
			depth:    depth,
			children: children,
			pending:  pending,
			moves:    make([]splitMove, len(pending)),
			beta:     β,
			vals:     vals,
			failHigh: failHigh,
		}
		sp.alpha.Store(int32(α))
		sp.ctx, sp.cancel = context.WithCancel(ctx)
		sp.wg.Add(len(st.jobs))
		for _, ch := range st.jobs {
			ch <- sp
		}
		s.pvSplitWork(sp, 0)
		sp.wg.Wait()
		sp.cancel()
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		if sp.err != nil {
			return 0, sp.err
		}
		if sp.impure {
			*impure++
		}
		maxReach = max(maxReach, sp.maxReach)

		// The move with the highest lower bound is the one to search as the
		// new principal variation. One at or past β ends the node.
		cand := -1
		for _, idx := range pending {
			if failHigh[idx] && (cand < 0 || vals[idx] > vals[cand]) {
				cand = idx
			}
		}
		if cand >= 0 && vals[cand] >= β {
			bestValue = vals[cand]
			bestMove = children[cand]
			childPV.Clear()
			pv.Update(bestMove, &childPV, bestValue-int16(s.initialSpread))
			if isRoot {
				children[cand].SetEstimatedValue(bestValue)
			}
			break
		}
		if cand < 0 {
			// Every move failed low against α, which did not move: their
			// values are upper bounds, as a sequential search would have them.
			for _, idx := range pending {
				if vals[idx] > bestValue {
					bestValue = vals[idx]
					bestMove = children[idx]
					childPV.Clear()
					pv.Update(bestMove, &childPV, bestValue-int16(s.initialSpread))
				}
				if isRoot {
					children[idx].SetEstimatedValue(vals[idx])
				}
			}
			break
		}
		if _, err := searchPV(cand); err != nil {
			return 0, err
		}
		// Left to search against the new α: the other moves that failed high,
		// which are only known to beat an older bound, and any move whose
		// upper bound is above α. There are none of the latter unless the
		// new line came back lower than the bound its null-window search gave.
		next := pending[:0]
		for _, idx := range pending {
			if idx == cand {
				continue
			}
			if failHigh[idx] || vals[idx] > α {
				next = append(next, idx)
			} else if isRoot {
				children[idx].SetEstimatedValue(vals[idx])
			}
		}
		pending = next
	}

	score := bestValue - int16(ourSpread)
	var flag uint8
	if bestValue <= alphaOrig {
		flag = TTUpper
	} else if bestValue >= β {
		flag = TTLower
	} else {
		flag = TTExact
	}
	pure := *impure == impureAtEntry
	storedDepth := uint8(min(depth, depthMask))
	if pure {
		storedDepth = uint8(maxReach)
	}
	s.ttable.storeEntry(nodeKey, score, flag<<6+storedDepth, bestMove.TinyMove(), pure)
	*reach = maxReach
	return bestValue, nil
}

func (s *Solver) iterativelyDeepenPVSplit(ctx context.Context, plies int) error {
	st := &pvSplitState{}
	s.pvs = st
	started := false
	// Threads that search the same move put off each other's moves the way
	// ABDADA's do.
	if s.abdadaTable == nil {
		s.abdadaTable = NewABDADATable()
	}
	s.abdadaOptim = pvSplitHelp
	defer func() {
		s.abdadaOptim = false
		if started {
			st.stop()
		}
	}()
	return s.iterativelyDeepenParallel(ctx, plies, "pvsplit",
		func(ctx context.Context, rootKey uint64, p int, α, β int16, pv *PVLine) (int16, error) {
			if !started {
				// The helpers' games are made by the driver, so they can
				// only start once it has run this far.
				st.start(s)
				started = true
			}
			st.path = st.path[:0]
			return s.pvSplit(ctx, rootKey, p, α, β, pv)
		})
}
