package preendgame

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
	"gonum.org/v1/gonum/stat/combin"
	"gopkg.in/yaml.v3"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/tinymove/conversions"
	"github.com/domino14/macondo/zobrist"
)

// nestedCacheKey uniquely identifies an information state at which
// nestedOurTurnSolve is entered. The result is a pure function of this state
// (given fixed solver config), so we can memoize it.
type nestedCacheKey struct {
	board          uint64                               // board-only Zobrist hash
	ourRack        [zobrist.MaxLetters]uint8            // Rack.LetArr narrowed to uint8
	unseen         [tilemapping.MaxAlphabetSize]uint8   // bag.Peek() ∪ opp.rack multiset
	scorelessTurns uint8                                // 0 or 1 in practice (2 ends the game)
}

// nestedCache is a shared, thread-safe memo table for nestedOurTurnSolve
// results, keyed on info-state. The value is a per-bag-signature verdict map:
// for each possible ordered bag content at the call site, what verdict would
// nestedOurTurnSolve return? This allows callers with the same info-state but
// different actual bag contents to share a single solve. It is reset at the
// start of each iterative-deepening ply.
type nestedCache struct {
	mu sync.RWMutex
	m  map[nestedCacheKey]map[string]PEGOutcome
}

func (c *nestedCache) reset() {
	c.mu.Lock()
	c.m = make(map[nestedCacheKey]map[string]PEGOutcome)
	c.mu.Unlock()
}

// lookup returns the per-bag verdict map for this info-state. The caller
// indexes into it by bagSig(currentBag) to get the specific verdict.
func (c *nestedCache) lookup(key nestedCacheKey) (map[string]PEGOutcome, bool) {
	c.mu.RLock()
	v, ok := c.m[key]
	c.mu.RUnlock()
	return v, ok
}

func (c *nestedCache) store(key nestedCacheKey, verdictMap map[string]PEGOutcome) {
	c.mu.Lock()
	c.m[key] = verdictMap
	c.mu.Unlock()
}

func (c *nestedCache) size() int {
	c.mu.RLock()
	n := len(c.m)
	c.mu.RUnlock()
	return n
}

// buildNestedCacheKey constructs the cache key from the current game state
// visible through the per-thread game copy. Must be called before any
// sub-permutation reshuffling inside nestedOurTurnSolve.
func (s *Solver) buildNestedCacheKey(g *game.Game) nestedCacheKey {
	var k nestedCacheKey
	k.board = s.ttable.Zobrist().BoardHash(g.Board().GetSquares())
	for i, ct := range g.RackFor(g.PlayerOnTurn()).LetArr {
		k.ourRack[i] = uint8(ct)
	}
	opp := 1 - g.PlayerOnTurn()
	for _, t := range g.Bag().Peek() {
		k.unseen[int(t)]++
	}
	for _, t := range g.RackFor(opp).TilesOn() {
		k.unseen[int(t)]++
	}
	k.scorelessTurns = uint8(g.ScorelessTurns())
	return k
}

// bagSigFromTiles encodes an ordered tile slice as a string map key.
func bagSigFromTiles(tiles []tilemapping.MachineLetter) string {
	b := make([]byte, len(tiles))
	for i, ml := range tiles {
		b[i] = byte(ml)
	}
	return string(b)
}

// bagSig returns a string key for the current ordered bag content.
func bagSig(g *game.Game) string {
	return bagSigFromTiles(g.Bag().Peek())
}

func (s *Solver) multithreadSolveGeneric(ctx context.Context, moves []*move.Move, logChan chan []byte) ([]*PreEndgamePlay, error) {
	// for every move, solve all the possible endgames.
	// - make play on board
	// - for tile in unseen:
	//   - if we've already seen this letter for this pre-endgame move
	//     increment its stats accordingly
	//   - overwrite letters on both racks accordingly
	//   - solve endgame from opp perspective
	//   - increment wins/losses accordingly for this move and letter
	// at the end sort stats by number of won endgames and then avg spread.

	s.plays = make([]*PreEndgamePlay, len(moves))
	for idx, play := range moves {
		s.plays[idx] = &PreEndgamePlay{Play: play}
	}
	maybeInBagTiles := make([]int, tilemapping.MaxAlphabetSize)
	for _, t := range s.game.RackFor(s.game.NextPlayer()).TilesOn() {
		maybeInBagTiles[t]++
	}
	for _, t := range s.game.Bag().Peek() {
		maybeInBagTiles[t]++
	}
	// If we have a partial or full opponent rack, these tiles cannot be in
	// the bag.
	for _, t := range s.knownOppRack {
		maybeInBagTiles[t]--
	}

	// When plays are fewer than threads, parallelize at the permutation level
	// so all threads stay busy. Pre-sort before workers start to avoid a race
	// on thread 0's game state. Otherwise use the cheaper per-play model.
	var sortedOpts [][]option
	if len(s.plays) < s.threads {
		sortedOpts = s.computeSortedOptions(maybeInBagTiles)
	}

	g := errgroup.Group{}
	winnerGroup := errgroup.Group{}
	// log.Debug().Interface("maybe-in-bag-tiles", maybeInBagTiles).Msg("unseen tiles")
	jobChan := make(chan job, s.threads)
	winnerChan := make(chan *PreEndgamePlay)

	var processed atomic.Uint32
	log.Info().Msgf("starting to process %d possible side-to-move plays", len(moves))
	for t := 0; t < s.threads; t++ {
		g.Go(func() error {
			for j := range jobChan {
				if err := s.handleJobGeneric(ctx, j, t, winnerChan); err != nil {
					log.Debug().AnErr("err", err).Msg("error-handling-job")
					// Don't exit, to avoid deadlock.
				}
				if s.logStream != nil {
					s.threadLogs[t].NestedCalls = int(s.threadNestedCalls[t])
					s.threadLogs[t].MaxNestedDepth = int(s.threadMaxNestedDepth[t])
					s.threadLogs[t].SubPermsEvaluated = int(s.threadSubPermsEvaluated[t])
					out, err := yaml.Marshal(s.threadLogs[t])
					if err != nil {
						log.Err(err).Msg("error-marshaling-logs")
					}
					logChan <- out
					logChan <- []byte("\n")
				}
				processed.Add(1)
				n := processed.Load()
				if n%100 == 0 {
					log.Info().Uint64("cutoffs", s.numCutoffs.Load()).Msgf("handled %d plays...", n)
				}
			}
			return nil
		})
	}

	tickerDone := make(chan struct{})
	go func() {
		t := time.NewTicker(60 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-tickerDone:
				return
			case <-ctx.Done():
				return
			case <-t.C:
				ch := s.nestedCacheHits.Load()
				cm := s.nestedCacheMisses.Load()
				cr := float64(0)
				if ch+cm > 0 {
					cr = float64(ch) / float64(ch+cm)
				}
				nestedByBagSize := make(map[int]uint64)
					for i := 1; i <= InBagMaxLimit; i++ {
						if n := s.numNestedByBagSize[i].Load(); n > 0 {
							nestedByBagSize[i] = n
						}
					}
					now := time.Now()
					s.inFlightMu.RLock()
					type permSnapshot struct {
						Play           string  `json:"play"`
						PermInBag      string  `json:"perm_in_bag"`
						OppRack        string  `json:"opp_rack"`
						OurRack        string  `json:"our_rack"`
						ElapsedS       float64 `json:"elapsed_s"`
						EndgamesOnPerm uint64  `json:"endgames_on_perm"`
						NestedBagSize  int32   `json:"nested_bag_size,omitempty"`
					}
					curEndgames := s.numEndgamesSolved.Load()
					inFlight := make([]permSnapshot, 0, s.threads)
					for t := 0; t < s.threads; t++ {
						p := s.inFlightPerms[t]
						if p.permInBag == "" {
							continue
						}
						threadNow := s.threadEndgamesSolved[t].Load()
						var permEndgames uint64
						if threadNow >= p.endgamesAtStart {
							permEndgames = threadNow - p.endgamesAtStart
						}
						inFlight = append(inFlight, permSnapshot{
							Play:           p.play,
							PermInBag:      p.permInBag,
							OppRack:        p.oppRack,
							OurRack:        p.ourRack,
							ElapsedS:       now.Sub(p.startedAt).Seconds(),
							EndgamesOnPerm: permEndgames,
							NestedBagSize:  s.threadNestedBagSize[t].Load(),
						})
					}
					s.inFlightMu.RUnlock()
					total := s.totalPerms.Load()
					done := processed.Load()
					remaining := uint32(0)
					if total > done {
						remaining = total - done
					}
					log.Info().
						Uint32("processed", done).
						Uint32("total", total).
						Uint32("remaining", remaining).
						Uint64("endgames-solved", curEndgames).
						Uint64("cutoffs", s.numCutoffs.Load()).
						Uint64("nested-calls", s.numNestedCalls.Load()).
						Uint64("max-nested-depth", s.maxNestedDepth.Load()).
						Uint64("sub-perms-evaluated", s.numSubPermsEvaluated.Load()).
						Uint64("nested-cache-hits", ch).
						Uint64("nested-cache-misses", cm).
						Float64("nested-cache-hit-rate", cr).
						Int("nested-cache-size", s.nestedCache.size()).
						Interface("nested-calls-by-bag-size", nestedByBagSize).
						Interface("in-flight", inFlight).
						Msg("peg-status")
			}
		}
	}()

	numCombos := combin.NumPermutations(s.numinbag+game.RackTileLimit,
		s.numinbag)

	// The determiner of the winner.
	winnerGroup.Go(func() error {
		for p := range winnerChan {
			if s.winnerSoFar != nil {
				if p.Points > s.winnerSoFar.Points {
					s.winnerSoFar = p
				}
			} else {
				s.winnerSoFar = p
			}
			// e.g. if we have three known losses in 4 games, we have at most 7 possible losses.
			ppotentialLosses := float32(numCombos) - p.Points
			s.potentialWinnerMutex.Lock()
			if ppotentialLosses < s.minPotentialLosses {
				log.Debug().
					Float32("potentialLosses", ppotentialLosses).
					Str("p", p.String()).
					Float32("minPotentialLosses", s.minPotentialLosses).Msg("new-fewest-potential-losses")
				s.minPotentialLosses = ppotentialLosses
			}
			s.potentialWinnerMutex.Unlock()
		}
		return nil
	})

	s.createGenericPEGJobs(ctx, maybeInBagTiles, sortedOpts, jobChan)
	err := g.Wait()
	close(tickerDone)
	if err != nil {
		return nil, err
	}

	// In per-perm mode, each worker processes one permutation of a shared pegPlay
	// and must not call finalize() (doing so mid-flight would prematurely lock in
	// a partial outcome). Finalize all plays here, after all workers have finished.
	if sortedOpts != nil {
		for _, p := range s.plays {
			p.finalize()
		}
	}

	close(winnerChan)
	winnerGroup.Wait()

	// sort plays by win %
	sort.Slice(s.plays, func(i, j int) bool {
		return s.plays[i].Points > s.plays[j].Points
	})

	if !s.skipTiebreaker {
		err = s.maybeTiebreak(ctx, maybeInBagTiles)
		if err != nil {
			log.Err(err).Msg("in-tiebreak")
			return nil, err
		}
	}

	if ctx.Err() != nil && (ctx.Err() == context.Canceled || ctx.Err() == context.DeadlineExceeded) {
		log.Info().Msg("timed out or stopped; returning best results so far...")
		err = ErrCanceledEarly
	}
	wh := s.nestedCacheHits.Load()
	wm := s.nestedCacheMisses.Load()
	wr := float64(0)
	if wh+wm > 0 {
		wr = float64(wh) / float64(wh+wm)
	}
	log.Info().Uint64("solved-endgames", s.numEndgamesSolved.Load()).
		Uint64("cutoff-moves", s.numCutoffs.Load()).
		Uint64("nested-calls", s.numNestedCalls.Load()).
		Uint64("max-nested-depth", s.maxNestedDepth.Load()).
		Uint64("sub-perms-evaluated", s.numSubPermsEvaluated.Load()).
		Uint64("nested-cache-hits", wh).
		Uint64("nested-cache-misses", wm).
		Float64("nested-cache-hit-rate", wr).
		Int("nested-cache-size", s.nestedCache.size()).
		Str("winner", s.plays[0].String()).Msg("winning-play")

	return s.plays, err
}

// computeSortedOptions generates all permutations for each play and sorts
// them by oppEstimate (highest first) so that the hardest permutations are
// processed first, surfacing losses early for the early-cutoff optimisation.
// Must be called before worker goroutines start so thread 0's game is not
// accessed concurrently.
func (s *Solver) computeSortedOptions(maybeInBagTiles []int) [][]option {
	permutations := generatePermutations(maybeInBagTiles, s.numinbag)
	g0 := s.endgameSolvers[0].Game()
	mg0 := s.endgameSolvers[0].Movegen()
	mg0.(*movegen.GordonGenerator).SetPlayRecorderTopPlay()
	snap := snapshotPEGState(g0)

	result := make([][]option, len(s.plays))
	for pi, p := range s.plays {
		opts := make([]option, len(permutations))
		for j, perm := range permutations {
			tiles := make([]tilemapping.MachineLetter, len(perm.Perm))
			for k, el := range perm.Perm {
				tiles[k] = tilemapping.MachineLetter(el)
			}
			opt := option{mls: tiles, ct: perm.Count}
			g0.ThrowRacksInFor(1 - g0.PlayerOnTurn())
			MoveTilesToBeginning(tiles, g0.Bag())
			if _, err := g0.SetRandomRack(1-g0.PlayerOnTurn(), nil); err == nil {
				if err := g0.PlayMove(p.Play, false, 0); err == nil {
					mg0.GenAll(g0.RackFor(g0.PlayerOnTurn()), false)
					if len(mg0.Plays()) > 0 {
						opt.oppEstimate = float64(mg0.Plays()[0].Equity())
					}
					g0.UnplayLastMove()
				}
			}
			opts[j] = opt
		}
		sort.Slice(opts, func(i, j int) bool {
			return opts[i].oppEstimate > opts[j].oppEstimate
		})
		for k := range opts {
			opts[k].idx = k
		}
		result[pi] = opts
	}

	snap.restore(g0)
	mg0.SetPlayRecorder(movegen.AllPlaysSmallRecorder)
	return result
}

func (s *Solver) createGenericPEGJobs(ctx context.Context, maybeInBagTiles []int, sortedOpts [][]option, jobChan chan job) {
	// Count jobs first so totalPerms is visible to the status ticker immediately.
	total := 0
	if sortedOpts != nil {
		for _, opts := range sortedOpts {
			total += len(opts)
		}
	} else {
		total = len(s.plays)
	}
	s.totalPerms.Store(uint32(total))
	log.Info().Int("numJobs", total).Msg("queued-jobs")

	if sortedOpts != nil {
		// Per-permutation mode: one job per (play, perm), pre-sorted hardest-first.
		for pi, p := range s.plays {
			for _, opt := range sortedOpts[pi] {
				jobChan <- job{ourMove: p, opt: opt}
			}
		}
	} else {
		// Per-play mode: one job per play; the job processes all permutations
		// internally, keeping oppEstimate computation parallelized across threads.
		for _, p := range s.plays {
			jobChan <- job{ourMove: p, maybeInBagTiles: maybeInBagTiles}
		}
	}
	close(jobChan)
}

func (s *Solver) maybeTiebreak(ctx context.Context, maybeInBagTiles []int) error {
	i := 0
	for {
		if i+1 >= len(s.plays) || s.plays[i].Points != s.plays[i+1].Points {
			break
		}
		i++
	}
	if i == 0 {
		log.Info().Str("winner", s.plays[0].String()).Msg("only one clear winner")
		return nil
	}

	numWinners := i + 1

	topPlayIdxs := []int{}
	// Only tiebreak plays that empty the bag. Spread tracking is broken
	// for non-bag-emptying plays: handleJobGeneric recursively explores
	// opponent responses and calls addSpreadStat at every endgame leaf,
	// summing spread across many leaves while TotalOutcomes stays fixed
	// at the inbagOption count. The result is a wildly inflated avgSpread
	// (see issue #476: a Pass play in the analyzer was reporting an
	// avgSpread of 27315 instead of values in the typical 60-70 range).
	for i := range numWinners {
		if s.plays[i].Play.TilesPlayed() >= s.numinbag {
			topPlayIdxs = append(topPlayIdxs, i)
		}
	}
	if len(topPlayIdxs) == 1 {
		log.Info().Str("winner", s.plays[topPlayIdxs[0]].String()).Msg("only one winner empties the bag")
		// Bring winner to the front.
		s.plays[topPlayIdxs[0]], s.plays[0] = s.plays[0], s.plays[topPlayIdxs[0]]
		return nil
	} else if len(topPlayIdxs) == 0 {
		log.Info().Str("winner", s.plays[0].String()).Msg("all winners do not empty the bag; will not tiebreak by spread")
		return nil
	} else if len(topPlayIdxs) != numWinners {
		log.Info().Int("ndiscarded", numWinners-len(topPlayIdxs)).
			Msg("non-bag-emptying plays are discarded for tiebreaks")
	}

	// There is more than one winning play.
	// Use total points scored as a first tie-breaker
	// (i.e. prior to solving endgames)
	sort.Slice(topPlayIdxs, func(i, j int) bool {
		return s.plays[topPlayIdxs[i]].Play.Score() > s.plays[topPlayIdxs[j]].Play.Score()
	})

	// Ensure avoid-prune plays are always included in tiebreak, even beyond
	// the limit. We only consider avoid-prune plays that empty the bag,
	// matching the constraint above; the spread for non-bag-emptying plays
	// is unreliable and would be excluded anyway.
	avoidPruneIdxs := []int{}
	for i := range numWinners {
		if s.shouldAvoidPrune(s.plays[i].Play) && s.plays[i].Play.TilesPlayed() >= s.numinbag {
			avoidPruneIdxs = append(avoidPruneIdxs, i)
		}
	}

	topN := min(len(topPlayIdxs), TieBreakerPlays)
	finalIdxs := topPlayIdxs[:topN]

	// Add any avoid-prune plays that aren't already in the top N
	for _, idx := range avoidPruneIdxs {
		found := false
		for _, topIdx := range finalIdxs {
			if idx == topIdx {
				found = true
				break
			}
		}
		if !found {
			finalIdxs = append(finalIdxs, idx)
		}
	}

	log.Info().Msgf("%d plays tied for first, taking top %d and tie-breaking (including %d avoid-prune plays)...",
		len(topPlayIdxs), len(finalIdxs), len(avoidPruneIdxs))
	topPlayIdxs = finalIdxs

	// We want to solve these endgames fully (to get an accurate spread)
	for _, es := range s.endgameSolvers {
		es.SetFirstWinOptim(false)
	}

	g := errgroup.Group{}
	winnerGroup := errgroup.Group{}
	jobChan := make(chan job, s.threads)
	winnerChan := make(chan *PreEndgamePlay)

	for t := 0; t < s.threads; t++ {
		g.Go(func() error {
			for j := range jobChan {
				if err := s.handleJobGeneric(ctx, j, t, winnerChan); err != nil {
					log.Debug().AnErr("err", err).Msg("error-handling-job")
					// Don't exit, to avoid deadlock.
				}
			}
			return nil
		})
	}

	// The determiner of the winner.
	winnerGroup.Go(func() error {
		for p := range winnerChan {
			if !s.winnerSoFar.spreadSet {
				s.winnerSoFar = p
			} else if p.Spread > s.winnerSoFar.Spread {
				s.winnerSoFar = p
			}

		}
		return nil
	})

	permutations := generatePermutations(maybeInBagTiles, s.numinbag)
	queuedJobs := 0
	for _, pidx := range topPlayIdxs {
		for pi, perm := range permutations {
			tiles := make([]tilemapping.MachineLetter, len(perm.Perm))
			for i, el := range perm.Perm {
				tiles[i] = tilemapping.MachineLetter(el)
			}
			jobChan <- job{
				ourMove:   s.plays[pidx],
				fullSolve: true,
				opt:       option{mls: tiles, ct: perm.Count, idx: pi},
			}
			queuedJobs++
		}
	}

	log.Info().Int("numTiebreakerJobs", queuedJobs).Msg("queued-jobs")
	close(jobChan)
	err := g.Wait()
	if err != nil {
		return err
	}

	close(winnerChan)
	winnerGroup.Wait()

	sort.Slice(s.plays, func(i, j int) bool {
		// plays without spread set should be at the bottom.
		if !s.plays[i].spreadSet {
			return false
		}
		if !s.plays[j].spreadSet {
			return true
		}
		return s.plays[i].Spread > s.plays[j].Spread
	})
	return nil
}

type option struct {
	mls         []tilemapping.MachineLetter
	ct          int
	oppEstimate float64
	idx         int
}

// permMatchesTrace returns true when this permutation should be processed.
// Always true when tracing is off. When tracing is on, matches
// sorted(mls[:7]) against traceTargetFirstRack and mls[7:] against
// traceTargetBagTail. When traceOnce is set, only the first matching perm
// is accepted.
// permMatchesTrace returns true if the current game state (opp rack + bag
// content) matches the trace filter targets. It must be called after
// SetRandomRack has been called for the opponent so that g.RackFor reflects
// the actual drawn rack. mls is unused; the check reads game state directly
// so it works regardless of numinbag relative to RackTileLimit.
func (s *Solver) permMatchesTrace(g *game.Game, _ []tilemapping.MachineLetter) bool {
	if s.traceWriter == nil {
		return true
	}
	if len(s.traceTargetBagTail) == 0 {
		return true
	}
	if s.traceOnce && s.traceSeenMatch.Load() {
		return false
	}
	if len(s.traceTargetBagTail) > 0 {
		// Peek() is front→back; drawing is back→front (draw order = reversed Peek()).
		// traceTargetBagTail is stored in draw order, so compare against reversed Peek().
		// Machine-letter comparison is allocation-free and correct for any alphabet.
		bagPeek := g.Bag().Peek()
		target := s.traceTargetBagTail
		if len(bagPeek) != len(target) {
			return false
		}
		for i, ml := range bagPeek {
			if ml != target[len(target)-1-i] {
				return false
			}
		}
	}
	s.traceSeenMatch.Store(true)
	return true
}

// smallMoveStr formats a SmallMove into a human-readable coordinate string.
func smallMoveStr(sm tinymove.SmallMove) string {
	if sm.IsPass() {
		return "PASS"
	}
	row, col, vert := sm.CoordsAndVertical()
	if vert {
		return fmt.Sprintf("%c%d score=%d", 'A'+col, row+1, sm.Score())
	}
	return fmt.Sprintf("%d%c score=%d", row+1, 'A'+col, sm.Score())
}

func (s *Solver) handleJobGeneric(ctx context.Context, j job, thread int,
	winnerChan chan *PreEndgamePlay) error {
	s.arenas[thread].Reset()
	s.threadNestedCalls[thread] = 0
	s.threadMaxNestedDepth[thread] = 0
	s.threadSubPermsEvaluated[thread] = 0

	defer func() {
		if r := recover(); r != nil {
			fmt.Println("-----RECOVER----")
			fmt.Printf("Recovered in handleJobGeneric. job=%v thread=%d\n", j, thread)
			fmt.Println("Game state")
			fmt.Println(s.endgameSolvers[thread].Game().ToDisplayText())
			panic("throwing panic again")
		}
	}()

	if ctx.Err() != nil {
		return ctx.Err()
	}
	if s.logStream != nil {
		s.threadLogs[thread] = jobLog{PEGPlay: j.ourMove.String()}
	}
	if s.skipLossOptim || s.earlyCutoffOptim {
		// Don't cut off moves marked as avoid-prune
		if !s.shouldAvoidPrune(j.ourMove.Play) {
			j.ourMove.RLock()
			if s.skipLossOptim && j.ourMove.FoundLosses > 0 {
				j.ourMove.RUnlock()
				j.ourMove.stopAnalyzing()
				s.numCutoffs.Add(1)
				return nil
			}
			s.potentialWinnerMutex.RLock()
			if s.earlyCutoffOptim && j.ourMove.FoundLosses > s.minPotentialLosses {
				s.potentialWinnerMutex.RUnlock()
				j.ourMove.RUnlock()
				j.ourMove.stopAnalyzing()
				s.numCutoffs.Add(1)
				if s.logStream != nil {
					s.threadLogs[thread].CutoffAtStart = true
					s.threadLogs[thread].FoundLosses = int(j.ourMove.FoundLosses)
					s.threadLogs[thread].MinPotentialLosses = int(s.minPotentialLosses)
				}
				return nil
			}
			s.potentialWinnerMutex.RUnlock()
			j.ourMove.RUnlock()
		}
	}

	if j.maybeInBagTiles != nil {
		return s.processJobPerPlay(ctx, j, thread, winnerChan)
	}
	return s.processJobPerPerm(ctx, j, thread, winnerChan)
}

// processJobPerPerm handles a single pre-assigned permutation. Used when
// len(plays) < threads so all threads stay busy even with few candidate plays.
func (s *Solver) processJobPerPerm(ctx context.Context, j job, thread int,
	winnerChan chan *PreEndgamePlay) error {
	g := s.endgameSolvers[thread].Game()
	mg := s.endgameSolvers[thread].Movegen()

	firstPlayEmptiesBag := j.ourMove.Play.TilesPlayed() >= s.numinbag
	if s.logStream != nil {
		s.threadLogs[thread].Options = make([]jobOptionLog, 1)
		s.threadLogs[thread].PEGPlayEmptiesBag = firstPlayEmptiesBag
		s.threadLogs[thread].EndgamePlies = s.curEndgamePlies
	}

	tilesLeftAfterPlay := s.numinbag - j.ourMove.Play.TilesPlayed()
	if s.maxTilesLeft >= 0 && tilesLeftAfterPlay > s.maxTilesLeft && len(s.solveOnlyMoves) == 0 {
		return nil
	}

	mg.SetPlayRecorder(movegen.AllPlaysSmallRecorder)

	g.ThrowRacksInFor(1 - g.PlayerOnTurn())
	MoveTilesToBeginning(j.opt.mls, g.Bag())
	_, err := g.SetRandomRack(1-g.PlayerOnTurn(), nil)
	if err != nil {
		return err
	}

	s.setInFlight(thread, j.ourMove.Play.ShortDescription(),
		tilemapping.MachineWord(j.opt.mls).UserVisible(g.Alphabet()),
		g.RackLettersFor(1-g.PlayerOnTurn()),
		g.RackLettersFor(g.PlayerOnTurn()))

	if !s.permMatchesTrace(g, j.opt.mls) {
		return nil
	}
	s.trace(0, "[outer] play=%s opp-rack=%s our-rack=%s bag-tail=%s perm-count=%d",
		j.ourMove.Play.ShortDescription(),
		g.RackLettersFor(1-g.PlayerOnTurn()),
		g.RackLettersFor(g.PlayerOnTurn()),
		s.traceTargetBagTail,
		j.opt.ct)

	var sm tinymove.SmallMove
	if j.ourMove.Play.Action() == move.MoveTypePass {
		sm = tinymove.PassMove()
	} else {
		tm := conversions.MoveToTinyMove(j.ourMove.Play)
		sm = tinymove.TilePlayMove(tm, int16(j.ourMove.Play.Score()),
			uint8(j.ourMove.Play.TilesPlayed()), uint8(j.ourMove.Play.PlayLength()))
	}
	if s.logStream != nil {
		s.threadLogs[thread].Options[0].PermutationCount = j.opt.ct
		s.threadLogs[thread].Options[0].PermutationInBag = tilemapping.MachineWord(j.opt.mls).UserVisible(g.Alphabet())
		s.threadLogs[thread].Options[0].OppRack = g.RackLettersFor(1 - g.PlayerOnTurn())
		s.threadLogs[thread].Options[0].OurRack = g.RackLettersFor(g.PlayerOnTurn())
	}

	err = s.recursiveSolve(ctx, thread, j.ourMove, sm, j.opt, winnerChan, 0, firstPlayEmptiesBag, j.fullSolve, 0)
	if err != nil {
		return err
	}
	// Do NOT call j.ourMove.finalize() here. Multiple threads process
	// different permutations of the same pegPlay concurrently; calling
	// finalize() mid-flight would prematurely lock in a partial outcome for
	// another thread's permutation. finalize() is called for all plays after
	// all workers complete (in multithreadSolveGeneric).
	return nil
}

// processJobPerPlay handles all permutations for a single candidate play on
// this thread. Used when len(plays) >= threads so threads stay busy without
// per-permutation job overhead. OppEstimate computation is parallelized since
// each thread does it for its own play.
func (s *Solver) processJobPerPlay(ctx context.Context, j job, thread int,
	winnerChan chan *PreEndgamePlay) error {
	g := s.endgameSolvers[thread].Game()
	mg := s.endgameSolvers[thread].Movegen()

	options := []option{}
	mg.(*movegen.GordonGenerator).SetPlayRecorderTopPlay()
	permutations := generatePermutations(j.maybeInBagTiles, s.numinbag)
	firstPlayEmptiesBag := j.ourMove.Play.TilesPlayed() >= s.numinbag
	tilesLeftAfterPlay := s.numinbag - j.ourMove.Play.TilesPlayed()
	if s.logStream != nil {
		s.threadLogs[thread].Options = make([]jobOptionLog, len(permutations))
		s.threadLogs[thread].PEGPlayEmptiesBag = firstPlayEmptiesBag
		s.threadLogs[thread].EndgamePlies = s.curEndgamePlies
	}
	if s.maxTilesLeft >= 0 && tilesLeftAfterPlay > s.maxTilesLeft && len(s.solveOnlyMoves) == 0 {
		return nil
	}
	for _, perm := range permutations {
		topEquity := 0.0
		tiles := make([]tilemapping.MachineLetter, len(perm.Perm))
		for idx, el := range perm.Perm {
			tiles[idx] = tilemapping.MachineLetter(el)
		}
		if !j.fullSolve {
			g.ThrowRacksInFor(1 - g.PlayerOnTurn())
			MoveTilesToBeginning(tiles, g.Bag())
			if _, err := g.SetRandomRack(1-g.PlayerOnTurn(), nil); err == nil {
				if err := g.PlayMove(j.ourMove.Play, false, 0); err == nil {
					mg.GenAll(g.RackFor(g.PlayerOnTurn()), false)
					if len(mg.Plays()) > 0 {
						topEquity = mg.Plays()[0].Equity()
					}
					g.UnplayLastMove()
				}
			}
		}
		options = append(options, option{
			mls:         tiles,
			ct:          perm.Count,
			oppEstimate: float64(topEquity),
		})
	}
	if !j.fullSolve {
		sort.Slice(options, func(i, j int) bool {
			return options[i].oppEstimate > options[j].oppEstimate
		})
	}

	mg.SetPlayRecorder(movegen.AllPlaysSmallRecorder)

	for idx := range options {
		options[idx].idx = idx
		j.ourMove.RLock()
		s.potentialWinnerMutex.RLock()
		if j.ourMove.FoundLosses > s.minPotentialLosses && s.earlyCutoffOptim && !s.shouldAvoidPrune(j.ourMove.Play) {
			s.potentialWinnerMutex.RUnlock()
			j.ourMove.RUnlock()
			j.ourMove.stopAnalyzing()
			s.numCutoffs.Add(uint64(len(options) - idx))
			if s.logStream != nil {
				s.threadLogs[thread].CutoffWhileIterating = true
				s.threadLogs[thread].FoundLosses = int(j.ourMove.FoundLosses)
				s.threadLogs[thread].MinPotentialLosses = int(s.minPotentialLosses)
			}
			return nil
		}
		s.potentialWinnerMutex.RUnlock()
		j.ourMove.RUnlock()

		g.ThrowRacksInFor(1 - g.PlayerOnTurn())
		MoveTilesToBeginning(options[idx].mls, g.Bag())
		_, err := g.SetRandomRack(1-g.PlayerOnTurn(), nil)
		if err != nil {
			return err
		}

		s.setInFlight(thread, j.ourMove.Play.ShortDescription(),
			tilemapping.MachineWord(options[idx].mls).UserVisible(g.Alphabet()),
			g.RackLettersFor(1-g.PlayerOnTurn()),
			g.RackLettersFor(g.PlayerOnTurn()))

		if !s.permMatchesTrace(g, options[idx].mls) {
			continue
		}
		s.trace(0, "[outer] play=%s opp-rack=%s our-rack=%s bag-tail=%s perm-count=%d",
			j.ourMove.Play.ShortDescription(),
			g.RackLettersFor(1-g.PlayerOnTurn()),
			g.RackLettersFor(g.PlayerOnTurn()),
			s.traceTargetBagTail,
			options[idx].ct)

		var sm tinymove.SmallMove
		if j.ourMove.Play.Action() == move.MoveTypePass {
			sm = tinymove.PassMove()
		} else {
			tm := conversions.MoveToTinyMove(j.ourMove.Play)
			sm = tinymove.TilePlayMove(tm, int16(j.ourMove.Play.Score()),
				uint8(j.ourMove.Play.TilesPlayed()), uint8(j.ourMove.Play.PlayLength()))
		}
		if s.logStream != nil {
			s.threadLogs[thread].Options[idx].PermutationCount = options[idx].ct
			s.threadLogs[thread].Options[idx].PermutationInBag = tilemapping.MachineWord(options[idx].mls).UserVisible(g.Alphabet())
			s.threadLogs[thread].Options[idx].OppRack = g.RackLettersFor(1 - g.PlayerOnTurn())
			s.threadLogs[thread].Options[idx].OurRack = g.RackLettersFor(g.PlayerOnTurn())
		}

		err = s.recursiveSolve(ctx, thread, j.ourMove, sm, options[idx], winnerChan, 0, firstPlayEmptiesBag, j.fullSolve, 0)
		if err != nil {
			return err
		}
		j.ourMove.finalize()
	}
	return nil
}

func (s *Solver) recursiveSolve(ctx context.Context, thread int, pegPlay *PreEndgamePlay,
	moveToMake tinymove.SmallMove, inbagOption option, winnerChan chan *PreEndgamePlay, depth int,
	pegPlayEmptiesBag, fullSolve bool, nestedDepth int) error {

	g := s.endgameSolvers[thread].Game()
	// fmt.Println(strings.Repeat(" ", depth), "entered recursive solve, inbag:",
	// 	tilemapping.MachineWord(inbagOption.mls).UserVisible(g.Alphabet()))
	// Quit early if we already have a loss for this bag option.
	// However, during tiebreak (fullSolve=true), we need to calculate spread values
	// even for known losses, so we skip this optimization.
	if pegPlay.HasLoss(inbagOption.mls) && !fullSolve {
		if s.logStream != nil {
			s.threadLogs[thread].Options[inbagOption.idx].CutoffBecauseAlreadyLoss = true
		}
		// fmt.Println(strings.Repeat(" ", depth), "already have loss for",
		// 	tilemapping.MachineWord(inbagOption.mls).UserVisible(g.Alphabet()))
		return nil
	}

	if g.Playing() == macondo.PlayState_GAME_OVER || g.Bag().TilesRemaining() == 0 {
		var finalSpread int16
		var oppPerspective bool
		var seq []*move.Move
		var val int16
		var err error
		var timeToSolve time.Duration
		if g.Playing() == macondo.PlayState_GAME_OVER {
			// game ended. Should have been because of two-pass
			finalSpread = int16(g.SpreadFor(s.solvingForPlayer))
			// fmt.Println(strings.Repeat(" ", depth), "game ended, score", finalSpread,
			// 	"solvingPlayer pts", g.PointsFor(s.solvingForPlayer),
			// 	"opp pts", g.PointsFor(1-s.solvingForPlayer))
		} else if g.Bag().TilesRemaining() == 0 {
			// if the bag is empty, we just have to solve endgames.
			if g.PlayerOnTurn() != s.solvingForPlayer {
				oppPerspective = true
			}
			// This is the spread after we make our play, from the POV of
			// the player currently on turn
			initialSpread := g.CurrentSpread()
			s.numEndgamesSolved.Add(1)
			s.threadEndgamesSolved[thread].Add(1)
			// Now let's solve the endgame.
			st := time.Now()
			val, seq, err = s.endgameSolvers[thread].QuickAndDirtySolve(ctx, s.curEndgamePlies, thread)
			if err != nil {
				log.Err(err).Msg("quick-and-dirty-solve-error")
				return err
			}
			timeToSolve = time.Since(st)
			finalSpread = val + int16(initialSpread)
			// fmt.Println(strings.Repeat(" ", depth),
			// 	"inbag:", tilemapping.MachineWord(inbagOption.mls).UserVisible(g.Alphabet()),
			// 	"val:", val,
			// 	"seq:", seq)
			if fullSolve && winnerChan != nil {
				ss := finalSpread
				if oppPerspective {
					ss = -ss
				}
				pegPlay.addSpreadStat(int(ss), inbagOption.ct)
				winnerChan <- pegPlay.Copy()
				return nil
			}
		}

		s.trace(depth+nestedDepth*4, "[d=%d n=%d] endgame spread=%+d pv=[%s] oppPerspective=%v",
			depth, nestedDepth, finalSpread,
			s.endgameSolvers[thread].ShortDetails(), oppPerspective)

		switch {
		case (finalSpread > 0 && oppPerspective) || (finalSpread < 0 && !oppPerspective):
			// win for our opponent = loss for us
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Str("ourMove", pegPlay.String()).Msg("we-lose")
			if pegPlayEmptiesBag {
				pegPlay.addWinPctStat(PEGLoss, inbagOption.ct, inbagOption.mls)
			} else {
				pegPlay.setUnfinalizedWinPctStat(PEGLoss, inbagOption.ct, inbagOption.mls)
				// fmt.Println(strings.Repeat(" ", depth), "setting unfinalized loss")
				// XXX: should figure out some way to quit early?
			}
		case finalSpread == 0:
			// draw
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Str("ourMove", pegPlay.String()).Msg("we-tie")
			if pegPlayEmptiesBag {
				pegPlay.addWinPctStat(PEGDraw, inbagOption.ct, inbagOption.mls)
			} else {
				pegPlay.setUnfinalizedWinPctStat(PEGDraw, inbagOption.ct, inbagOption.mls)
				// fmt.Println(strings.Repeat(" ", depth), "setting unfinalized draw")
			}
		case (finalSpread < 0 && oppPerspective) || (finalSpread > 0 && !oppPerspective):
			// loss for our opponent = win for us
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Str("ourMove", pegPlay.String()).Msg("we-win")
			if pegPlayEmptiesBag {
				pegPlay.addWinPctStat(PEGWin, inbagOption.ct, inbagOption.mls)
			} else {
				// if depth > 1 && !oppPerspective {
				// 	// If the turn is back to us and we've already had an opponent move,
				// 	// it means that we don't need to try every possible response to
				// 	// that opp move
				// 	fmt.Println(strings.Repeat(" ", depth), "setting finalized win - optimistic")
				// 	pegPlay.addWinPctStat(PEGWin, inbagOption.ct, inbagOption.mls)
				// } else {
				pegPlay.setUnfinalizedWinPctStat(PEGWin, inbagOption.ct, inbagOption.mls)
				// fmt.Println(strings.Repeat(" ", depth), "setting unfinalized win")
				// }
			}
		}

		if s.logStream != nil && winnerChan != nil {
			s.threadLogs[thread].Options[inbagOption.idx].FinalSpread = int(finalSpread)
			s.threadLogs[thread].Options[inbagOption.idx].OppPerspective = oppPerspective
			s.threadLogs[thread].Options[inbagOption.idx].EndgameMoves = fmt.Sprintf("%v", seq)
			s.threadLogs[thread].Options[inbagOption.idx].GameEnded = g.Playing() == macondo.PlayState_GAME_OVER
			s.threadLogs[thread].Options[inbagOption.idx].TimeToSolveMs = timeToSolve.Milliseconds()
		}

		if pegPlayEmptiesBag && winnerChan != nil {
			winnerChan <- pegPlay.Copy()
		}
		// Otherwise, don't send via winnerChan. We would not be sure enough of the
		// pegPlay's actual Points value, since all of its points could still
		// be unsettled (i.e. they could be eventual draws or losses).
		// XXX: figure out a better cutoff algorithm.
		return nil

	}

	// If the bag is not empty, we must recursively play until it is empty.
	// Use PlaySmallMoveWithDraw to avoid converting SmallMove→Move (saves allocs).
	_, err := g.PlaySmallMoveWithDraw(&moveToMake)
	if err != nil {
		log.Err(err).Msg("play-move-err")
		return err
	}
	s.trace(depth+nestedDepth*4, "[d=%d n=%d] played=%s our=%s opp=%s bag-remaining=%d scoreless=%d",
		depth, nestedDepth,
		smallMoveStr(moveToMake),
		g.RackLettersFor(s.solvingForPlayer),
		g.RackLettersFor(1-s.solvingForPlayer),
		g.Bag().TilesRemaining(),
		g.ScorelessTurns())

	// If the bag is STILL not empty after making our last move:
	if g.Bag().TilesRemaining() > 0 && g.Playing() != macondo.PlayState_GAME_OVER {
		if g.PlayerOnTurn() == s.solvingForPlayer {
			// Our turn: run a nested PEG to find if any reply guarantees a win
			// across all bag orderings we might face.
			err = s.iterateOurReplies(ctx, thread, pegPlay, inbagOption, pegPlayEmptiesBag, nestedDepth)
		} else {
			// Opp's turn: enumerate all replies exhaustively (pessimistic).
			genPlays := s.allocSortedReplies(thread, moveToMake)
			defer s.arenas[thread].Dealloc(len(genPlays))
			err = s.iterateOppReplies(ctx, thread, pegPlay, inbagOption, winnerChan, depth, pegPlayEmptiesBag, fullSolve, genPlays, nestedDepth)
		}
		if err != nil {
			g.UnplayLastMove()
			return err
		}
	} else {
		// bag is empty or game is over; recurse once more to hit the base case.
		err = s.recursiveSolve(ctx, thread, pegPlay, tinymove.DefaultSmallMove, inbagOption, winnerChan, depth+1, pegPlayEmptiesBag, fullSolve, nestedDepth)
		if err != nil {
			log.Err(err).Msg("bag-empty-recursive-solve-err")
		}
	}

	g.UnplayLastMove()
	return err
}

// allocSortedReplies generates all legal replies for the player currently on turn,
// arena-allocates a copy sorted by estimated value (passes prioritized when the
// previous move was also a pass).
func (s *Solver) allocSortedReplies(thread int, prevMove tinymove.SmallMove) []tinymove.SmallMove {
	g := s.endgameSolvers[thread].Game()
	mg := s.endgameSolvers[thread].Movegen()
	mg.GenAll(g.RackFor(g.PlayerOnTurn()), false)
	plays := mg.SmallPlays()
	genPlays := s.arenas[thread].Alloc(len(plays))
	copy(genPlays, plays)
	for idx := range genPlays {
		genPlays[idx].SetEstimatedValue(int16(genPlays[idx].Score()))
		if prevMove.IsPass() && genPlays[idx].IsPass() {
			genPlays[idx].AddEstimatedValue(negamax.EarlyPassBF)
		}
	}
	slices.SortFunc(genPlays, func(a, b tinymove.SmallMove) int {
		return int(b.EstimatedValue()) - int(a.EstimatedValue())
	})
	return genPlays
}

// iterateOppReplies calls recursiveSolve for each opponent reply in genPlays,
// stopping as soon as the current inbagOption.mls is confirmed a loss. A loss
// from any single opp reply is absorbing (setUnfinalizedWinPctStat makes loss
// irrevocable), so remaining replies cannot change the verdict.
func (s *Solver) iterateOppReplies(ctx context.Context, thread int, pegPlay *PreEndgamePlay,
	inbagOption option, winnerChan chan *PreEndgamePlay, depth int,
	pegPlayEmptiesBag, fullSolve bool, genPlays []tinymove.SmallMove, nestedDepth int) error {
	for idx := range genPlays {
		s.trace(depth+nestedDepth*4, "[d=%d n=%d] opp-reply %d/%d %s",
			depth, nestedDepth, idx+1, len(genPlays), smallMoveStr(genPlays[idx]))
		err := s.recursiveSolve(ctx, thread, pegPlay, genPlays[idx], inbagOption, winnerChan, depth+1, pegPlayEmptiesBag, fullSolve, nestedDepth)
		if err != nil {
			log.Err(err).Msg("recursive-solve-err")
			return err
		}
		if !fullSolve && pegPlay.HasLoss(inbagOption.mls) {
			break
		}
	}
	return nil
}

// iterateOurReplies calls recursiveSolve for each of our candidate replies in genPlays.
// iterateOurReplies handles the "our turn, bag still non-empty" frame by running
// a nested PEG: finds if any reply guarantees a win across all bag orderings
// consistent with our current info state, then records that outcome on the outer
// inbagOption.
func (s *Solver) iterateOurReplies(ctx context.Context, thread int, pegPlay *PreEndgamePlay,
	inbagOption option, pegPlayEmptiesBag bool, nestedDepth int) error {
	if s.nestedDepthLimit >= 0 && nestedDepth+1 > s.nestedDepthLimit {
		// Bag not empty but recursion cap reached; skip this bag permutation.
		// Only bag-emptying continuations are evaluated at the cap depth.
		return nil
	}
	outcome, err := s.nestedOurTurnSolve(ctx, thread, nestedDepth+1)
	if err != nil {
		return err
	}
	if pegPlayEmptiesBag {
		pegPlay.addWinPctStat(outcome, inbagOption.ct, inbagOption.mls)
	} else {
		pegPlay.setUnfinalizedWinPctStat(outcome, inbagOption.ct, inbagOption.mls)
	}
	return nil
}

// nestedOurTurnSolve handles the "our turn again with bag still non-empty" frame
// under an imperfect-info model. We don't know which tile is in the bag; we only
// know the multiset of unseen tiles (current bag ∪ opp's rack). We evaluate every
// candidate play P across all bag-orderings consistent with that unseen multiset
// (sub-perms), score each P by wins + 0.5·draws, then pick the play(s) tied for
// the best score. The verdict returned to the outer caller is that play's outcome
// at the *actual* current bag ordering (pessimistic among ties).
//
// The result is cached keyed on info-state (board, our rack, unseen multiset,
// scoreless turns). The cache value is a per-bag-sig verdict map so that any
// caller reaching the same info-state but with a different actual bag ordering
// can retrieve its specific verdict without re-solving.
func (s *Solver) nestedOurTurnSolve(ctx context.Context, thread int, nestedDepth int) (PEGOutcome, error) {
	if ctx.Err() != nil {
		return PEGNotInitialized, ctx.Err()
	}
	s.numNestedCalls.Add(1)
	s.threadNestedCalls[thread]++
	for {
		cur := s.maxNestedDepth.Load()
		if uint64(nestedDepth) <= cur {
			break
		}
		if s.maxNestedDepth.CompareAndSwap(cur, uint64(nestedDepth)) {
			break
		}
	}
	if uint64(nestedDepth) > s.threadMaxNestedDepth[thread] {
		s.threadMaxNestedDepth[thread] = uint64(nestedDepth)
	}

	g := s.endgameSolvers[thread].Game()
	mg := s.endgameSolvers[thread].Movegen()

	// Capture the caller's actual bag ordering before any reshuffling.
	// This is the key into the per-bag verdict map for this call.
	currentBagSig := bagSig(g)

	// Cache lookup: the verdict is a pure function of (info-state, bag-sig).
	cacheKey := s.buildNestedCacheKey(g)
	if verdictMap, ok := s.nestedCache.lookup(cacheKey); ok {
		s.nestedCacheHits.Add(1)
		if v, found := verdictMap[currentBagSig]; found {
			return v, nil
		}
		// Defensive: info-state hit but bag-sig absent — fall through to recompute.
	}
	s.nestedCacheMisses.Add(1)
	nestedLeafStart := s.threadEndgamesSolved[thread].Load()

	// Save bag + racks; restore on return so the caller's UnplayLastMove is clean.
	snap := snapshotPEGState(g)
	defer snap.restore(g)

	opp := 1 - g.PlayerOnTurn()
	subBagSize := g.Bag().TilesRemaining()
	if subBagSize <= InBagMaxLimit {
		s.numNestedByBagSize[subBagSize].Add(1)
	}
	s.threadNestedBagSize[thread].Store(int32(subBagSize))
	defer s.threadNestedBagSize[thread].Store(0)

	// Unseen pool = current bag + opp's rack. This is our full info-set; we
	// cannot distinguish which unseen tiles are in the bag vs. on opp's rack.
	unseenList := make([]int, tilemapping.MaxAlphabetSize)
	for _, t := range g.Bag().Peek() {
		unseenList[int(t)]++
	}
	for _, t := range g.RackFor(opp).TilesOn() {
		unseenList[int(t)]++
	}
	subPerms := generatePermutations(unseenList, subBagSize)

	// Sort sub-perms by opp's expected best score descending: perms where opp scores
	// highest (hardest for us) come first, surfacing losses early so the
	// strict->minLossesSoFar cutoff eliminates weak sub-plays sooner.
	{
		type spWithEst struct {
			sp  Permutation
			est float64
		}
		spe := make([]spWithEst, len(subPerms))
		mg.(*movegen.GordonGenerator).SetPlayRecorderTopPlay()
		for i, sp := range subPerms {
			tiles := make([]tilemapping.MachineLetter, len(sp.Perm))
			for j, el := range sp.Perm {
				tiles[j] = tilemapping.MachineLetter(el)
			}
			spe[i] = spWithEst{sp: sp}
			g.ThrowRacksInFor(opp)
			MoveTilesToBeginning(tiles, g.Bag())
			if _, err := g.SetRandomRack(opp, nil); err == nil {
				mg.GenAll(g.RackFor(opp), false)
				if plays := mg.Plays(); len(plays) > 0 {
					spe[i].est = plays[0].Equity()
				}
			}
		}
		snap.restore(g)
		sort.Slice(spe, func(i, j int) bool { return spe[i].est > spe[j].est })
		for i := range spe {
			subPerms[i] = spe[i].sp
		}
		mg.SetPlayRecorder(movegen.AllPlaysSmallRecorder)
	}

	if s.skipDeepPass {
		lastMoveWasPass := g.ScorelessTurns() > g.LastScorelessTurns()
		mg.SetGenPass(lastMoveWasPass)
		defer mg.SetGenPass(true)
	}
	mg.GenAll(g.RackFor(g.PlayerOnTurn()), false)
	ourPlays := mg.SmallPlays()
	subPlays := s.arenas[thread].Alloc(len(ourPlays))
	copy(subPlays, ourPlays)
	defer s.arenas[thread].Dealloc(len(subPlays))
	// Sort sub-plays by score + leave so the strongest play is evaluated first,
	// tightening minLossesSoFar quickly and enabling the strict->cutoff to fire.
	if s.leaveCalc != nil {
		tmpMove := &move.Move{}
		rack := g.RackFor(g.PlayerOnTurn())
		for i := range subPlays {
			var est float64
			if subPlays[i].IsPass() {
				est = s.leaveCalc.LeaveValue(tilemapping.MachineWord(rack.TilesOn()))
			} else {
				conversions.SmallMoveToMove(subPlays[i], tmpMove, g.Alphabet(), g.Board(), rack)
				est = float64(subPlays[i].Score()) + s.leaveCalc.LeaveValue(tmpMove.Leave())
			}
			subPlays[i].SetEstimatedValue(int16(est))
		}
	} else {
		for i := range subPlays {
			subPlays[i].SetEstimatedValue(int16(subPlays[i].Score()))
		}
	}
	slices.SortFunc(subPlays, func(a, b tinymove.SmallMove) int {
		return int(b.EstimatedValue()) - int(a.EstimatedValue())
	})

	s.trace(nestedDepth*4, "[nested=%d] ENTER our=%s opp=%s bag=%s subBagSize=%d numSubPerms=%d numSubPlays=%d",
		nestedDepth,
		g.RackLettersFor(g.PlayerOnTurn()),
		g.RackLettersFor(opp),
		tilemapping.MachineWord(g.Bag().Peek()).UserVisible(g.Alphabet()),
		subBagSize, len(subPerms), len(subPlays))

	// Evaluate each candidate play across all sub-perms. Track minLossesSoFar
	// for the early-cutoff: plays whose accumulated losses already exceed the
	// current leader can never tie, so we skip their remaining sub-perms.
	allSubPegPlays := make([]*PreEndgamePlay, len(subPlays))
	minLossesSoFar := float32(math.MaxFloat32)

	for subMIdx, subM := range subPlays {
		s.trace(nestedDepth*4, "[nested=%d] subM %d/%d %s",
			nestedDepth, subMIdx+1, len(subPlays), smallMoveStr(subM))

		subPegPlay := &PreEndgamePlay{Play: &move.Move{}}
		allSubPegPlays[subMIdx] = subPegPlay
		subEmptiesBag := int(subM.TilesPlayed()) >= subBagSize

		for pi, subPerm := range subPerms {
			tiles := make([]tilemapping.MachineLetter, len(subPerm.Perm))
			for i, el := range subPerm.Perm {
				tiles[i] = tilemapping.MachineLetter(el)
			}
			subOption := option{mls: tiles, ct: subPerm.Count, idx: pi}

			s.numSubPermsEvaluated.Add(1)
			s.threadSubPermsEvaluated[thread]++

			s.trace(nestedDepth*4+2, "[nested=%d]   subPerm %d/%d tiles=%s",
				nestedDepth, pi+1, len(subPerms),
				tilemapping.MachineWord(tiles).UserVisible(g.Alphabet()))

			g.ThrowRacksInFor(opp)
			MoveTilesToBeginning(tiles, g.Bag())
			if _, err := g.SetRandomRack(opp, nil); err != nil {
				return PEGNotInitialized, err
			}

			if err := s.recursiveSolve(ctx, thread, subPegPlay, subM, subOption, nil, 0, subEmptiesBag, false, nestedDepth); err != nil {
				return PEGNotInitialized, err
			}

			// Strict-greater cutoff: accumulated losses already exceed the best
			// play seen so far, so this play can never tie or beat the leader.
			// Tied plays (==) survive and keep their full outcome row.
			if subPegPlay.FoundLosses > minLossesSoFar {
				s.trace(nestedDepth*4+2, "[nested=%d]   CUTOFF losses=%.1f > min=%.1f, skipping rest",
					nestedDepth, subPegPlay.FoundLosses, minLossesSoFar)
				break
			}
		}

		subPegPlay.finalize()
		if subPegPlay.FoundLosses < minLossesSoFar {
			minLossesSoFar = subPegPlay.FoundLosses
			// Early exit: this play wins every sub-perm with no draws, and all
			// sub-perms were actually evaluated (none depth-skipped). Verdict is
			// WIN for every bag ordering — no need to evaluate remaining plays.
			if minLossesSoFar == 0 && len(subPegPlay.outcomesArray) == len(subPerms) {
				verdictMap := make(map[string]PEGOutcome, len(subPerms))
				for _, o := range subPegPlay.outcomesArray {
					verdictMap[bagSigFromTiles(o.tiles)] = PEGWin
				}
				verdict := verdictMap[currentBagSig]
				s.trace(nestedDepth*4, "[nested=%d] EARLY-WIN verdict=%s leaves=%d",
					nestedDepth, verdict, s.threadEndgamesSolved[thread].Load()-nestedLeafStart)
				s.nestedCache.store(cacheKey, verdictMap)
				return verdict, nil
			}
		}
		s.trace(nestedDepth*4, "[nested=%d] subM %d/%d done: points=%.1f losses=%.1f minLosses=%.1f",
			nestedDepth, subMIdx+1, len(subPlays),
			subPegPlay.Points, subPegPlay.FoundLosses, minLossesSoFar)
	}

	// Collect the tied set: all plays at the minimum loss score. By the strict->
	// cutoff invariant, every tied play completed all its sub-perms, so its
	// outcomesArray is a complete per-sub-perm outcome row.
	var tiedPlays []*PreEndgamePlay
	const epsilon = float32(1e-4)
	for _, p := range allSubPegPlays {
		if p != nil && p.FoundLosses <= minLossesSoFar+epsilon {
			tiedPlays = append(tiedPlays, p)
		}
	}

	// Build the per-bagSig verdict map. For each sub-perm, the verdict is the
	// worst (most pessimistic) outcome across all tied plays: we don't know which
	// of the tied plays we'd actually pick, so we assume the worst case.
	verdictMap := make(map[string]PEGOutcome, len(subPerms))
	if len(tiedPlays) == 0 {
		// No candidate plays (e.g., only passes suppressed by skipDeepPass).
		for _, subPerm := range subPerms {
			tiles := make([]tilemapping.MachineLetter, len(subPerm.Perm))
			for i, el := range subPerm.Perm {
				tiles[i] = tilemapping.MachineLetter(el)
			}
			verdictMap[bagSigFromTiles(tiles)] = PEGLoss
		}
	} else {
		// tiedPlays[0] is the reference for sub-perm enumeration order;
		// all tied plays have the same outcomesArray length and index correspondence.
		ref := tiedPlays[0]
		for j, o := range ref.outcomesArray {
			sig := bagSigFromTiles(o.tiles)
			worst := o.outcome
			for _, tp := range tiedPlays[1:] {
				if tp.outcomesArray[j].outcome > worst { // higher = more pessimistic
					worst = tp.outcomesArray[j].outcome
				}
			}
			verdictMap[sig] = worst
		}
	}

	verdict := verdictMap[currentBagSig]
	s.trace(nestedDepth*4, "[nested=%d] VERDICT=%s tied-set=%d minLosses=%.1f leaves=%d",
		nestedDepth, verdict, len(tiedPlays), minLossesSoFar,
		s.threadEndgamesSolved[thread].Load()-nestedLeafStart)

	s.nestedCache.store(cacheKey, verdictMap)
	return verdict, nil
}

// pegStateSnapshot captures bag order and both racks so the nested PEG can
// restore them after reshuffling state to test sub-permutations.
type pegStateSnapshot struct {
	bag   *tilemapping.Bag
	rack0 []tilemapping.MachineLetter
	rack1 []tilemapping.MachineLetter
}

// snapshotPEGState saves the current bag (including its draw order) and both
// player racks. The saved bag does NOT include the rack tiles; racks are saved
// separately.
func snapshotPEGState(g *game.Game) pegStateSnapshot {
	return pegStateSnapshot{
		bag:   g.Bag().Copy(),
		rack0: append([]tilemapping.MachineLetter(nil), g.RackFor(0).TilesOn()...),
		rack1: append([]tilemapping.MachineLetter(nil), g.RackFor(1).TilesOn()...),
	}
}

// restore puts the game back to the saved state by:
//  1. Throwing current racks into the bag (so the bag absorbs them).
//  2. Overwriting the bag with the saved copy (correct tile order, excluding racks).
//  3. Directly setting both racks from the saved slices without removing from bag.
func (snap *pegStateSnapshot) restore(g *game.Game) {
	g.ThrowRacksIn()
	g.Bag().CopyFrom(snap.bag)
	g.RackFor(0).Set(snap.rack0)
	g.RackFor(1).Set(snap.rack1)
}

type Permutation struct {
	Perm  []int
	Count int
}

func generatePermutations(list []int, k int) []Permutation {
	var result []Permutation
	origList := append([]int{}, list...)
	listCpy := append([]int{}, list...)
	generate(listCpy, origList, k, []int{}, &result)
	return result
}

func generate(list []int, origList []int, k int, currentPerm []int, result *[]Permutation) {
	if k == 0 {
		*result = append(
			*result,
			Permutation{
				Perm:  append([]int{}, currentPerm...),
				Count: product(origList, currentPerm)})
		return
	}

	for i := 0; i < len(list); i++ {
		if list[i] > 0 {
			list[i]--
			currentPerm = append(currentPerm, i)
			generate(list, origList, k-1, currentPerm, result)
			currentPerm = currentPerm[:len(currentPerm)-1]
			list[i]++
		}
	}
}

func product(list []int, currentPerm []int) int {
	result := 1
	for _, index := range currentPerm {
		result *= list[index]
		list[index]--
	}
	for _, index := range currentPerm {
		list[index]++
	}
	return result
}
