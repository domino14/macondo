package preendgame

import (
	"context"
	"fmt"
	"slices"
	"sort"
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
)

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
	if err != nil {
		return nil, err
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
	log.Info().Uint64("solved-endgames", s.numEndgamesSolved.Load()).
		Uint64("cutoff-moves", s.numCutoffs.Load()).
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
		firstPlayEmptiesBag := p.Play.TilesPlayed() >= s.numinbag
		opts := make([]option, len(permutations))
		for j, perm := range permutations {
			tiles := make([]tilemapping.MachineLetter, len(perm.Perm))
			for k, el := range perm.Perm {
				tiles[k] = tilemapping.MachineLetter(el)
			}
			opt := option{mls: tiles, ct: perm.Count}
			if firstPlayEmptiesBag {
				g0.ThrowRacksInFor(1 - g0.PlayerOnTurn())
				moveTilesToBeginning(tiles, g0.Bag())
				if _, err := g0.SetRandomRack(1-g0.PlayerOnTurn(), nil); err == nil {
					if err := g0.PlayMove(p.Play, false, 0); err == nil {
						mg0.GenAll(g0.RackFor(g0.PlayerOnTurn()), false)
						if len(mg0.Plays()) > 0 {
							opt.oppEstimate = float64(mg0.Plays()[0].Equity())
						}
						g0.UnplayLastMove()
					}
				}
			}
			opts[j] = opt
		}
		if firstPlayEmptiesBag {
			sort.Slice(opts, func(i, j int) bool {
				return opts[i].oppEstimate > opts[j].oppEstimate
			})
		}
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
	queuedJobs := 0
	if sortedOpts != nil {
		// Per-permutation mode: one job per (play, perm), pre-sorted hardest-first.
		for pi, p := range s.plays {
			for _, opt := range sortedOpts[pi] {
				jobChan <- job{ourMove: p, opt: opt}
				queuedJobs++
			}
		}
	} else {
		// Per-play mode: one job per play; the job processes all permutations
		// internally, keeping oppEstimate computation parallelized across threads.
		for _, p := range s.plays {
			jobChan <- job{ourMove: p, maybeInBagTiles: maybeInBagTiles}
			queuedJobs++
		}
	}
	log.Info().Int("numJobs", queuedJobs).Msg("queued-jobs")
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

func (s *Solver) handleJobGeneric(ctx context.Context, j job, thread int,
	winnerChan chan *PreEndgamePlay) error {
	s.arenas[thread].Reset()

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

	if !firstPlayEmptiesBag && s.skipNonEmptyingOptim {
		return nil
	}

	mg.SetPlayRecorder(movegen.AllPlaysSmallRecorder)

	g.ThrowRacksInFor(1 - g.PlayerOnTurn())
	moveTilesToBeginning(j.opt.mls, g.Bag())
	_, err := g.SetRandomRack(1-g.PlayerOnTurn(), nil)
	if err != nil {
		return err
	}

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

	err = s.recursiveSolve(ctx, thread, j.ourMove, sm, j.opt, winnerChan, 0, firstPlayEmptiesBag, j.fullSolve)
	if err != nil {
		return err
	}
	j.ourMove.finalize()
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
	if s.logStream != nil {
		s.threadLogs[thread].Options = make([]jobOptionLog, len(permutations))
		s.threadLogs[thread].PEGPlayEmptiesBag = firstPlayEmptiesBag
		s.threadLogs[thread].EndgamePlies = s.curEndgamePlies
	}
	for _, perm := range permutations {
		topEquity := 0.0
		tiles := make([]tilemapping.MachineLetter, len(perm.Perm))
		for idx, el := range perm.Perm {
			tiles[idx] = tilemapping.MachineLetter(el)
		}
		if firstPlayEmptiesBag && !j.fullSolve {
			g.ThrowRacksInFor(1 - g.PlayerOnTurn())
			moveTilesToBeginning(tiles, g.Bag())
			_, err := g.SetRandomRack(1-g.PlayerOnTurn(), nil)
			if err != nil {
				return err
			}
			err = g.PlayMove(j.ourMove.Play, false, 0)
			if err != nil {
				return err
			}
			mg.GenAll(g.RackFor(g.PlayerOnTurn()), false)
			topEquity = mg.Plays()[0].Equity()
			g.UnplayLastMove()
		} else if s.skipNonEmptyingOptim {
			return nil
		}
		options = append(options, option{
			mls:         tiles,
			ct:          perm.Count,
			oppEstimate: float64(topEquity),
		})
	}
	if firstPlayEmptiesBag && !j.fullSolve {
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
		moveTilesToBeginning(options[idx].mls, g.Bag())
		_, err := g.SetRandomRack(1-g.PlayerOnTurn(), nil)
		if err != nil {
			return err
		}

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

		err = s.recursiveSolve(ctx, thread, j.ourMove, sm, options[idx], winnerChan, 0, firstPlayEmptiesBag, j.fullSolve)
		if err != nil {
			return err
		}
		j.ourMove.finalize()
	}
	return nil
}

func (s *Solver) recursiveSolve(ctx context.Context, thread int, pegPlay *PreEndgamePlay,
	moveToMake tinymove.SmallMove, inbagOption option, winnerChan chan *PreEndgamePlay, depth int,
	pegPlayEmptiesBag, fullSolve bool) error {

	defer func() {
		if r := recover(); r != nil {
			fmt.Println("-----RECOVER----")
			fmt.Printf("Recovered in recursiveSolve. thread=%d pegPlay=%v moveToMake=%v inBagOption=%v depth=%d pegPlayEmptiesBag=%v\n",
				thread, pegPlay, moveToMake, inbagOption, depth, pegPlayEmptiesBag)
			fmt.Println("Game state is")
			fmt.Println(s.endgameSolvers[thread].Game().ToDisplayText())
			panic("throwing panic again")
		}
	}()

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
			// Now let's solve the endgame.
			st := time.Now()
			val, seq, err = s.endgameSolvers[thread].QuickAndDirtySolve(ctx, s.curEndgamePlies, thread)
			if err != nil {
				log.Err(err).Msg("quick-and-dirty-solve-error")
				return err
			}
			timeToSolve = time.Since(st)
			s.numEndgamesSolved.Add(1)
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
	tempm := &move.Move{}
	conversions.SmallMoveToMove(moveToMake, tempm, g.Alphabet(), g.Board(), g.RackFor(g.PlayerOnTurn()))
	err := g.PlayMove(tempm, false, 0)
	if err != nil {
		log.Err(err).Msg("play-move-err")
		return err
	}
	// fmt.Println(strings.Repeat(" ", depth), "playing move", tempm.ShortDescription(), "onturnnow", g.PlayerOnTurn())

	// If the bag is STILL not empty after making our last move:
	if g.Bag().TilesRemaining() > 0 && g.Playing() != macondo.PlayState_GAME_OVER {
		if g.PlayerOnTurn() == s.solvingForPlayer {
			// Our turn: run a nested PEG to find if any reply guarantees a win
			// across all bag orderings we might face.
			err = s.iterateOurReplies(ctx, thread, pegPlay, inbagOption, pegPlayEmptiesBag)
		} else {
			// Opp's turn: enumerate all replies exhaustively (pessimistic).
			genPlays := s.allocSortedReplies(thread, moveToMake)
			defer s.arenas[thread].Dealloc(len(genPlays))
			err = s.iterateOppReplies(ctx, thread, pegPlay, inbagOption, winnerChan, depth, pegPlayEmptiesBag, fullSolve, genPlays)
		}
		if err != nil {
			g.UnplayLastMove()
			return err
		}
	} else {
		// bag is empty or game is over; recurse once more to hit the base case.
		err = s.recursiveSolve(ctx, thread, pegPlay, tinymove.DefaultSmallMove, inbagOption, winnerChan, depth+1, pegPlayEmptiesBag, fullSolve)
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

// iterateOppReplies calls recursiveSolve for every opponent reply in genPlays.
// All plays are enumerated — none are skipped — so the leaves' calls to
// setUnfinalizedWinPctStat accumulate every possible opp outcome for inbagOption,
// letting the pessimistic aggregation rule (any Loss dominates) take effect.
func (s *Solver) iterateOppReplies(ctx context.Context, thread int, pegPlay *PreEndgamePlay,
	inbagOption option, winnerChan chan *PreEndgamePlay, depth int,
	pegPlayEmptiesBag, fullSolve bool, genPlays []tinymove.SmallMove) error {
	for idx := range genPlays {
		err := s.recursiveSolve(ctx, thread, pegPlay, genPlays[idx], inbagOption, winnerChan, depth+1, pegPlayEmptiesBag, fullSolve)
		if err != nil {
			log.Err(err).Msg("recursive-solve-err")
			return err
		}
	}
	return nil
}

// iterateOurReplies calls recursiveSolve for each of our candidate replies in genPlays.
// NOTE: The early exit below is the known bug — it stops as soon as one reply has
// recorded a PEGWin for this specific inbagOption, without checking whether that
// reply also wins across all other bag orderings we would face in a real game.
// Replaced by nestedOurTurnSolve in a later PR.
// iterateOurReplies handles the "our turn, bag still non-empty" frame by running
// a nested PEG: finds if any reply guarantees a win across all bag orderings
// consistent with our current info state, then records that outcome on the outer
// inbagOption.
func (s *Solver) iterateOurReplies(ctx context.Context, thread int, pegPlay *PreEndgamePlay,
	inbagOption option, pegPlayEmptiesBag bool) error {
	outcome, err := s.nestedOurTurnSolve(ctx, thread)
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

// nestedOurTurnSolve is called when, during recursive descent, it becomes our
// turn again with the bag still non-empty. Instead of enumerating our replies
// at the fixed outer bag ordering (which we don't actually know in a real game),
// it runs a proper nested PEG: for each candidate reply, it checks whether that
// reply wins across ALL bag orderings consistent with the current info state.
//
// Returns:
//   - PEGWin  if some reply guarantees a win across all sub-orderings.
//   - PEGDraw if no reply guarantees a win but some guarantees a non-loss.
//   - PEGLoss otherwise.
//
// The caller must call setUnfinalizedWinPctStat with the returned outcome for
// the outer inbagOption. fullSolve must be false (tiebreak paths never reach
// our-turn frames because tiebreak only applies to bag-emptying outer plays).
func (s *Solver) nestedOurTurnSolve(ctx context.Context, thread int) (PEGOutcome, error) {
	if ctx.Err() != nil {
		return PEGNotInitialized, ctx.Err()
	}
	g := s.endgameSolvers[thread].Game()
	mg := s.endgameSolvers[thread].Movegen()

	// Save bag + racks. The defer restores them once we're done testing
	// sub-permutations, so the caller's UnplayLastMove sees the right state.
	snap := snapshotPEGState(g)
	defer snap.restore(g)

	opp := 1 - g.PlayerOnTurn()
	subBagSize := g.Bag().TilesRemaining()

	// Build the unseen tile pool from our perspective: current bag + opp's rack.
	unseenList := make([]int, tilemapping.MaxAlphabetSize)
	for _, t := range g.Bag().Peek() {
		unseenList[int(t)]++
	}
	for _, t := range g.RackFor(opp).TilesOn() {
		unseenList[int(t)]++
	}
	subPerms := generatePermutations(unseenList, subBagSize)

	// Generate our candidate replies. We snapshot our rack's tile list before
	// doing anything that might disturb the movegen state.
	mg.GenAll(g.RackFor(g.PlayerOnTurn()), false)
	ourPlays := mg.SmallPlays()
	subPlays := s.arenas[thread].Alloc(len(ourPlays))
	copy(subPlays, ourPlays)
	defer s.arenas[thread].Dealloc(len(subPlays))

	anyNonLoss := false

	for _, subM := range subPlays {
		subPegPlay := &PreEndgamePlay{Play: &move.Move{}}
		subEmptiesBag := int(subM.TilesPlayed()) >= subBagSize

		for pi, subPerm := range subPerms {
			tiles := make([]tilemapping.MachineLetter, len(subPerm.Perm))
			for i, el := range subPerm.Perm {
				tiles[i] = tilemapping.MachineLetter(el)
			}
			subOption := option{mls: tiles, ct: subPerm.Count, idx: pi}

			// Mirror handleJobGeneric's per-permutation setup (peg_generic.go:482-490):
			// put opp's tiles back in the bag, reorder for this sub-perm, redraw opp's rack.
			g.ThrowRacksInFor(opp)
			moveTilesToBeginning(tiles, g.Bag())
			if _, err := g.SetRandomRack(opp, nil); err != nil {
				return PEGNotInitialized, err
			}

			// winnerChan=nil: nested sub-plays must not touch the global winner
			// accounting (minPotentialLosses, earlyCutoffOptim).
			if err := s.recursiveSolve(ctx, thread, subPegPlay, subM, subOption, nil, 0, subEmptiesBag, false); err != nil {
				return PEGNotInitialized, err
			}

			// Early exit: a loss for any sub-perm means this sub-play can never
			// be guaranteed-win or guaranteed-non-loss. Skip remaining sub-perms.
			if subPegPlay.HasLoss(subOption.mls) {
				break
			}
		}

		subPegPlay.finalize()

		if subPegPlay.IsGuaranteedWin() {
			return PEGWin, nil
		}
		if subPegPlay.IsGuaranteedNonLoss() {
			anyNonLoss = true
		}
	}

	if anyNonLoss {
		return PEGDraw, nil
	}
	return PEGLoss, nil
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
