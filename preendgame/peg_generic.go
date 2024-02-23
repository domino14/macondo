package preendgame

import (
	"context"
	"sort"
	"sync/atomic"

	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
	"gonum.org/v1/gonum/stat/combin"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/tinymove/conversions"
)

func (s *Solver) multithreadSolveGeneric(ctx context.Context, moves []*move.Move) ([]*PreEndgamePlay, error) {
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

	g := errgroup.Group{}
	winnerGroup := errgroup.Group{}
	// log.Debug().Interface("maybe-in-bag-tiles", maybeInBagTiles).Msg("unseen tiles")
	jobChan := make(chan job, s.threads)
	winnerChan := make(chan *PreEndgamePlay)

	var processed atomic.Uint32

	for t := 0; t < s.threads; t++ {
		t := t
		g.Go(func() error {
			for j := range jobChan {
				if err := s.handleJobGeneric(ctx, j, t, winnerChan); err != nil {
					log.Debug().AnErr("err", err).Msg("error-handling-job")
					// Don't exit, to avoid deadlock.
				}
				processed.Add(1)
				n := processed.Load()
				if n%500 == 0 {
					log.Info().Uint64("cutoffs", s.numCutoffs.Load()).Msgf("processed %d endgames...", n)
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
			if ppotentialLosses < s.minPotentialLosses {
				log.Info().
					Float32("potentialLosses", ppotentialLosses).
					Str("p", p.String()).
					Float32("n", s.minPotentialLosses).Msg("min potential losses")
				s.minPotentialLosses = ppotentialLosses
			}
		}
		return nil
	})

	s.createGenericPEGJobs(ctx, maybeInBagTiles, jobChan)
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
	// handle this in a bit.

	// if !s.skipTiebreaker {
	// 	err = s.maybeTiebreak(ctx, maybeInBagTiles)
	// 	if err != nil {
	// 		return nil, err
	// 	}
	// }

	if ctx.Err() != nil && (ctx.Err() == context.Canceled || ctx.Err() == context.DeadlineExceeded) {
		log.Info().Msg("timed out or stopped; returning best results so far...")
		err = ErrCanceledEarly
	}
	log.Info().Uint64("solved-endgames", s.numEndgamesSolved.Load()).
		Uint64("cutoff-moves", s.numCutoffs.Load()).
		Str("winner", s.plays[0].String()).Msg("winning-play")

	return s.plays, err
}

func (s *Solver) createGenericPEGJobs(ctx context.Context, maybeInBagTiles []int, jobChan chan job) {
	queuedJobs := 0

	for _, p := range s.plays {
		j := job{
			ourMove:         p,
			maybeInBagTiles: maybeInBagTiles,
		}
		queuedJobs++
		jobChan <- j
	}

	// if !s.skipPassOptim {
	// 	// Handle pass.
	// 	// First, try to pass back with all possible racks.
	// 	for t, count := range maybeInBagTiles {
	// 		if count == 0 {
	// 			continue
	// 		}
	// 		j := job{
	// 			ourMove:   ourPass,
	// 			theirMove: move.NewPassMove(nil, s.game.Alphabet()),
	// 			inbag:     []tilemapping.MachineLetter{tilemapping.MachineLetter(t)},
	// 			numDraws:  count,
	// 		}
	// 		queuedJobs++
	// 		jobChan <- j
	// 	}

	// 	// Then, for every combination of 7 tiles they could have,
	// 	// generate all plays, make each play, and solve endgame from our
	// 	// perspective.
	// 	theirPossibleRack := tilemapping.NewRack(s.game.Alphabet())
	// 	theirPossibleRack.Set(unseenRack)
	// 	// Generate all possible plays for our opponent that are not the pass
	// 	// back (was just handled above).
	// 	s.movegen.SetGenPass(false)
	// 	theirMoves := s.movegen.GenAll(theirPossibleRack, false)
	// 	s.movegen.SetGenPass(true)
	// 	for _, m := range theirMoves {
	// 		if moveIsPossible(m.Tiles(), s.knownOppRack) {
	// 			j := job{
	// 				ourMove:   ourPass,
	// 				theirMove: m,
	// 				inbag:     unseenRack,
	// 			}
	// 			queuedJobs++
	// 			jobChan <- j
	// 		}
	// 	}
	// } else {
	// 	log.Info().Msg("skipping pass analysis")
	// }

	log.Info().Int("numJobs", queuedJobs).Msg("queued-jobs")
	close(jobChan)
}

type option struct {
	mls         []tilemapping.MachineLetter
	ct          int
	oppEstimate float64
}

func (s *Solver) handleJobGeneric(ctx context.Context, j job, thread int,
	winnerChan chan *PreEndgamePlay) error {
	// handle a job generically.
	// parameters are the job move, and tiles that are unseen to us
	// (maybe in bag)

	if ctx.Err() != nil {
		return ctx.Err()
	}
	if s.skipLossOptim || s.earlyCutoffOptim {
		j.ourMove.RLock()
		if s.skipLossOptim && j.ourMove.FoundLosses > 0 {
			j.ourMove.RUnlock()
			j.ourMove.stopAnalyzing()
			s.numCutoffs.Add(1)
			return nil
		}
		// we should check to see if our move has more found losses than
		// _any_ fully analyzed move. If so, it can't possibly win.

		if s.earlyCutoffOptim && j.ourMove.FoundLosses > s.minPotentialLosses {
			// cut off this play. We already have more losses than the
			// fully analyzed play with the minimum known number of losses.
			j.ourMove.RUnlock()
			// log.Debug().Float32("foundLosses", j.ourMove.FoundLosses).
			// 	Float32("minKnownLosses", s.minPotentialLosses).
			// 	Str("ourMove", j.ourMove.String()).
			// 	Msg("stop-analyzing-move")
			j.ourMove.stopAnalyzing()
			s.numCutoffs.Add(1)
			return nil
		}
		j.ourMove.RUnlock()
	}
	g := s.endgameSolvers[thread].Game()
	mg := s.endgameSolvers[thread].Movegen()

	options := []option{}
	mg.SetPlayRecorder(movegen.TopPlayOnlyRecorder)
	permutations := generatePermutations(j.maybeInBagTiles, s.numinbag)
	firstPlayEmptiesBag := j.ourMove.Play.TilesPlayed() >= s.numinbag

	for _, perm := range permutations {
		// use FixedOrder setting to draw known tiles for opponent
		topEquity := 0.0 // or something
		// Basically, put the tiles we (player on turn) want to draw on the left side
		// of the bag.
		// The bag drawing algorithm draws tiles from right to left. We put the
		// "inbag" tiles to the left/"beginning" of the bag.
		tiles := make([]tilemapping.MachineLetter, len(perm.Perm))
		for idx, el := range perm.Perm {
			// Essentially flip the order of the permutation. Since
			// we draw right to left, we want to present the permutation
			// to the user as the order that the bag is being drawn in.
			tiles[len(perm.Perm)-idx-1] = tilemapping.MachineLetter(el)
		}
		// If our first play empties the bag, we want to try to solve the resulting
		// endgames in an advantageous order.
		if firstPlayEmptiesBag {
			g.ThrowRacksInFor(1 - g.PlayerOnTurn())
			moveTilesToBeginning(tiles, g.Bag())
			// And redraw tiles for opponent. Note that this is not an actual
			// random rack! We are choosing which tiles to draw via the
			// moveTilesToBeginning call above and the fixedOrder setting for the bag.
			// This will leave the tiles in "j.inbag" in the bag, for us (player on turn)
			// to draw after we make our play.
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
		}
		// gen top move, find score, sort by scores. We just need
		// a rough estimate of how good our opp's next move will be.

		options = append(options, option{
			mls:         tiles,
			ct:          perm.Count,
			oppEstimate: float64(topEquity),
		})
	}
	// Sort by oppEstimate from most to least.
	// We want to get losing endgames (for us) out of the way early
	// to help with cutoff.
	if firstPlayEmptiesBag {
		sort.Slice(options, func(i, j int) bool {
			return options[i].oppEstimate > options[j].oppEstimate
		})
	}

	mg.SetPlayRecorder(movegen.AllPlaysSmallRecorder)

	// now recursively solve endgames and stuff.
	for idx := range options {
		j.ourMove.RLock()
		if j.ourMove.FoundLosses > s.minPotentialLosses {
			// cut off this play. We already have more losses than the
			// fully analyzed play with the minimum known number of losses.
			j.ourMove.RUnlock()
			// log.Debug().Float32("foundLosses", j.ourMove.FoundLosses).
			// 	Float32("minKnownLosses", s.minPotentialLosses).
			// 	Str("ourMove", j.ourMove.String()).
			// 	Int("optionsIdx", idx).
			// 	Int("thread", thread).
			// 	Int("cutoff", len(options)-idx).
			// 	Msg("stop-analyzing-move-handleentireloop")
			j.ourMove.stopAnalyzing()
			s.numCutoffs.Add(uint64(len(options) - idx))
			return nil
		}
		j.ourMove.RUnlock()

		g.ThrowRacksInFor(1 - g.PlayerOnTurn())
		moveTilesToBeginning(options[idx].mls, g.Bag())
		// not actually a random rack, but it should have been established
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

		err = s.recursiveSolve(ctx, thread, j.ourMove, &sm,
			options[idx], winnerChan, 0)
		if err != nil {
			return err
		}

	}
	return nil
}

func (s *Solver) recursiveSolve(ctx context.Context, thread int, pegPlay *PreEndgamePlay,
	moveToMake *tinymove.SmallMove, inbagOption option, winnerChan chan *PreEndgamePlay, depth int) error {

	g := s.endgameSolvers[thread].Game()
	mg := s.endgameSolvers[thread].Movegen()

	// Quit early if we already have a loss for this bag option.
	if pegPlay.HasLoss(inbagOption.mls) {
		return nil
	}

	if g.Playing() == macondo.PlayState_GAME_OVER || g.Bag().TilesRemaining() == 0 {
		var finalSpread int16
		var oppPerspective bool
		if g.Playing() == macondo.PlayState_GAME_OVER {
			// game ended. Should have been because of two-pass
			finalSpread = int16(g.SpreadFor(s.solvingForPlayer))
			if g.CurrentSpread() == -int(finalSpread) {
				oppPerspective = true
			}
		} else if g.Bag().TilesRemaining() == 0 {
			// if the bag is empty, we just have to solve endgames.
			if g.PlayerOnTurn() != s.solvingForPlayer {
				oppPerspective = true
			}
			// This is the spread after we make our play, from the POV of our
			// opponent.
			initialSpread := g.CurrentSpread()
			// Now let's solve the endgame for our opponent.
			// log.Debug().Int("thread", thread).Str("ourMove", pegPlay.String()).Int("initialSpread", initialSpread).Msg("about-to-solve-endgame")
			val, _, err := s.endgameSolvers[thread].QuickAndDirtySolve(ctx, s.curEndgamePlies, thread)
			if err != nil {
				return err
			}
			s.numEndgamesSolved.Add(1)
			finalSpread = val + int16(initialSpread)
		}

		switch {
		case (finalSpread > 0 && oppPerspective) || (finalSpread < 0 && !oppPerspective):
			// win for our opponent = loss for us
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Str("ourMove", pegPlay.String()).Msg("we-lose")
			pegPlay.addWinPctStat(PEGLoss, inbagOption.ct, inbagOption.mls)
		case finalSpread == 0:
			// draw
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Str("ourMove", pegPlay.String()).Msg("we-tie")
			pegPlay.addWinPctStat(PEGDraw, inbagOption.ct, inbagOption.mls)
		case (finalSpread < 0 && oppPerspective) || (finalSpread > 0 && !oppPerspective):
			// loss for our opponent = win for us
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Str("ourMove", pegPlay.String()).Msg("we-win")
			pegPlay.addWinPctStat(PEGWin, inbagOption.ct, inbagOption.mls)
		}

		winnerChan <- pegPlay
		return nil

	}

	// If the bag is not empty, we must recursively play until it is empty.
	tempm := &move.Move{}
	conversions.SmallMoveToMove(moveToMake, tempm, g.Alphabet(), g.Board(), g.RackFor(g.PlayerOnTurn()))
	err := g.PlayMove(tempm, false, 0)
	if err != nil {
		return err
	}

	var mm *tinymove.SmallMove
	if g.Bag().TilesRemaining() > 0 {
		mg.GenAll(g.RackFor(g.PlayerOnTurn()), false)
		plays := mg.SmallPlays()
		genPlays := make([]tinymove.SmallMove, len(plays))
		copy(genPlays, plays)
		movegen.SmallPlaySlicePool.Put(&plays)

		for idx := range genPlays {
			genPlays[idx].SetEstimatedValue(int16(genPlays[idx].Score()))
			// Always consider passes first as a reply to passes, in order
			// to get some easy info fast.
			if moveToMake.IsPass() && genPlays[idx].IsPass() {
				genPlays[idx].AddEstimatedValue(negamax.EarlyPassOffset)
			}
		}
		sort.Slice(genPlays, func(i int, j int) bool {
			return genPlays[i].EstimatedValue() > genPlays[j].EstimatedValue()
		})
		// XXX: we also need to ignore plays that are not among the best
		// we found. We assume that we (player who the PEG is being solved for)
		// would never make an incorrect play (i.e. one that doesn't win
		// as much as the winners).

		for idx := range genPlays {
			mm = &genPlays[idx]
			err = s.recursiveSolve(ctx, thread, pegPlay, mm, inbagOption, winnerChan, depth+1)
			if err != nil {
				return err
			}
		}
	} else {
		// if the bag is empty after we've played moveToMake, the next
		// iteration here will solve the endgames.
		err = s.recursiveSolve(ctx, thread, pegPlay, nil, inbagOption, winnerChan, depth+1)
	}
	g.UnplayLastMove()
	return err
}

type Permutation struct {
	Perm  []int
	Count int
}

func generatePermutations(list []int, k int) []Permutation {
	var result []Permutation
	origList := append([]int{}, list...)
	generate(list, origList, k, []int{}, &result)
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
