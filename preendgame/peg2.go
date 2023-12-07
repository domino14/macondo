package preendgame

import (
	"context"
	"sort"
	"sync/atomic"

	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tilemapping"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
)

// peg2 contains code for solving a 2-in-the-bag preendgame.
// of course we should be able to make the peg code generic so it can solve
// any number in the bag. However, I'm not very smart, and a lot of this
// will be copy-pasted/ugly code until I can figure out how to combine them.
// Make it work, make it beautiful, make it fast.

// multithreadSolve2 solves a 2-PEG, of course.
func (s *Solver) multithreadSolve2(ctx context.Context, moves []*move.Move) ([]*PreEndgamePlay, error) {
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
				if err := s.handleJob2(ctx, j, t, winnerChan); err != nil {
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
			// 9*8 = 72 possible tile draws. Note that order matters sometimes
			// and sometimes not. Order matters if our play uses just one tile.
			ppotentialLosses := 72.0 - p.Points
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

	s.createPEGJobs2(ctx, maybeInBagTiles, jobChan)
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
	// XXX: this is trickier. Skip tiebreaker for right now.
	// if !s.skipTiebreaker {
	// 	err = s.maybeTiebreak2(ctx, maybeInBagTiles)
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

func (s *Solver) createPEGJobs2(ctx context.Context, maybeInBagTiles []int, jobChan chan job) {
	queuedJobs := 0
	var ourPass *PreEndgamePlay

	unseenRack := []tilemapping.MachineLetter{}
	unseenRack = append(unseenRack, s.game.RackFor(s.game.NextPlayer()).TilesOn()...)
	unseenRack = append(unseenRack, s.game.Bag().Peek()...)

	for _, p := range s.plays {
		if p.Play.Action() == move.MoveTypePass {
			// passes handled differently.
			ourPass = p
			continue
		}
		j := job{
			ourMove:         p,
			maybeInBagTiles: maybeInBagTiles,
		}
		queuedJobs++
		jobChan <- j
	}

	if !s.skipPassOptim {
		// Handle pass.
		// First, try to pass back with all possible racks.
		for i := 0; i < len(maybeInBagTiles); i++ {
			for j := i; j < len(maybeInBagTiles); j++ {
				if maybeInBagTiles[i] == 0 || maybeInBagTiles[j] == 0 {
					continue
				}
				count := maybeInBagTiles[i]
				inbag := []tilemapping.MachineLetter{tilemapping.MachineLetter(i)}
				if i != j {
					inbag = append(inbag, tilemapping.MachineLetter(j))
					count *= maybeInBagTiles[j]
					// since we need to do permutations, not combinations
					// (order matters)
					count *= 2
				} else {
					if count == 1 {
						continue
					}
					count *= (maybeInBagTiles[j] - 1)
				}
				j := job{
					ourMove:   ourPass,
					theirMove: move.NewPassMove(nil, s.game.Alphabet()),
					inbag:     inbag,
					numDraws:  count,
				}
				queuedJobs++
				jobChan <- j
			}
		}

		// Then, for every combination of 7 tiles they could have,
		// generate all plays, make each play, and solve endgame from our
		// perspective.
		theirPossibleRack := tilemapping.NewRack(s.game.Alphabet())
		theirPossibleRack.Set(unseenRack)
		// Generate all possible plays for our opponent that are not the pass
		// back (was just handled above).
		s.movegen.SetGenPass(false)
		theirMoves := s.movegen.GenAll(theirPossibleRack, false)
		s.movegen.SetGenPass(true)
		for _, m := range theirMoves {
			if moveIsPossible(m.Tiles(), s.knownOppRack) {
				j := job{
					ourMove:   ourPass,
					theirMove: m,
					inbag:     unseenRack,
				}
				queuedJobs++
				jobChan <- j
			}
		}
	} else {
		log.Info().Msg("skipping pass analysis")
	}

	log.Info().Int("numJobs", queuedJobs).Msg("queued-jobs")
	close(jobChan)
}

func (s *Solver) handleJob2(ctx context.Context, j job, thread int, winnerChan chan *PreEndgamePlay) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if s.skipLossOptim {
		j.ourMove.RLock()
		if j.ourMove.FoundLosses > 0 {
			j.ourMove.RUnlock()
			j.ourMove.stopAnalyzing()
			s.numCutoffs.Add(1)
			return nil
		}
		j.ourMove.RUnlock()
	}

	if s.earlyCutoffOptim {
		j.ourMove.RLock()
		// we should check to see if our move has more found losses than
		// _any_ fully analyzed move. If so, it can't possibly win.

		if j.ourMove.FoundLosses > s.minPotentialLosses {
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
	if j.ourMove.Play.Action() == move.MoveTypePass {
		if j.theirMove.Action() != move.MoveTypePass {
			return s.handleNonpassResponseToPass2(ctx, j, thread)
		} else {
			return s.handlePassResponseToPass(ctx, j, thread, winnerChan)
		}
	}
	if len(j.maybeInBagTiles) > 0 {
		return s.handleEntirePreendgamePlay2(ctx, j, thread, winnerChan)
	} else if j.fullSolve {
		return s.handleFullSolve(ctx, j, thread, winnerChan)
	} else {
		panic("should not be here")
	}
	return nil
}

func (s *Solver) handleNonpassResponseToPass2(ctx context.Context, j job, thread int) error {
	// This function handles a situation in the pre-endgame where we start with
	// a pass, but opponent makes a play that is not a pass.

	// - we have unseen tiles (j.inbag, not a great name for this variable)
	// and tiles in the play (j.theirMove)
	// - determine which tiles could be in the bag and still allow j.theirMove to
	// be played. Call this set <T>
	// - If for EVERY tileset in <T> we have a result of LOSS already, we can exit
	// early.
	// - If for ANY tileset in <T> we have a result of DRAW, we must still analyze
	// to make sure this isn't a loss.
	// - If for ANY tileset in <T> we have a result of WIN, we must still analyze
	// to make sure this isn't a draw or loss.
	// clean this up, way too inefficient.
	pt := possibleTilesInBag(j.inbag, j.theirMove.Tiles(), s.knownOppRack)
	// XXX: Figure out why we don't need to send to winnerChan in this function
	if len(pt) == 0 {
		log.Warn().Msgf("possible tiles in bag is empty; inbag = %v, theirMove = %v, oppRack = %v",
			j.inbag, j.theirMove, s.knownOppRack)
		return nil
	}
	// create 2-tile permutations of leaves.
	splitPt := permuteLeaves(pt, 2)

	if j.ourMove.AllHaveLoss(splitPt) {
		// log.Debug().Str("their-move", j.theirMove.ShortDescription()).
		// 	Msg("exiting-early-no-new-info")
		return nil
	}

	g := s.endgameSolvers[thread].Game()

	// throw opponent's rack in
	g.ThrowRacksInFor(1 - g.PlayerOnTurn())
	rack := tilemapping.NewRack(g.Alphabet())
	rack.Set(j.inbag)
	// Assign opponent the entire rack, which may be longer than 7 tiles long.
	g.SetRackForOnly(1-g.PlayerOnTurn(), rack)

	// log.Debug().Interface("drawnLetters",
	// 	tilemapping.MachineWord(j.inbag).UserVisible(g.Alphabet())).
	// 	Int("ct", j.numDraws).
	// 	Int("thread", thread).
	// 	Str("rack-for-us", g.RackLettersFor(g.PlayerOnTurn())).
	// 	Str("rack-for-them", g.RackLettersFor(1-g.PlayerOnTurn())).
	// 	Str("their-play", j.theirMove.ShortDescription()).
	// 	Msgf("trying-peg-play; splitpt=%v", splitPt)

	// Play our pass
	err := g.PlayMove(j.ourMove.Play, false, 0)
	if err != nil {
		return err
	}
	// Play their play
	err = g.PlayMove(j.theirMove, false, 0)
	if err != nil {
		return err
	}
	// XXX LOOKING HERE IDK WHATS GOING ON
	// solve the endgame from OUR perspective
	// This is the spread for us currently.
	initialSpread := g.CurrentSpread()
	// This only works if theirMove emptied the bag. If it didn't...
	//
	val, _, err := s.endgameSolvers[thread].QuickAndDirtySolve(ctx, s.curEndgamePlies, thread)
	if err != nil {
		return err
	}
	s.numEndgamesSolved.Add(1)

	// val is the gain in spread after endgame (or loss, if negative), from
	// our own POV.
	// so the actual final spread is val + initialSpread
	finalSpread := val + int16(initialSpread)

	for _, tileset := range splitPt {
		ct := 0
		for _, t := range j.inbag {
			// XXX: assumes 1-PEG. Rework this later.
			if tileset[0] == t {
				ct++
			}
		}

		switch {

		case finalSpread > 0:
			// win for us
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msgf("p-we-win-tileset-%v", tileset)
			j.ourMove.setWinPctStat(PEGWin, ct, tileset)
		case finalSpread == 0:
			// draw
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msgf("p-we-tie-tileset-%v", tileset)
			j.ourMove.setWinPctStat(PEGDraw, ct, tileset)
		case finalSpread < 0:
			// loss for us
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msgf("p-we-lose-tileset-%v", tileset)
			j.ourMove.setWinPctStat(PEGLoss, ct, tileset)
		}
	}

	g.UnplayLastMove() // Unplay opponent's last move.
	g.UnplayLastMove() // and unplay the pass from our end that started this whole thing.

	return nil
}

func (s *Solver) handleEntirePreendgamePlay2(ctx context.Context, j job, thread int,
	winnerChan chan *PreEndgamePlay) error {
	// j.maybeInBagTiles has all the tiles possibly in bag.
	// We must:
	// 1) Figure out a heuristic order in which to run the endgames
	// 2) Start endgames sequentially
	// 3) Check after each endgame whether we should cut off further evaluation
	// Note: do not handle passes here.
	g := s.endgameSolvers[thread].Game()
	mg := s.endgameSolvers[thread].Movegen()
	type option struct {
		ml          tilemapping.MachineLetter
		ct          int
		oppEstimate float64
	}
	options := []option{}
	mg.SetPlayRecorder(movegen.TopPlayOnlyRecorder)
	for t, ct := range j.maybeInBagTiles {
		if ct == 0 {
			continue
		}
		// use FixedOrder setting to draw known tiles for opponent
		g.ThrowRacksInFor(1 - g.PlayerOnTurn())
		// Basically, put the tiles we (player on turn) want to draw on the left side
		// of the bag.
		// The bag drawing algorithm draws tiles from right to left. We put the
		// "inbag" tiles to the left/"beginning" of the bag so that the player
		// NOT ON TURN can't draw them.
		moveTilesToBeginning(
			[]tilemapping.MachineLetter{tilemapping.MachineLetter(t)},
			g.Bag())
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
		// gen top move, find score, sort by scores. We just need
		// a rough estimate of how good our opp's next move will be.

		mg.GenAll(g.RackFor(g.PlayerOnTurn()), false)
		options = append(options, option{
			ml:          tilemapping.MachineLetter(t),
			ct:          ct,
			oppEstimate: mg.Plays()[0].Equity(),
		})
		g.UnplayLastMove()
	}
	// Sort by oppEstimate from most to least.
	// We want to get losing endgames (for us) out of the way early
	// to help with cutoff.
	sort.Slice(options, func(i, j int) bool {
		return options[i].oppEstimate > options[j].oppEstimate
	})
	mg.SetPlayRecorder(movegen.AllPlaysSmallRecorder)
	// Now solve all endgames sequentially.
	for idx := range options {
		if s.earlyCutoffOptim {
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
		}

		inbag := []tilemapping.MachineLetter{options[idx].ml}

		// see comments above. We are establishing a known tile order.
		g.ThrowRacksInFor(1 - g.PlayerOnTurn())
		moveTilesToBeginning(inbag, g.Bag())
		_, err := g.SetRandomRack(1-g.PlayerOnTurn(), nil)
		if err != nil {
			return err
		}
		err = g.PlayMove(j.ourMove.Play, false, 0)
		if err != nil {
			return err
		}
		var finalSpread int16
		// This is the spread after we make our play, from the POV of our
		// opponent.
		initialSpread := g.CurrentSpread()
		// Now let's solve the endgame for our opponent.
		// log.Debug().Int("thread", thread).Str("ourMove", j.ourMove.String()).Int("initialSpread", initialSpread).Msg("about-to-solve-endgame")
		val, _, err := s.endgameSolvers[thread].QuickAndDirtySolve(ctx, s.curEndgamePlies, thread)
		if err != nil {
			return err
		}
		s.numEndgamesSolved.Add(1)

		// val is the gain in spread after endgame (or loss, if negative), from
		// POV of opponent.
		// so the actual final spread is val + initialSpread
		finalSpread = val + int16(initialSpread)

		switch {

		case finalSpread > 0:
			// win for our opponent = loss for us
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Str("ourMove", j.ourMove.String()).Msg("we-lose")
			j.ourMove.addWinPctStat(PEGLoss, options[idx].ct, inbag)
		case finalSpread == 0:
			// draw
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Str("ourMove", j.ourMove.String()).Msg("we-tie")
			j.ourMove.addWinPctStat(PEGDraw, options[idx].ct, inbag)
		case finalSpread < 0:
			// loss for our opponent = win for us
			// log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Str("ourMove", j.ourMove.String()).Msg("we-win")
			j.ourMove.addWinPctStat(PEGWin, options[idx].ct, inbag)
		}
		g.UnplayLastMove()
		winnerChan <- j.ourMove
	}
	return nil
}
