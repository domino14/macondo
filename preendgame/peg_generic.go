package preendgame

import (
	"context"
	"fmt"
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
	// only tiebreak plays that empty the bag.
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

	topN := min(len(topPlayIdxs), TieBreakerPlays)
	log.Info().Msgf("%d plays tied for first, taking top %d and tie-breaking...", len(topPlayIdxs), topN)
	topPlayIdxs = topPlayIdxs[:topN]

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

	queuedJobs := 0
	for _, pidx := range topPlayIdxs {
		j := job{
			ourMove:         s.plays[pidx],
			maybeInBagTiles: maybeInBagTiles,
			fullSolve:       true,
		}
		queuedJobs++
		jobChan <- j
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
	// handle a job generically.
	// parameters are the job move, and tiles that are unseen to us
	// (maybe in bag)

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
		j.ourMove.RLock()
		if s.skipLossOptim && j.ourMove.FoundLosses > 0 {
			j.ourMove.RUnlock()
			j.ourMove.stopAnalyzing()
			s.numCutoffs.Add(1)
			return nil
		}
		// we should check to see if our move has more found losses than
		// _any_ fully analyzed move. If so, it can't possibly win.
		s.potentialWinnerMutex.RLock()
		if s.earlyCutoffOptim && j.ourMove.FoundLosses > s.minPotentialLosses {
			// cut off this play. We already have more losses than the
			// fully analyzed play with the minimum known number of losses.
			s.potentialWinnerMutex.RUnlock()
			j.ourMove.RUnlock()
			// log.Debug().Float32("foundLosses", j.ourMove.FoundLosses).
			// 	Float32("minKnownLosses", s.minPotentialLosses).
			// 	Str("ourMove", j.ourMove.String()).
			// 	Msg("stop-analyzing-move")
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
	g := s.endgameSolvers[thread].Game()
	mg := s.endgameSolvers[thread].Movegen().(*movegen.GordonGenerator)

	options := []option{}
	mg.SetTopPlayOnlyRecorder()
	mg.SetShadowEnabled(true)
	permutations := generatePermutations(j.maybeInBagTiles, s.numinbag)
	// fmt.Println("perms", permutations)
	firstPlayEmptiesBag := j.ourMove.Play.TilesPlayed() >= s.numinbag
	if s.logStream != nil {
		s.threadLogs[thread].Options = make([]jobOptionLog, len(permutations))
		s.threadLogs[thread].PEGPlayEmptiesBag = firstPlayEmptiesBag
		s.threadLogs[thread].EndgamePlies = s.curEndgamePlies
	}
	for _, perm := range permutations {
		// fmt.Println("perm", perm)
		// use FixedOrder setting to draw known tiles for opponent
		topEquity := 0.0 // or something
		// Basically, put the tiles we (player on turn) want to draw on the left side
		// of the bag.
		// The bag drawing algorithm draws tiles from right to left. We put the
		// "inbag" tiles to the left/"beginning" of the bag.
		tiles := make([]tilemapping.MachineLetter, len(perm.Perm))
		for idx, el := range perm.Perm {
			tiles[idx] = tilemapping.MachineLetter(el)
		}
		// If our first play empties the bag, we want to try to solve the resulting
		// endgames in an advantageous order.
		if firstPlayEmptiesBag && !j.fullSolve {
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
		} else if s.skipNonEmptyingOptim {
			// If the play does not empty the bag, then return if we
			// want to skip plays that don't empty the bag.
			return nil
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
	if firstPlayEmptiesBag && !j.fullSolve {
		sort.Slice(options, func(i, j int) bool {
			return options[i].oppEstimate > options[j].oppEstimate
		})
	}

	mg.SetPlayRecorder(movegen.PlayRecorderSmallMove)
	mg.SetShadowEnabled(false)

	// now recursively solve endgames and stuff.
	for idx := range options {
		options[idx].idx = idx
		j.ourMove.RLock()
		s.potentialWinnerMutex.RLock()
		if j.ourMove.FoundLosses > s.minPotentialLosses && s.earlyCutoffOptim {
			s.potentialWinnerMutex.RUnlock()
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
		if s.logStream != nil {
			s.threadLogs[thread].Options[idx].PermutationCount = options[idx].ct
			s.threadLogs[thread].Options[idx].PermutationInBag = tilemapping.MachineWord(options[idx].mls).UserVisible(g.Alphabet())
			s.threadLogs[thread].Options[idx].OppRack = g.RackLettersFor(1 - g.PlayerOnTurn())
			s.threadLogs[thread].Options[idx].OurRack = g.RackLettersFor(g.PlayerOnTurn())
		}

		err = s.recursiveSolve(ctx, thread, j.ourMove, sm,
			options[idx], winnerChan, 0, firstPlayEmptiesBag, j.fullSolve)
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
	mg := s.endgameSolvers[thread].Movegen()
	// fmt.Println(strings.Repeat(" ", depth), "entered recursive solve, inbag:",
	// 	tilemapping.MachineWord(inbagOption.mls).UserVisible(g.Alphabet()))
	// Quit early if we already have a loss for this bag option.
	if pegPlay.HasLoss(inbagOption.mls) {
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
			if fullSolve {
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

		if s.logStream != nil {
			s.threadLogs[thread].Options[inbagOption.idx].FinalSpread = int(finalSpread)
			s.threadLogs[thread].Options[inbagOption.idx].OppPerspective = oppPerspective
			s.threadLogs[thread].Options[inbagOption.idx].EndgameMoves = fmt.Sprintf("%v", seq)
			s.threadLogs[thread].Options[inbagOption.idx].GameEnded = g.Playing() == macondo.PlayState_GAME_OVER
			s.threadLogs[thread].Options[inbagOption.idx].TimeToSolveMs = timeToSolve.Milliseconds()
		}

		if pegPlayEmptiesBag {
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
				genPlays[idx].AddEstimatedValue(negamax.EarlyPassBF)
			}
		}
		sort.Slice(genPlays, func(i int, j int) bool {
			return genPlays[i].EstimatedValue() > genPlays[j].EstimatedValue()
		})

		for idx := range genPlays {

			// remove; only for print purposes
			tempm := &move.Move{}
			conversions.SmallMoveToMove(genPlays[idx], tempm, g.Alphabet(), g.Board(), g.RackFor(g.PlayerOnTurn()))

			// fmt.Println(strings.Repeat(" ", depth), "onturn", g.PlayerOnTurn(), "idx", idx, "try next:", tempm.ShortDescription())
			err = s.recursiveSolve(ctx, thread, pegPlay, genPlays[idx], inbagOption, winnerChan, depth+1, pegPlayEmptiesBag, fullSolve)
			if err != nil {
				log.Err(err).Msg("recursive-solve-err")
				g.UnplayLastMove()
				return err
			}
			if g.PlayerOnTurn() == s.solvingForPlayer {
				// We're back to solving for ourselves. We need to be optimistic.
				// We assume that we (player who the PEG is being solved for)
				// would never make an incorrect play (i.e. one that doesn't win
				// as much as the winners). Therefore if we've already found a win
				// deeper in the tree, exit early and assume we will find the best
				// reply to our opponent.
				if pegPlay.OutcomeFor(inbagOption.mls) == PEGWin {
					// fmt.Println(strings.Repeat(" ", depth), "onturn", g.PlayerOnTurn(), "breaking early cuzza win")
					break
				}
			}

		}
	} else {
		// if the bag is empty after we've played moveToMake, the next
		// iteration here will solve the endgames.
		// fmt.Println(strings.Repeat(" ", depth), "bag is empty or game is over; recursing again to finalize")
		err = s.recursiveSolve(ctx, thread, pegPlay, tinymove.DefaultSmallMove, inbagOption, winnerChan, depth+1, pegPlayEmptiesBag, fullSolve)
		if err != nil {
			log.Err(err).Msg("bag-empty-recursive-solve-err")
		}
	}
	// fmt.Println(strings.Repeat(" ", depth), "unplaying last move")

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
