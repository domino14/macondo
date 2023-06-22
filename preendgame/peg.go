package preendgame

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tilemapping"
	"github.com/domino14/macondo/zobrist"
	"golang.org/x/sync/errgroup"
)

const InBagMaxLimit = 1

type PreEndgamePlay struct {
	sync.RWMutex
	play      *move.Move
	wins      float32
	winsWith  [][]tilemapping.MachineLetter
	drawsWith [][]tilemapping.MachineLetter
	losesWith [][]tilemapping.MachineLetter
	ignore    bool
}

func (p *PreEndgamePlay) addWinPctStat(win float32, tiles []tilemapping.MachineLetter) {
	p.Lock()
	defer p.Unlock()
	p.wins += win
	switch win {
	case 1:
		p.winsWith = append(p.winsWith, tiles)
	case 0.5:
		p.drawsWith = append(p.drawsWith, tiles)
	case 0:
		p.losesWith = append(p.losesWith, tiles)
	}
}

func (p *PreEndgamePlay) String() string {
	return fmt.Sprintf("<play %v, wins %f>", p.play.ShortDescription(), p.wins)
}

type Solver struct {
	endgameSolvers []*negamax.Solver

	movegen       movegen.MoveGenerator
	game          *game.Game
	gaddag        *kwg.KWG
	ttable        *negamax.TranspositionTable
	zobrist       *zobrist.Zobrist
	endgamePlies  int
	initialSpread int
	threads       int
	plays         []*PreEndgamePlay
}

// Init initializes the solver. It creates all the parallel endgame solvers.
func (s *Solver) Init(g *game.Game, gd *kwg.KWG) error {
	s.threads = int(math.Max(1, float64(runtime.NumCPU()-1)))
	s.ttable = &negamax.TranspositionTable{}
	s.ttable.SetMultiThreadedMode()
	s.game = g.Copy()
	s.game.SetBackupMode(game.SimulationMode)
	s.endgamePlies = 4
	s.gaddag = gd
	return nil
}

func (s *Solver) Solve(ctx context.Context) error {
	playerOnTurn := s.game.PlayerOnTurn()
	if s.game.RackFor(1-playerOnTurn).NumTiles() == 0 {
		_, err := s.game.SetRandomRack(1-playerOnTurn, nil)
		if err != nil {
			return err
		}
	}

	if s.game.Bag().TilesRemaining() > InBagMaxLimit {
		return fmt.Errorf("bag has too many tiles remaining; limit is %d", InBagMaxLimit)
	} else if s.game.Bag().TilesRemaining() == 0 {
		return errors.New("bag is empty; use endgame solver instead")
	}
	s.endgameSolvers = make([]*negamax.Solver, s.threads)
	s.ttable.Reset(0.25)
	s.zobrist = &zobrist.Zobrist{}
	s.zobrist.Initialize(s.game.Board().Dim())
	s.initialSpread = s.game.CurrentSpread()

	for idx := range s.endgameSolvers {
		es := &negamax.Solver{}
		// Copy the game so each endgame solver can manipulate it independently.
		g := s.game.Copy()
		g.SetBackupMode(game.SimulationMode)
		// we need to set the state stack length now to account for the PEG move.
		// there can also be passes etc. just add a hacky number.
		// XXX: state stack length should probably just be a fixed large number.
		g.SetStateStackLength(s.endgamePlies + 5)
		// Set max scoreless turns to 2 in the endgame so we don't generate
		// unnecessary sequences of passes.
		g.SetMaxScorelessTurns(2)
		// Set a fixed order for the bag. This makes it easy for us to control
		// what tiles we draw after making a move.
		g.Bag().SetFixedOrder(true)
		mg := movegen.NewGordonGenerator(s.gaddag, g.Board(), g.Bag().LetterDistribution())
		err := es.Init(mg, g)
		if err != nil {
			return err
		}
		// Endgame itself should be single-threaded; we are solving many individual
		// endgames in parallel.
		es.SetThreads(1)
		// share the same transposition table and zobrist params across all endgames.
		// hopefully this speeds things up.
		es.SetTranspositionTable(s.ttable)
		es.SetZobrist(s.zobrist)
		// Endgame should quit early if it finds any win.
		es.SetFirstWinOptim(true)
		s.endgameSolvers[idx] = es
	}
	s.movegen = movegen.NewGordonGenerator(s.gaddag, s.game.Board(), s.game.Bag().LetterDistribution())
	s.movegen.SetGenPass(true)

	fmt.Println(s.game.ToDisplayText())

	moves := s.movegen.GenAll(s.game.RackFor(s.game.PlayerOnTurn()), false)
	log.Info().Int("nmoves", len(moves)).Msg("peg-generated-moves")
	err := s.multithreadSolve(ctx, moves)

	return err
}

type job struct {
	ourMove      *PreEndgamePlay
	lettersDrawn []tilemapping.MachineLetter
	numDraws     int // how many ways can lettersDrawn be drawn?
}

func (s *Solver) multithreadSolve(ctx context.Context, moves []*move.Move) error {
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
		s.plays[idx] = &PreEndgamePlay{play: play}
	}

	unseenTiles := make([]int, tilemapping.MaxAlphabetSize)
	for _, t := range s.game.RackFor(s.game.NextPlayer()).TilesOn() {
		unseenTiles[t]++
	}
	for _, t := range s.game.Bag().Peek() {
		unseenTiles[t]++
	}
	g := errgroup.Group{}
	log.Debug().Interface("unseen-tiles", unseenTiles).Msg("unseen tiles")
	jobChan := make(chan job, s.threads*2)

	for t := 0; t < s.threads; t++ {
		t := t
		g.Go(func() error {
			for j := range jobChan {
				if err := s.handleJob(ctx, j, t); err != nil {
					return err
				}
			}
			return nil
		})
	}
	queuedJobs := 0
	for _, p := range s.plays {
		if p.play.Action() == move.MoveTypePass {
			// passes handled differently.
			continue
		}
		for t, count := range unseenTiles {
			if count == 0 {
				continue
			}
			j := job{
				ourMove:      p,
				lettersDrawn: []tilemapping.MachineLetter{tilemapping.MachineLetter(t)},
				numDraws:     count,
			}
			queuedJobs++
			jobChan <- j
		}
	}
	log.Info().Int("numJobs", queuedJobs).Msg("queued-jobs")
	close(jobChan)
	err := g.Wait()

	// sort plays by win %
	sort.Slice(s.plays, func(i, j int) bool {
		return s.plays[i].wins > s.plays[j].wins
	})

	return err
}

func (s *Solver) setEndgamePlies(p int) {
	s.endgamePlies = p
}

func (s *Solver) handleJob(ctx context.Context, j job, thread int) error {
	g := s.endgameSolvers[thread].Game()
	// throw opponent's rack in, and then enforce a tile order.
	g.ThrowRacksInFor(1 - g.PlayerOnTurn())
	// Basically, put the tiles we want to draw at the beginning of the bag.
	// The bag drawing algorithm draws tiles from right to left.
	moveTilesToBeginning(j.lettersDrawn, g.Bag())
	// And redraw tiles for opponent. Note that this is not an actual
	// random rack! We are choosing which tiles to draw via the
	// moveTilesToBeginning call above and the fixedOrder setting for the bag.
	_, err := g.SetRandomRack(1-g.PlayerOnTurn(), nil)
	if err != nil {
		return err
	}
	// Finally play move.
	// XXX: Remove this log line / replace drawnLetters as UserVisible allocates/is slow
	log.Debug().Interface("drawnLetters",
		tilemapping.MachineWord(j.lettersDrawn).UserVisible(g.Alphabet())).
		Int("ct", j.numDraws).
		Int("thread", thread).
		Str("rack-for-us", g.RackLettersFor(g.PlayerOnTurn())).
		Str("rack-for-them", g.RackLettersFor(1-g.PlayerOnTurn())).
		Str("play", j.ourMove.play.ShortDescription()).Msg("trying-peg-play")

	err = g.PlayMove(j.ourMove.play, false, 0)
	if err != nil {
		return err
	}
	// This is the spread after we make our play, from the POV of our
	// opponent.
	initialSpread := g.CurrentSpread()
	// Now let's solve the endgame for our opponent.
	val, _, err := s.endgameSolvers[thread].QuickAndDirtySolve(ctx, s.endgamePlies, thread)
	if err != nil {
		return err
	}
	// val is the gain in spread after endgame (or loss, if negative), from
	// POV of opponent.
	// so the actual final spread is val + initialSpread
	finalSpread := val + int16(initialSpread)

	switch {

	case finalSpread > 0:
		// win for our opponent = loss for us
		log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msg("we-lose")
		j.ourMove.addWinPctStat(0, j.lettersDrawn)
	case finalSpread == 0:
		// draw
		log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msg("we-tie")
		j.ourMove.addWinPctStat(0.5*float32(j.numDraws), j.lettersDrawn)
	case finalSpread < 0:
		// loss for our opponent = win for us
		log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msg("we-win")
		j.ourMove.addWinPctStat(float32(j.numDraws), j.lettersDrawn)
	}
	log.Debug().Int("thread", thread).Msg("peg-unplay")

	g.UnplayLastMove()
	return nil
}

func moveTilesToBeginning(order []tilemapping.MachineLetter, bag *tilemapping.Bag) {
	bagContents := bag.Tiles()
	lastPlacedTile := 0
	for didx := lastPlacedTile; didx < len(order); didx++ {
		for bidx, bagTile := range bagContents {
			place := len(order) - didx - 1
			desiredTile := order[didx]
			if desiredTile == bagTile {
				bag.SwapTile(place, bidx)
				lastPlacedTile++
				break
			}
		}
	}

}
