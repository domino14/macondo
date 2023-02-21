package rangefinder

import (
	"context"
	"errors"
	"math"
	"runtime"
	"sync"

	"github.com/rs/zerolog/log"
	"github.com/samber/lo"
	"golang.org/x/sync/errgroup"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

var ErrMoveTypeNotSupported = errors.New("opponent move type not suitable for inference")
var ErrNoEvents = errors.New("no events")
var ErrBagEmpty = errors.New("bag is empty")

const (
	// If the player found a play within this limit, then count the rack
	// for inferences.
	InferenceEquityLimit = 1
)

type RangeFinder struct {
	origGame          *game.Game
	gameCopies        []*game.Game
	equityCalculators []equity.EquityCalculator
	aiplayers         []*aiturnplayer.AIStaticTurnPlayer
	iterationCount    int
	threads           int

	working      bool
	readyToInfer bool

	inferenceBagMap []uint8
	cfg             *config.Config
	lastOppMove     *move.Move
	// tiles used by the last opponent's move, from their rack:
	lastOppMoveRackTiles []alphabet.MachineLetter
	inferences           [][]alphabet.MachineLetter
}

func (r *RangeFinder) Init(game *game.Game, eqCalcs []equity.EquityCalculator,
	cfg *config.Config) {

	r.origGame = game
	r.equityCalculators = eqCalcs
	r.threads = int(math.Max(1, float64(runtime.NumCPU()-1)))
	r.cfg = cfg
}

func (r *RangeFinder) SetThreads(t int) {
	r.threads = t
}

func (r *RangeFinder) PrepareFinder() error {
	evts := r.origGame.History().Events
	if len(evts) == 0 {
		return ErrNoEvents
	}
	if r.origGame.Bag().TilesRemaining() == 0 {
		return ErrBagEmpty
	}
	oppEvt := evts[len(evts)-1]
	if oppEvt.Type != macondo.GameEvent_EXCHANGE && oppEvt.Type != macondo.GameEvent_TILE_PLACEMENT_MOVE {
		return ErrMoveTypeNotSupported
	}

	// We must reset the game back to what it looked like before the opp's move.
	// Do this with a copy.
	history := r.origGame.History()

	history.Events = history.Events[:len(evts)-1]
	gameCopy, err := game.NewFromHistory(history, r.origGame.Rules(), len(history.Events))
	if err != nil {
		return err
	}

	// create rack from the last move.
	r.lastOppMove, err = game.MoveFromEvent(oppEvt, r.origGame.Alphabet(), gameCopy.Board())
	if err != nil {
		return err
	}
	r.lastOppMoveRackTiles = lo.Filter(r.lastOppMove.Tiles(), func(t alphabet.MachineLetter, idx int) bool {
		return t != alphabet.PlayedThroughMarker
	})
	gameCopy.ThrowRacksIn()

	r.inferenceBagMap = gameCopy.Bag().PeekMap()

	gameCopy.SetRandomRack(gameCopy.PlayerOnTurn(), r.lastOppMoveRackTiles)
	// Save the state of the bag after we assign the random rack. Remove only
	// lastOppMove rack but nothing else.
	for _, ml := range r.lastOppMoveRackTiles {
		idx, _ := ml.IntrinsicTileIdx()
		r.inferenceBagMap[idx]--
	}

	r.gameCopies = []*game.Game{}
	r.aiplayers = []*aiturnplayer.AIStaticTurnPlayer{}

	for i := 0; i < r.threads; i++ {
		r.gameCopies = append(r.gameCopies, gameCopy.Copy())

		player, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(r.gameCopies[i], r.origGame.Config(), r.equityCalculators)
		if err != nil {
			return err
		}
		r.aiplayers = append(r.aiplayers, player)
	}

	r.readyToInfer = true
	r.iterationCount = 0
	return nil
}

func (r *RangeFinder) Infer(ctx context.Context) error {
	if !r.readyToInfer {
		return errors.New("not ready")
	}
	r.working = true
	defer func() {
		r.working = false
		log.Info().Msg("inference engine quitting")
	}()

	syncExitChan := make(chan bool, r.threads)
	ctrl := errgroup.Group{}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ctrl.Go(func() error {
		defer func() {
			log.Debug().Msgf("Inference engine controller thread exiting")
		}()
		for range ctx.Done() {
		}
		log.Debug().Msgf("Context is done: %v", ctx.Err())
		for t := 0; t < r.threads; t++ {
			syncExitChan <- true
		}
		log.Debug().Msgf("Sent sync-exit messages to children threads...")

		return ctx.Err()
	})

	g := errgroup.Group{}
	var iterMutex sync.Mutex
	for t := 0; t < r.threads; t++ {
		t := t
		g.Go(func() error {
			defer func() {
				log.Debug().Msgf("Thread %v exiting inferrer", t)
			}()
			log.Debug().Msgf("Thread %v starting inferrer", t)
			for {
				iterMutex.Lock()
				r.iterationCount++
				iterMutex.Unlock()
				// TODO: single iteration of infer
				inference, err := r.inferSingle(t)
				if err != nil {
					log.Err(err).Msg("infer-single-error")
					cancel()
				}
				if len(inference) > 0 {
					iterMutex.Lock()
					r.inferences = append(r.inferences, inference)
					iterMutex.Unlock()
				}
				select {
				case v := <-syncExitChan:
					log.Debug().Msgf("Thread %v got sync msg %v", t, v)
					return nil
				default:
					// Do nothing
				}
			}
		})
	}

	err := g.Wait()
	log.Debug().Msgf("errgroup returned err %v", err)

	ctrlErr := ctrl.Wait()
	log.Debug().Msgf("ctrl errgroup returned err %v", ctrlErr)

	if ctrlErr == context.Canceled || ctrlErr == context.DeadlineExceeded {
		// Not actually an error
		log.Debug().Msg("it's ok, not an error")
		return nil
	}
	return ctrlErr

}

func (r *RangeFinder) inferSingle(thread int) ([]alphabet.MachineLetter, error) {
	g := r.gameCopies[thread]
	// Since we took back the last move, the player on turn should be our opponent
	// (the person whose rack we are inferring)
	opp := g.PlayerOnTurn()
	extraDrawn, err := g.SetRandomRack(opp, r.lastOppMoveRackTiles)
	if err != nil {
		return nil, err
	}
	log.Debug().Interface("rack", g.RackLettersFor(opp)).Msg("rack")
	log.Debug().Interface("extra-drawn", extraDrawn).Msg("extra-drawn")

	bestMoves := r.aiplayers[thread].GenerateMoves(15)

	winningEquity := bestMoves[0].Equity()
	for _, m := range bestMoves {
		if m.Equity()+InferenceEquityLimit >= winningEquity {
			// consider this move
			if movesAreSame(m, r.lastOppMove, g.Board()) {
				// copy extraDrawn, as setRandomRack does not allocate for it.
				tiles := make([]alphabet.MachineLetter, len(extraDrawn))
				copy(tiles, extraDrawn)
				return tiles, nil
			}
		}
	}
	return nil, nil
}

func movesAreSame(m1 *move.Move, m2 *move.Move, g *board.GameBoard) bool {
	if m1.Action() == move.MoveTypeExchange && m2.Action() == move.MoveTypeExchange {
		// we just care about the number of tiles exchanged here, since
		// presumably we don't actually know which tiles were exchanged
		return m1.TilesPlayed() == m2.TilesPlayed()
	}
	// Otherwise, it's a tile-play move

	checkTransposition := false
	if g.IsEmpty() {
		checkTransposition = true
	}
	ignoreLeave := true
	if m1.Equals(m2, checkTransposition, ignoreLeave) {
		return true
	}

	// Otherwise check if it's a single-tile move.
	if m1.TilesPlayed() == 1 && m2.TilesPlayed() == 1 &&
		uniqueSingleTileKey(m1) == uniqueSingleTileKey(m2) {
		return true
	}
	return false
}

func uniqueSingleTileKey(m *move.Move) int {
	// Find the tile.
	var idx int
	var tile alphabet.MachineLetter
	for idx, tile = range m.Tiles() {
		if tile != alphabet.PlayedThroughMarker {
			break
		}
	}
	row, col, vert := m.CoordsAndVertical()
	// We want to get the coordinate of the tile that is on the board itself.
	if vert {
		row += idx
	} else {
		col += idx
	}
	// A unique, fast to compute key for this play.
	return row + alphabet.MaxAlphabetSize*col +
		alphabet.MaxAlphabetSize*alphabet.MaxAlphabetSize*int(tile)
}
