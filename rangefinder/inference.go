package rangefinder

import (
	"context"
	"errors"
	"math"
	"runtime"

	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
)

var ErrUnableToInfer = errors.New("unable to infer")
var ErrNoEvents = errors.New("no events")
var ErrBagEmpty = errors.New("bag is empty")

type RangeFinder struct {
	origGame          *game.Game
	gameCopies        []*game.Game
	equityCalculators []equity.EquityCalculator
	aiplayers         []aiturnplayer.AITurnPlayer

	threads int

	working bool
	cfg     *config.Config
}

func (r *RangeFinder) Init(game *game.Game, eqCalcs []equity.EquityCalculator,
	cfg *config.Config) {

	r.origGame = game
	r.equityCalculators = eqCalcs
	r.threads = int(math.Max(1, float64(runtime.NumCPU()-1)))
	r.cfg = cfg
}

func (r *RangeFinder) makeGameCopies() error {
	log.Debug().Int("threads", r.threads).Msg("makeGameCopies")
	r.gameCopies = []*game.Game{}
	r.aiplayers = []aiturnplayer.AITurnPlayer{}

	for i := 0; i < r.threads; i++ {
		r.gameCopies = append(r.gameCopies, r.origGame.Copy())

		player, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(r.gameCopies[i], r.origGame.Config(), r.equityCalculators)
		if err != nil {
			return err
		}
		r.aiplayers = append(r.aiplayers, player)
	}
	return nil
}

func (r *RangeFinder) Infer(ctx context.Context) error {
	evts := r.origGame.History().Events
	if len(evts) == 0 {
		return ErrNoEvents
	}
	if r.origGame.Bag().TilesRemaining() == 0 {
		return ErrBagEmpty
	}

	r.working = true
	defer func() {
		r.working = false
		log.Info().Msg("inference engine quitting")
	}()

	syncExitChan := make(chan bool, r.threads)
	ctrl := errgroup.Group{}

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
	for t := 0; t < r.threads; t++ {
		t := t
		g.Go(func() error {
			defer func() {
				log.Debug().Msgf("Thread %v exiting inferrer", t)
			}()
			log.Debug().Msgf("Thread %v starting inferrer", t)
			for {

				// TODO: single iteration of infer

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
	/*
		oppMove := evts[len(evts)-1]
		switch oppMove.Type {
		case macondo.GameEvent_TILE_PLACEMENT_MOVE:

		case macondo.GameEvent_EXCHANGE:
			// for exchange all the info that we have is the number of tiles.
			// still, this could potentially be useful.
		default:
			// For now we can only do inferences if last move is tile placement
			// or exchange.
			return ErrUnableToInfer
		}*/
	if ctrlErr == context.Canceled || ctrlErr == context.DeadlineExceeded {
		// Not actually an error
		log.Debug().Msg("it's ok, not an error")
		return nil
	}
	return ctrlErr

}
