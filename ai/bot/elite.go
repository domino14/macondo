package bot

import (
	"context"
	"math"
	"runtime"
	"strings"
	"time"

	"github.com/pkg/errors"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/kwg"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

const InferencesSimLimit = 2

func endGameBest(ctx context.Context, p *BotTurnPlayer, endgamePlies int) (*move.Move, error) {
	logger := zerolog.Ctx(ctx)

	if !HasEndgame(p.botType, p.cfg.BotSpec) {
		// Just return the static best play if we don't have an endgame engine.
		return p.GenerateMoves(1)[0], nil
	}
	gd, err := kwg.GetKWG(p.Game.Config().WGLConfig(), p.Game.LexiconName())
	if err != nil {
		return nil, err
	}
	// make a copy of the game.
	gameCopy := p.Game.Copy()
	gameCopy.SetBackupMode(game.SimulationMode)
	gameCopy.SetStateStackLength(endgamePlies + 1)
	gen1 := movegen.NewGordonGenerator(gd, gameCopy.Board(), p.Game.Rules().LetterDistribution())
	err = p.endgamer.Init(gen1, gameCopy)
	if err != nil {
		return nil, err
	}
	maxThreads := runtime.NumCPU()
	if maxThreads > negamax.MaxLazySMPThreads {
		maxThreads = negamax.MaxLazySMPThreads
	}
	p.endgamer.SetThreads(maxThreads)
	p.endgamer.SetPreventSlowroll(true)
	v, seq, err := p.endgamer.Solve(ctx, endgamePlies)
	if err != nil {
		return nil, err
	}
	p.lastCalculatedDetails = p.endgamer.ShortDetails()

	logger.Info().Int16("best-endgame-val", v).Interface("seq", seq).Msg("endgame-solve-done")
	return seq[0], nil
}

func preendgameBest(ctx context.Context, p *BotTurnPlayer) (*move.Move, error) {
	logger := zerolog.Ctx(ctx)
	if !HasPreendgame(p.botType, p.cfg.BotSpec) {
		// Just return the static best play if we don't have a pre-endgame engine
		return p.GenerateMoves(1)[0], nil
	}
	gd, err := kwg.GetKWG(p.Game.Config().WGLConfig(), p.Game.LexiconName())
	if err != nil {
		return nil, err
	}
	err = p.preendgamer.Init(p.Game, gd)
	if err != nil {
		return nil, err
	}
	// If we're down by a lot, we will probably need to bingo out, so set
	// endgame plies to only a couple
	ourSpread := math.Abs(float64(p.Game.SpreadFor(p.Game.PlayerOnTurn())))
	switch {
	case ourSpread >= 100:
		p.preendgamer.SetEndgamePlies(2)
	case ourSpread >= 80:
		p.preendgamer.SetEndgamePlies(3)
	case ourSpread >= 60:
		p.preendgamer.SetEndgamePlies(4)
	case ourSpread >= 50:
		p.preendgamer.SetEndgamePlies(5)
	default:
		p.preendgamer.SetEndgamePlies(7)
	}
	p.preendgamer.SetIterativeDeepening(true)
	if p.cfg.UseOppRacksInAnalysis {
		oppRack := p.Game.RackFor(p.Game.NextPlayer())
		logger.Info().Str("rack", oppRack.String()).Msg("setting-known-opp-rack")
		p.preendgamer.SetKnownOppRack(oppRack.TilesOn())
	}
	p.preendgamer.SetEarlyCutoffOptim(true)

	moves, err := p.preendgamer.Solve(ctx)
	if err != nil {
		log.Err(err).Msg("preendgamer-solve-error")
		return nil, err
	}
	p.lastCalculatedDetails = p.preendgamer.ShortDetails()

	return moves[0].Play, nil

}

func monteCarloBest(ctx context.Context, p *BotTurnPlayer, simPlies int, moves []*move.Move) (*move.Move, error) {
	logger := zerolog.Ctx(ctx)

	var inferTimeout context.Context
	var cancel context.CancelFunc
	if HasInfer(p.botType, p.cfg.BotSpec) && p.Game.Bag().TilesRemaining() > 0 {
		logger.Debug().Msg("running inference..")
		p.inferencer.Init(p.Game, p.simmerCalcs, p.Config())
		if p.simThreads != 0 {
			p.inferencer.SetThreads(p.simThreads)
		}
		err := p.inferencer.PrepareFinder(p.Game.RackFor(p.Game.PlayerOnTurn()).TilesOn())
		if err != nil {
			// ignore all errors and move on.
			logger.Debug().AnErr("inference-prepare-error", err).Msg("probably-ok")
		} else {
			inferTimeout, cancel = context.WithTimeout(context.Background(),
				time.Duration(20*int(time.Second)))
			defer cancel()
			err = p.inferencer.Infer(inferTimeout)
			if err != nil {
				// ignore all errors and move on.
				logger.Debug().AnErr("inference-error", err).Msg("probably-ok")
			}
		}
	}

	p.simmer.Init(p.Game, p.simmerCalcs, p.simmerCalcs[0].(*equity.CombinedStaticCalculator), p.Config())
	if p.simThreads != 0 {
		p.simmer.SetThreads(p.simThreads)
	}
	p.simmer.PrepareSim(simPlies, moves)
	p.simmer.SetStoppingCondition(montecarlo.Stop99)
	if p.cfg.StochasticStaticEval {
		p.simmer.SetStochasticStaticEval(true)
	}
	// p.simmer.SetAutostopIterationsCutoff(2500)
	// p.simmer.SetAutostopPPScaling(1500)

	if HasInfer(p.botType, p.cfg.BotSpec) && len(p.inferencer.Inferences().InferredRacks) > InferencesSimLimit {
		logger.Info().Int("inferences", len(p.inferencer.Inferences().InferredRacks)).Msg("using inferences in sim")
		p.simmer.SetInferences(p.inferencer.Inferences().InferredRacks, p.inferencer.Inferences().RackLength, montecarlo.InferenceWeightedRandomRacks)
	}
	if p.cfg.UseOppRacksInAnalysis {
		oppRack := p.Game.RackFor(p.Game.NextPlayer())
		logger.Info().Str("rack", oppRack.String()).Msg("setting-known-opp-rack")
		p.simmer.SetKnownOppRack(oppRack.TilesOn())
	}

	// Simulate is a blocking play:
	err := p.simmer.Simulate(ctx)
	if err != nil {
		return nil, err
	}
	play := p.simmer.WinningPlay()
	logger.Info().Interface("winning-move", play.Move().String()).Msg("sim-done")
	p.lastCalculatedDetails = p.simmer.ShortDetails(4)
	return play.Move(), nil
}

// note that this function does not obey the context timeout (since it doesn't check it)
func montecarloBestWithMagpie(ctx context.Context, p *BotTurnPlayer, simPlies int, moves []*move.Move) (*move.Move, string, error) {
	logger := zerolog.Ctx(ctx)

	if p.magpie == nil {
		// Initialize Magpie if not already done
		p.magpie = magpie.NewMagpie(p.Game.Config())
	}

	cgp := p.Game.ToCGP(true, game.WithMagpieMode(true), game.WithHideLexicon(true))
	// let magpie generate its own moves, just pass in the length.
	bestMove, rawOutput := p.magpie.BestSimmingMove(cgp, p.Game.Lexicon().Name(), simPlies, len(moves))
	logger.Info().Str("best-move", bestMove).Msg("magpie-sim-done")

	// Otherwise, it's a regular tile-play move. Split at the dot to get the position and tiles.
	parts := strings.SplitN(bestMove, ".", 2)

	m, err := p.ParseMove(p.Game.PlayerOnTurn(), false, parts, false)
	if err != nil {
		log.Err(err).Msg("error-parsing-best-move")
		return nil, "", errors.Wrap(err, "error parsing best move from magpie")
	}

	return m, rawOutput, nil
}
