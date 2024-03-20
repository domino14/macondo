package bot

import (
	"context"
	"math"
	"runtime"
	"time"

	"github.com/rs/zerolog"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

const InferencesSimLimit = 400

// Elite bot uses Monte Carlo simulations to rank plays, plays an endgame,
// a pre-endgame (when ready).

// BestPlay picks the highest play by win percentage. It uses montecarlo
// and some other smart things to figure it out.
func eliteBestPlay(ctx context.Context, p *BotTurnPlayer) (*move.Move, error) {
	logger := zerolog.Ctx(ctx)
	var moves []*move.Move
	// First determine what stage of the game we are in.
	tr := p.Game.Bag().TilesRemaining()
	// We don't necessarily know the number of tiles on our opponent's rack.
	opp := p.Game.RackFor(p.Game.NextPlayer()).NumTiles()
	// If this is an annotated game, we may not have full rack info.
	unseen := int(opp) + tr
	// Assume our own rack is fully known, however. So if unseen == 7, the bag
	// is empty and we should assign the oppRack accordingly.
	useEndgame := false
	usePreendgame := false
	endgamePlies := 0
	simPlies := 0

	if unseen <= 7 {
		useEndgame = true
		if tr > 0 {
			logger.Debug().Msg("assigning all unseen to opp")
			// bag is actually empty. Assign all of unseen to the opponent.
			mls := make([]tilemapping.MachineLetter, tr)
			err := p.Game.Bag().Draw(tr, mls)
			if err != nil {
				return nil, err
			}
			for _, t := range mls {
				p.Game.RackFor(p.Game.NextPlayer()).Add(t)
			}
		}
		// Just some sort of estimate
		endgamePlies = unseen + int(p.Game.RackFor(p.Game.PlayerOnTurn()).NumTiles())
	} else if unseen > 7 && unseen <= 8 {
		usePreendgame = true
	} else if unseen > 8 && unseen <= 14 {
		moves = p.GenerateMoves(80)
		simPlies = unseen
	} else {
		moves = p.GenerateMoves(40)
		if p.minSimPlies > 2 {
			simPlies = p.minSimPlies
		} else {
			simPlies = 2
		}
	}
	simThreads := p.simThreads
	if p.simThreads == 0 {
		simThreads = p.simmer.Threads()
	}

	logger.Info().Int("simPlies", simPlies).
		Int("simThreads", simThreads).
		Int("endgamePlies", endgamePlies).
		Bool("useEndgame", useEndgame).
		Int("unseen", unseen).
		Bool("useKnownOppRack", p.cfg.UseOppRacksInAnalysis).
		Int("consideredMoves", len(moves)).Msg("elite-player")

	if useEndgame {
		return endGameBest(ctx, p, endgamePlies)
	} else if usePreendgame {
		return preendgameBest(ctx, p)
	} else {
		return nonEndgameBest(ctx, p, simPlies, moves)
	}

}

func endGameBest(ctx context.Context, p *BotTurnPlayer, endgamePlies int) (*move.Move, error) {
	logger := zerolog.Ctx(ctx)

	if !HasEndgame(p.botType) {
		// Just return the static best play if we don't have an endgame engine.
		return p.GenerateMoves(1)[0], nil
	}
	gd, err := kwg.Get(p.Game.Config().AllSettings(), p.Game.LexiconName())
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
	p.endgamer.SetThreads(maxThreads)
	v, seq, err := p.endgamer.Solve(ctx, endgamePlies)
	if err != nil {
		return nil, err
	}
	logger.Info().Int16("best-endgame-val", v).Interface("seq", seq).Msg("endgame-solve-done")
	return seq[0], nil
}

func preendgameBest(ctx context.Context, p *BotTurnPlayer) (*move.Move, error) {
	logger := zerolog.Ctx(ctx)
	if !HasPreendgame(p.botType) {
		// Just return the static best play if we don't have a pre-endgame engine
		return p.GenerateMoves(1)[0], nil
	}
	gd, err := kwg.Get(p.Game.Config().AllSettings(), p.Game.LexiconName())
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
		return nil, err
	}
	return moves[0].Play, nil

}

func nonEndgameBest(ctx context.Context, p *BotTurnPlayer, simPlies int, moves []*move.Move) (*move.Move, error) {
	// use montecarlo if we have it.
	logger := zerolog.Ctx(ctx)

	if !hasSimming(p.botType) {
		return moves[0], nil
	}
	var inferTimeout context.Context
	var cancel context.CancelFunc
	if HasInfer(p.botType) {
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
				time.Duration(5*int(time.Second)))
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

	if HasInfer(p.botType) && len(p.inferencer.Inferences()) > InferencesSimLimit {
		logger.Debug().Int("inferences", len(p.inferencer.Inferences())).Msg("using inferences in sim")
		p.simmer.SetInferences(p.inferencer.Inferences(), montecarlo.InferenceCycle)
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
	logger.Debug().Interface("winning-move", play.Move().String()).Msg("sim-done")
	return play.Move(), nil

}
