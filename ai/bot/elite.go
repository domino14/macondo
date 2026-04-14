package bot

import (
	"context"
	"errors"
	"math"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

const InferencesSimLimit = 2

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
		Bool("stochasticStaticEval", p.cfg.StochasticStaticEval).
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
	p.endgamer.SetThreads(runtime.NumCPU())
	// Use auto mode: picks ABDADA for 300+ moves, otherwise LazySMP
	p.endgamer.SetParallelAlgorithm(negamax.ParallelAlgoAuto)
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
	if !HasPreendgame(p.botType) {
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

func nonEndgameBest(ctx context.Context, p *BotTurnPlayer, simPlies int, moves []*move.Move) (*move.Move, error) {
	// use montecarlo if we have it.
	logger := zerolog.Ctx(ctx)

	if !hasSimming(p.botType) {
		return moves[0], nil
	}
	var inferTimeout context.Context
	var cancel context.CancelFunc
	inferenceRan := false
	if HasInfer(p.botType) && p.Game.Bag().TilesRemaining() > 0 {
		logger.Debug().Msg("running inference..")
		p.inferencer.Init(p.Game, p.simmerCalcs, p.Config())
		if p.simThreads != 0 {
			p.inferencer.SetThreads(p.simThreads)
		}
		if p.cfg.InferenceTau > 0 {
			p.inferencer.SetTau(p.cfg.InferenceTau)
		}
		if p.cfg.InferenceSimIters > 0 {
			p.inferencer.SetSimIters(p.cfg.InferenceSimIters)
		}
		err := p.inferencer.PrepareFinder(p.Game.RackFor(p.Game.PlayerOnTurn()).TilesOn())
		if err != nil {
			// Expected early in the game (no events yet, bingo, etc.)
			logger.Debug().AnErr("inference-prepare-error", err).Msg("inference-skipped")
		} else {
			inferTimeSecs := 20
			if p.cfg.InferenceTimeSecs > 0 {
				inferTimeSecs = p.cfg.InferenceTimeSecs
			}
			logger.Info().Float64("tau", p.inferencer.Tau()).
				Int("timeSecs", inferTimeSecs).
				Int("simIters", p.inferencer.SimIters()).
				Msg("inference-tau")
			inferTimeout, cancel = context.WithTimeout(context.Background(),
				time.Duration(inferTimeSecs)*time.Second)
			defer cancel()
			err = p.inferencer.Infer(inferTimeout)
			if err != nil {
				// ignore all errors and move on.
				logger.Debug().AnErr("inference-error", err).Msg("probably-ok")
			}
			inferenceRan = true
		}
	}

	p.simmer.Init(p.Game, p.simmerCalcs, p.simmerCalcs[0].(*equity.CombinedStaticCalculator), p.Config())
	p.simmer.TryLoadWMP(p.Config().WGLConfig(), p.Game.LexiconName())
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

	if HasInfer(p.botType) && inferenceRan {
		nInferred := len(p.inferencer.Inferences().InferredRacks)
		if nInferred > InferencesSimLimit {
			logger.Info().Int("inferences", nInferred).Msg("using inferences in sim")
			p.simmer.SetInferences(p.inferencer.Inferences().InferredRacks, p.inferencer.Inferences().RackLength, montecarlo.InferenceWeightedRandomRacks)
		} else {
			logger.Info().Int("inferences", nInferred).Msg("too few inferences, skipping")
		}
	}
	if p.cfg.UseOppRacksInAnalysis {
		oppRack := p.Game.RackFor(p.Game.NextPlayer())
		logger.Info().Str("rack", oppRack.String()).Msg("setting-known-opp-rack")
		p.simmer.SetKnownOppRack(oppRack.TilesOn())
	}
	if p.cfg.OracleInference && p.Game.Bag().TilesRemaining() > 0 {
		leave, err := extractTrueLeave(p.Game)
		if err == nil {
			logger.Info().Str("leave", tilemapping.MachineWord(leave).UserVisible(p.Game.Alphabet())).Msg("oracle-inference")
			p.simmer.SetKnownOppRack(leave)
		} else {
			logger.Debug().AnErr("oracle-inference-err", err).Msg("oracle-inference-skipped")
		}
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

// extractTrueLeave finds the opponent's last tile-placement or exchange event
// in the game history and returns the leave they held after playing.
// Used by the oracle inference mode to establish an upper bound for inference.
//
// We do NOT use game.MoveFromEvent here because that function calls
// modifyForPlaythrough against the current board, which by this point already
// has those tiles on it — causing every tile to be flagged as a play-through
// (set to 0) and the computed leave to equal the full rack.
// Instead we parse the leave directly from the event fields without touching
// the board: strip '.' play-through markers from PlayedTiles (GCG notation),
// leaving only the tiles that came from the rack.
func extractTrueLeave(g *game.Game) ([]tilemapping.MachineLetter, error) {
	evts := g.History().Events[:g.Turn()]
	if len(evts) == 0 {
		return nil, errors.New("no events")
	}
	oppEvtIdx := len(evts) - 1
	oppIdx := evts[oppEvtIdx].PlayerIndex
	for oppEvtIdx >= 0 {
		evt := evts[oppEvtIdx]
		if evt.PlayerIndex != oppIdx {
			break
		}
		if evt.Type == pb.GameEvent_CHALLENGE_BONUS {
			oppEvtIdx--
			continue
		}
		if evt.Type == pb.GameEvent_TILE_PLACEMENT_MOVE {
			rack, err := tilemapping.ToMachineWord(evt.Rack, g.Alphabet())
			if err != nil {
				return nil, err
			}
			// Strip '.' (play-through markers in GCG notation) so we're left
			// with only the tiles that came from the rack.
			playedStr := strings.ReplaceAll(evt.PlayedTiles, ".", "")
			played, err := tilemapping.ToMachineWord(playedStr, g.Alphabet())
			if err != nil {
				return nil, err
			}
			leave, err := tilemapping.Leave(rack, played, true)
			if err != nil {
				return nil, err
			}
			return []tilemapping.MachineLetter(leave), nil
		}
		if evt.Type == pb.GameEvent_EXCHANGE {
			rack, err := tilemapping.ToMachineWord(evt.Rack, g.Alphabet())
			if err != nil {
				return nil, err
			}
			// If only a count was stored we can't determine which tiles were
			// exchanged, so we can't compute the leave.
			if _, err := strconv.Atoi(evt.Exchanged); err == nil {
				return nil, errors.New("exchange event only stores tile count, not tiles")
			}
			exchanged, err := tilemapping.ToMachineWord(evt.Exchanged, g.Alphabet())
			if err != nil {
				return nil, err
			}
			leave, err := tilemapping.Leave(rack, exchanged, false)
			if err != nil {
				return nil, err
			}
			return []tilemapping.MachineLetter(leave), nil
		}
		oppEvtIdx--
	}
	return nil, errors.New("no opponent tile-placement or exchange event found")
}
