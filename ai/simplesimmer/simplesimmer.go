package simplesimmer

import (
	"context"
	"os"
	"time"

	"github.com/rs/zerolog/log"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// A SimpleSimmer is a simple wrapper to allow a simple MonteCarlo simming player
// to work and be used elsewhere in the code. Maybe it can be extended later.
type SimpleSimmer struct {
	aiturnplayer.AIStaticTurnPlayer
	simmer      *montecarlo.Simmer
	simmerCalcs []equity.EquityCalculator
	numPlies    int
	maxIters    int
	logging     bool
}

func NewSimpleSimmerFromGame(g *game.Game) (*SimpleSimmer, error) {
	btp := &SimpleSimmer{}

	log.Debug().Msg("adding fields for simmer")
	leaveFile := "" // use default
	if g.Rules().BoardName() == board.SuperCrosswordGameLayout {
		leaveFile = "super-leaves.klv2"
	}
	c, err := equity.NewCombinedStaticCalculator(
		g.LexiconName(), g.Config(), leaveFile, equity.PEGAdjustmentFilename)
	if err != nil {
		return nil, err
	}
	btp.Game = g
	btp.simmer = &montecarlo.Simmer{}
	btp.simmerCalcs = []equity.EquityCalculator{c}

	aip, err := aiturnplayer.AddAIFields(&btp.BaseTurnPlayer, g.Config(), []equity.EquityCalculator{c})
	if err != nil {
		return nil, err
	}
	btp.AIStaticTurnPlayer = *aip

	btp.numPlies = 2
	btp.maxIters = 100

	return btp, nil
}

func (p *SimpleSimmer) SetLogging(l bool) {
	p.logging = l
}

func (p *SimpleSimmer) GenerateMoves(numPlays int) []*move.Move {
	curRack := p.RackFor(p.PlayerOnTurn())
	oppRack := p.RackFor(p.NextPlayer())
	gen := p.MoveGenerator()
	// in case we don't have full rack info:
	unseen := int(oppRack.NumTiles()) + p.Bag().TilesRemaining()
	exchAllowed := unseen-game.RackTileLimit >= p.ExchangeLimit()
	gen.SetMaxCanExchange(game.MaxCanExchange(unseen-game.RackTileLimit, p.ExchangeLimit()))
	gen.GenAll(curRack, exchAllowed)
	plays := gen.(*movegen.GordonGenerator).Plays()

	p.AssignEquity(plays, p.Board(), p.Bag(), oppRack)
	return p.TopPlays(plays, numPlays)
}

func (p *SimpleSimmer) BestPlay(ctx context.Context) (*move.Move, error) {
	moves := p.GenerateMoves(20)
	p.simmer.Init(p.Game, p.simmerCalcs, p.simmerCalcs[0].(*equity.CombinedStaticCalculator), p.Config())
	p.simmer.SetThreads(1)
	// 2 plies for this simple simmer. can maybe configure later.
	p.simmer.PrepareSim(p.numPlies, moves)
	p.simmer.SetAutostopCheckInterval(64)
	p.simmer.SetStoppingCondition(montecarlo.Stop99)
	p.simmer.SimSingleThread(p.maxIters, p.numPlies) // can also configure later.
	wp := p.simmer.WinningPlay()
	return wp.Move(), nil
}

func (p *SimpleSimmer) GenAndSim(ctx context.Context, nMoves int, addedMove *move.Move) (string, error) {
	t := time.Now()
	moves := p.GenerateMoves(nMoves)
	if addedMove != nil {
		log.Debug().Str("adding-move", addedMove.ShortDescription()).Msg("in-gen-and-sim")
		moveAlreadyThere := false
		for i := range moves {
			checktrans := p.Game.Board().IsEmpty()
			if moves[i].Equals(addedMove, checktrans, false) {
				moveAlreadyThere = true
				break
			}
		}
		if !moveAlreadyThere {
			moves = append(moves, addedMove)
		}
	}
	p.simmer.Init(p.Game, p.simmerCalcs, p.simmerCalcs[0].(*equity.CombinedStaticCalculator), p.Config())
	p.simmer.SetThreads(1)
	p.simmer.PrepareSim(p.numPlies, moves)
	p.simmer.SetAutostopCheckInterval(64)
	p.simmer.SetStoppingCondition(montecarlo.Stop99)
	var logfile *os.File
	var err error
	var filename string
	if p.logging {
		logfile, err = os.CreateTemp("", "")
		if err != nil {
			return "", err
		}
		p.simmer.SetLogStream(logfile)
		filename = logfile.Name()
	}

	p.simmer.SimSingleThread(p.maxIters, p.numPlies) // can also configure later.
	if p.logging {
		logfile.Close()
	}
	log.Debug().Dur("t", time.Since(t)).Msg("time-elapsed-after-sim")
	return filename, nil
}

func (p *SimpleSimmer) BestPlays() *montecarlo.SimmedPlays {
	return p.simmer.PlaysByWinProb()
}
