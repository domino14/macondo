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

// miniSimMinSamples is how many rollouts every play gets before a mini-sim
// may prefer one or stop: the 128 iterations the old auto-stopper waited for
// before pruning. MAGPIE's 500 would exceed the mini-sims' 200-iteration
// budget, leaving them no adaptive sampling at all.
const miniSimMinSamples = 128

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
	btp.maxIters = 200

	return btp, nil
}

func (p *SimpleSimmer) SetLogging(l bool) {
	p.logging = l
}

func (p *SimpleSimmer) SetMaxIters(n int) {
	p.maxIters = n
}

// SetPlies sets how deep the simmer looks (default 2).
func (p *SimpleSimmer) SetPlies(n int) {
	p.numPlies = n
}

func (p *SimpleSimmer) GenerateMoves(numPlays int) []*move.Move {
	return p.AIStaticTurnPlayer.GenerateMoves(numPlays)
}

func (p *SimpleSimmer) BestPlay(ctx context.Context) (*move.Move, error) {
	moves := p.GenerateMoves(20)
	p.simmer.Init(p.Game, p.simmerCalcs, p.simmerCalcs[0].(*equity.CombinedStaticCalculator), p.Config())
	p.simmer.SetThreads(1)
	// 2 plies for this simple simmer. can maybe configure later.
	p.simmer.PrepareSim(p.numPlies, moves)
	p.simmer.SetStoppingCondition(montecarlo.Stop99)
	p.simmer.SetMinSamplesPerPlay(miniSimMinSamples)
	// No bag cycling: a mini-sim's 200 rollouts a play would be only about 15
	// cycled samples, 10 of them the minimum, leaving the top-two sampling
	// almost nothing to allocate.
	p.simmer.SetBagCycle(false)
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
			if moves[i].Equals(addedMove, checktrans, true) {
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
	p.simmer.SetStoppingCondition(montecarlo.Stop99)
	p.simmer.SetMinSamplesPerPlay(miniSimMinSamples)
	p.simmer.SetBagCycle(false)
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
