package bot

import (
	"context"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/alphabeta"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/turnplayer"
)

// ElitePlayer uses Monte Carlo simulations to rank plays, plays an endgame,
// a pre-endgame (when ready).
type ElitePlayer struct {
	BotTurnPlayer
	endgamer *alphabeta.Solver
}

func NewElitePlayer(conf *config.Config, opts *turnplayer.GameOptions, players []*pb.PlayerInfo,
	botType pb.BotRequest_BotCode) (*ElitePlayer, error) {
	endgameSolver := &alphabeta.Solver{}

	btp, err := NewBotTurnPlayer(conf, opts, players, botType)
	if err != nil {
		return nil, err
	}
	return &ElitePlayer{
		BotTurnPlayer: *btp,
		endgamer:      endgameSolver,
	}, nil
}

// BestPlay picks the highest play by win percentage. It uses montecarlo
// and some other smart things to figure it out.
func (p *ElitePlayer) BestPlay(ctx context.Context) (*move.Move, error) {

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
	endgamePlies := 0
	simPlies := 0
	if unseen <= 7 && tr > 0 {
		log.Debug().Msg("assigning all unseen to opp")
		// bag is actually empty. Assign all of unseen to the opponent.
		mls := make([]alphabet.MachineLetter, tr)
		p.Game.Bag().Draw(tr, mls)
		for _, t := range mls {
			p.Game.RackFor(p.Game.NextPlayer()).Add(t)
		}
		useEndgame = true
		// Just some sort of estimate
		endgamePlies = unseen + int(p.Game.RackFor(p.Game.PlayerOnTurn()).NumTiles())
	} else if unseen > 7 && unseen <= 14 {
		// at some point check for the specific case of 1 or 2 PEG when
		// the code is ready.
		moves = p.GenerateMoves(100)
		simPlies = unseen
	} else {
		moves = p.GenerateMoves(50)
		simPlies = 2
	}
	log.Debug().Msgf("simplies %v", simPlies)

	if useEndgame {
		gd, err := gaddag.Get(p.Game.Config(), p.Game.LexiconName())
		if err != nil {
			return nil, err
		}
		p.Game.SetBackupMode(game.SimulationMode)
		p.Game.SetStateStackLength(endgamePlies)
		gen1 := movegen.NewGordonGenerator(gd, p.Game.Board(), p.Game.Rules().LetterDistribution())
		gen2 := movegen.NewGordonGenerator(gd, p.Game.Board(), p.Game.Rules().LetterDistribution())
		err = p.endgamer.Init(gen1, gen2, p.Game, p.Game.Config())
		if err != nil {
			return nil, err
		}
		v, seq, err := p.endgamer.Solve(ctx, endgamePlies)
		if err != nil {
			return nil, err
		}
		log.Debug().Msgf("best endgame val: %f", v)
		return seq[0], nil
	} else {
		// use montecarlo.
		c, err := equity.NewCombinedStaticCalculator(
			p.LexiconName(), p.Config(), equity.LeaveFilename, equity.PEGAdjustmentFilename)
		if err != nil {
			return nil, err
		}
		simmer := &montecarlo.Simmer{}
		simmer.Init(p.Game, []equity.EquityCalculator{c}, c, p.Config())
		simmer.PrepareSim(simPlies, moves)
		simmer.SetStoppingCondition(montecarlo.Stop99)
		// Simulate is a blocking play:
		err = simmer.Simulate(ctx)
		if err != nil {
			return nil, err
		}
		plays := simmer.WinningPlays()
		return plays[0].Move(), nil
	}

}

// TopPlays sorts the plays by equity and returns the top N. It assumes
// that the equities have already been assigned.
// func (p *ElitePlayer) TopPlays(ctx context.Context, moves []*move.Move, n int) []*move.Move {
// 	sort.Slice(moves, func(i, j int) bool {
// 		return moves[j].Equity() < moves[i].Equity()
// 	})
// 	if n > len(moves) {
// 		n = len(moves)
// 	}
// 	return moves[:n]
// }

// func (p *ElitePlayer) GetBotType() pb.BotRequest_BotCode {
// 	return p.botType
// }

// func (p *ElitePlayer) SetGame(g *game.Game) {
// 	p.game = g
// }

// func (p *ElitePlayer) SetMovegen(g movegen.MoveGenerator) {
// 	p.movegen = g
// }

// func (p *ElitePlayer) Movegen() movegen.MoveGenerator {
// 	return p.movegen
// }
