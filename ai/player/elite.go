package player

import (
	"context"
	"sort"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/endgame/alphabeta"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
)

// SimmingPlayer uses Monte Carlo simulations to rank plays.
type SimmingPlayer struct {
	strategy strategy.Strategizer
	botType  pb.BotRequest_BotCode
	game     *game.Game
	movegen  movegen.MoveGenerator
	endgamer *alphabeta.Solver
}

func NewSimmingPlayer(s strategy.Strategizer, botType pb.BotRequest_BotCode) *SimmingPlayer {
	endgameSolver := &alphabeta.Solver{}
	return &SimmingPlayer{
		strategy: s,
		botType:  botType,
		endgamer: endgameSolver,
	}
}

func (p *SimmingPlayer) Strategizer() strategy.Strategizer {
	return p.strategy
}

// AssignEquity uses the strategizer to assign an equity to every move.
// This is the sole module dedicated to assigning equities. (Perhaps it
// should be named something else?)
func (p *SimmingPlayer) AssignEquity(moves []*move.Move, board *board.GameBoard,
	bag *alphabet.Bag, oppRack *alphabet.Rack) {
	for _, m := range moves {
		m.SetEquity(p.strategy.Equity(m, board, bag, oppRack))
	}
}

// BestPlay picks the highest play by win percentage. It uses montecarlo
// and some other smart things to figure it out.
func (p *SimmingPlayer) BestPlay(ctx context.Context, moves []*move.Move) *move.Move {
	// First determine what stage of the game we are in.
	tr := p.game.Bag().TilesRemaining()
	// We don't necessarily know the number of tiles on our opponent's rack.
	opp := p.game.RackFor(p.game.NextPlayer()).NumTiles()
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
		p.game.Bag().Draw(tr, mls)
		for _, t := range mls {
			p.game.RackFor(p.game.NextPlayer()).Add(t)
		}
		useEndgame = true
		// Just some sort of estimate
		endgamePlies = unseen + int(p.game.RackFor(p.game.PlayerOnTurn()).NumTiles())
	} else if unseen > 7 && unseen <= 14 {
		// at some point check for the specific case of 1 or 2 PEG when
		// the code is ready.
		simPlies = unseen
	} else {
		simPlies = 2
	}
	log.Debug().Msgf("simplies %v", simPlies)

	if useEndgame {
		gd, _ := gaddag.Get(p.game.Config(), p.game.LexiconName())
		p.game.SetBackupMode(game.SimulationMode)
		p.game.SetStateStackLength(endgamePlies)
		gen1 := movegen.NewGordonGenerator(gd, p.game.Board(), p.game.Rules().LetterDistribution())
		gen2 := movegen.NewGordonGenerator(gd, p.game.Board(), p.game.Rules().LetterDistribution())
		p.endgamer.Init(gen1, gen2, p.game, p.game.Config())
	}

	topEquity := -Infinity
	var topMove *move.Move
	for i := 0; i < len(moves); i++ {
		if moves[i].Equity() > topEquity {
			topEquity = moves[i].Equity()
			topMove = moves[i]
		}
	}
	return topMove
}

// TopPlays sorts the plays by equity and returns the top N. It assumes
// that the equities have already been assigned.
func (p *SimmingPlayer) TopPlays(ctx context.Context, moves []*move.Move, n int) []*move.Move {
	sort.Slice(moves, func(i, j int) bool {
		return moves[j].Equity() < moves[i].Equity()
	})
	if n > len(moves) {
		n = len(moves)
	}
	return moves[:n]
}

func (p *SimmingPlayer) GetBotType() pb.BotRequest_BotCode {
	return p.botType
}

func (p *SimmingPlayer) SetGame(g *game.Game) {
	p.game = g
}

func (p *SimmingPlayer) SetMovegen(g movegen.MoveGenerator) {
	p.movegen = g
}

func (p *SimmingPlayer) Movegen() movegen.MoveGenerator {
	return p.movegen
}
