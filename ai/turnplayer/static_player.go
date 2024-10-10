package turnplayer

import (
	"context"
	"sort"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/samber/lo"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/turnplayer"
)

// StaticPassPenalty is a penalty for passing without making a move.
const StaticPassPenalty = -50.0

type AIStaticTurnPlayer struct {
	turnplayer.BaseTurnPlayer

	calculators []equity.EquityCalculator

	gen movegen.MoveGenerator
	cfg *config.Config
}

func NewAIStaticTurnPlayer(conf *config.Config, opts *turnplayer.GameOptions,
	players []*pb.PlayerInfo, calculators []equity.EquityCalculator) (*AIStaticTurnPlayer, error) {

	opts.SetDefaults(conf)
	rules, err := game.NewBasicGameRules(conf, opts.Lexicon.Name, opts.BoardLayoutName,
		opts.Lexicon.Distribution, game.CrossScoreAndSet, opts.Variant)

	if err != nil {
		return nil, err
	}
	p, err := turnplayer.BaseTurnPlayerFromRules(opts, players, rules)
	if err != nil {
		return nil, err
	}
	return AddAIFields(p, conf, calculators)
}

func NewAIStaticTurnPlayerFromGame(g *game.Game, conf *config.Config, calculators []equity.EquityCalculator) (*AIStaticTurnPlayer, error) {
	gr := &turnplayer.BaseTurnPlayer{Game: g}
	return AddAIFields(gr, conf, calculators)
}

func AddAIFields(p *turnplayer.BaseTurnPlayer, conf *config.Config, calculators []equity.EquityCalculator) (*AIStaticTurnPlayer, error) {
	gd, err := kwg.Get(conf.WGLConfig(), p.LexiconName())
	if err != nil {
		return nil, err
	}
	gen := movegen.NewGordonGenerator(gd, p.Board(), p.Bag().LetterDistribution())
	gen.SetEquityCalculators(calculators)
	ret := &AIStaticTurnPlayer{*p, calculators, gen, conf}
	return ret, nil
}

func (p *AIStaticTurnPlayer) AssignEquity(plays []*move.Move, board *board.GameBoard, bag *tilemapping.Bag, oppRack *tilemapping.Rack) {
	for _, m := range plays {
		if m.Action() == move.MoveTypePass {
			m.SetEquity(StaticPassPenalty)
		} else {
			m.SetEquity(lo.SumBy(p.calculators, func(c equity.EquityCalculator) float64 {
				return c.Equity(m, board, bag, oppRack)
			}))
		}
	}
}

func (p *AIStaticTurnPlayer) TopPlays(plays []*move.Move, ct int) []*move.Move {
	sort.Slice(plays, func(i, j int) bool {
		return plays[i].Equity() > plays[j].Equity()
	})
	if ct > len(plays) {
		ct = len(plays)
	}
	return plays[:ct]
}

func (p *AIStaticTurnPlayer) BestPlay(ctx context.Context) (*move.Move, error) {
	return p.GenerateMoves(1)[0], nil
}

func (p *AIStaticTurnPlayer) GenerateMoves(numPlays int) []*move.Move {
	curRack := p.RackFor(p.PlayerOnTurn())
	oppRack := p.RackFor(p.NextPlayer())
	// in case we don't have full rack info:
	unseen := int(oppRack.NumTiles()) + p.Bag().TilesRemaining()
	exchAllowed := unseen-game.RackTileLimit >= p.Rules().ExchangeLimit()
	p.gen.SetMaxCanExchange(game.MaxCanExchange(unseen-game.RackTileLimit, p.Rules().ExchangeLimit()))
	p.gen.GenAll(curRack, exchAllowed)

	plays := p.gen.(*movegen.GordonGenerator).Plays()
	p.AssignEquity(plays, p.Board(), p.Bag(), oppRack)
	return p.TopPlays(plays, numPlays)
}

func (p *AIStaticTurnPlayer) MoveGenerator() movegen.MoveGenerator {
	return p.gen
}

func (p *AIStaticTurnPlayer) Calculators() []equity.EquityCalculator {
	return p.calculators
}

func (p *AIStaticTurnPlayer) SetCalculators(c []equity.EquityCalculator) {
	p.calculators = c
}

func (p *AIStaticTurnPlayer) GetBotType() pb.BotRequest_BotCode {
	// No bot associated with just a plain AIStaticTurnPlayer
	return pb.BotRequest_UNKNOWN
}
