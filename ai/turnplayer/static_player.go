package turnplayer

import (
	"sort"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/samber/lo"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/turnplayer"
)

type AIStaticTurnPlayer struct {
	turnplayer.BaseTurnPlayer

	calculators []equity.EquityCalculator

	gen movegen.MoveGenerator
	cfg *config.Config
}

func NewAIStaticTurnPlayer(conf *config.Config, opts *turnplayer.GameOptions,
	players []*pb.PlayerInfo, botType pb.BotRequest_BotCode) (*AIStaticTurnPlayer, error) {

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
	return addAIFields(p, conf, botType)
}

func addAIFields(p *turnplayer.BaseTurnPlayer, conf *config.Config, botType pb.BotRequest_BotCode) (*AIStaticTurnPlayer, error) {
	calculators := []equity.EquityCalculator{}

	if botType == pb.BotRequest_SIMMING_BOT {

	} else {
		c1, err := equity.NewExhaustiveLeaveCalculator(p.LexiconName(), conf, equity.LeaveFilename)
		if err != nil {
			return nil, err
		}
		c2 := &equity.OpeningAdjustmentCalculator{}
		c3, err := equity.NewPreEndgameAdjustmentCalculator(conf, p.LexiconName(), equity.PEGAdjustmentFilename)
		if err != nil {
			return nil, err
		}
		c4 := &equity.EndgameAdjustmentCalculator{}
		calculators = []equity.EquityCalculator{c1, c2, c3, c4}
	}

	gd, err := gaddag.Get(conf, p.LexiconName())
	if err != nil {
		return nil, err
	}
	gen := movegen.NewGordonGenerator(gd, p.Board(), p.Bag().LetterDistribution())
	ret := &AIStaticTurnPlayer{*p, calculators, gen, conf}
	return ret, nil
}

func (p *AIStaticTurnPlayer) AssignEquity(plays []*move.Move, board *board.GameBoard, bag *alphabet.Bag, oppRack *alphabet.Rack) {
	for _, m := range plays {
		m.SetEquity(lo.SumBy(p.calculators, func(c equity.EquityCalculator) float64 {
			return c.Equity(m, board, bag, oppRack)
		}))
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
