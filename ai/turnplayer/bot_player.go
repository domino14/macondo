package turnplayer

import (
	"sort"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/turnplayer"
)

type BotTurnPlayer struct {
	AIStaticTurnPlayer
	botType pb.BotRequest_BotCode
}

func NewBotTurnPlayer(conf *config.Config, opts *turnplayer.GameOptions,
	players []*pb.PlayerInfo, botType pb.BotRequest_BotCode) (*BotTurnPlayer, error) {

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
	return addBotFields(p, conf, botType)
}

func addBotFields(p *turnplayer.BaseTurnPlayer, conf *config.Config, botType pb.BotRequest_BotCode) (*BotTurnPlayer, error) {
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

	aip, err := addAIFields(p, conf, calculators)
	if err != nil {
		return nil, err
	}
	return &BotTurnPlayer{
		AIStaticTurnPlayer: *aip,
		botType:            botType,
	}, nil
}

func (p *BotTurnPlayer) GenerateMoves(numPlays int) []*move.Move {
	curRack := p.RackFor(p.PlayerOnTurn())
	oppRack := p.RackFor(p.NextPlayer())

	p.gen.GenAll(curRack, p.Bag().TilesRemaining() >= game.ExchangeLimit)

	plays := p.gen.Plays()

	p.AssignEquity(plays, p.Board(), p.Bag(), oppRack)
	if numPlays == 1 {
		// Plays aren't sorted yet
		sort.Slice(plays, func(i, j int) bool {
			return plays[i].Equity() > plays[j].Equity()
		})
		return []*move.Move{filter(p.cfg, &p.Game, curRack, plays, p.botType)}
	}

	return p.TopPlays(plays, numPlays)
}

func (p *BotTurnPlayer) SetEquityCalculators(calcs []equity.EquityCalculator) {
	p.calculators = calcs
}

func (p *BotTurnPlayer) GetBotType() pb.BotRequest_BotCode {
	return p.botType
}

func (p *BotTurnPlayer) SetBotType(b pb.BotRequest_BotCode) {
	p.botType = b
}
