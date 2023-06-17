package bot

import (
	"context"
	"sort"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/rangefinder"
	"github.com/domino14/macondo/turnplayer"
)

type BotConfig struct {
	config.Config
	PEGAdjustmentFile string
	LeavesFile        string
	MinSimPlies       int
}

type BotTurnPlayer struct {
	aiturnplayer.AIStaticTurnPlayer
	botType     pb.BotRequest_BotCode
	endgamer    *negamax.Solver
	simmer      *montecarlo.Simmer
	simmerCalcs []equity.EquityCalculator
	simThreads  int
	minSimPlies int

	inferencer *rangefinder.RangeFinder
}

func NewBotTurnPlayer(conf *BotConfig, opts *turnplayer.GameOptions,
	players []*pb.PlayerInfo, botType pb.BotRequest_BotCode) (*BotTurnPlayer, error) {

	opts.SetDefaults(&conf.Config)
	rules, err := game.NewBasicGameRules(&conf.Config, opts.Lexicon.Name, opts.BoardLayoutName,
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

func NewBotTurnPlayerFromGame(g *game.Game, conf *BotConfig, botType pb.BotRequest_BotCode) (*BotTurnPlayer, error) {
	gr := &turnplayer.BaseTurnPlayer{Game: g}
	return addBotFields(gr, conf, botType)
}

func addBotFields(p *turnplayer.BaseTurnPlayer, conf *BotConfig, botType pb.BotRequest_BotCode) (*BotTurnPlayer, error) {
	var calculators []equity.EquityCalculator

	if botType == pb.BotRequest_NO_LEAVE_BOT {
		calculators = []equity.EquityCalculator{equity.NewNoLeaveCalculator()}
	} else {
		c1, err := equity.NewExhaustiveLeaveCalculator(
			p.LexiconName(), &conf.Config, conf.LeavesFile)
		if err != nil {
			return nil, err
		}
		c2 := &equity.OpeningAdjustmentCalculator{}
		c3, err := equity.NewPreEndgameAdjustmentCalculator(
			&conf.Config, p.LexiconName(), conf.PEGAdjustmentFile)
		if err != nil {
			return nil, err
		}
		c4 := &equity.EndgameAdjustmentCalculator{}
		calculators = []equity.EquityCalculator{c1, c2, c3, c4}
	}
	aip, err := aiturnplayer.AddAIFields(p, &conf.Config, calculators)
	if err != nil {
		return nil, err
	}
	btp := &BotTurnPlayer{
		AIStaticTurnPlayer: *aip,
		botType:            botType,
	}

	// If it is a simming bot, add more fields.
	if hasSimming(botType) {
		c, err := equity.NewCombinedStaticCalculator(
			p.LexiconName(), p.Config(), "", equity.PEGAdjustmentFilename)
		if err != nil {
			return nil, err
		}
		btp.simmer = &montecarlo.Simmer{}
		btp.simmerCalcs = []equity.EquityCalculator{c}
		if conf.MinSimPlies > 0 {
			btp.SetMinSimPlies(conf.MinSimPlies)
		}
	}
	if hasEndgame(botType) {
		btp.endgamer = &negamax.Solver{}
	}
	if HasInfer(botType) {
		btp.inferencer = &rangefinder.RangeFinder{}
	}

	return btp, nil
}

func (p *BotTurnPlayer) GenerateMoves(numPlays int) []*move.Move {
	curRack := p.RackFor(p.PlayerOnTurn())
	oppRack := p.RackFor(p.NextPlayer())
	gen := p.MoveGenerator()
	// re-set back to all-plays recorder, in case the movegen was re-used
	// by the endgame engine that only uses "minimal moves".
	// XXX: this is ugly, do we even need minimal moves?
	gen.SetGenPass(false)
	gen.SetPlayRecorder(movegen.AllPlaysRecorder)
	gen.GenAll(curRack, p.Bag().TilesRemaining() >= game.ExchangeLimit)

	plays := gen.(*movegen.GordonGenerator).Plays()

	p.AssignEquity(plays, p.Board(), p.Bag(), oppRack)
	if numPlays == 1 {
		// Plays aren't sorted yet
		sort.Slice(plays, func(i, j int) bool {
			return plays[i].Equity() > plays[j].Equity()
		})
		return []*move.Move{filter(p.Config(), p.Game, curRack, plays, p.botType)}
	}

	return p.TopPlays(plays, numPlays)
}

func (p *BotTurnPlayer) BestPlay(ctx context.Context) (*move.Move, error) {
	if hasSimming(p.botType) || hasEndgame(p.botType) || HasInfer(p.botType) {
		return eliteBestPlay(ctx, p)
	}
	return p.GenerateMoves(1)[0], nil
}

func (p *BotTurnPlayer) SetEquityCalculators(calcs []equity.EquityCalculator) {
	p.SetCalculators(calcs)
}

func (p *BotTurnPlayer) GetBotType() pb.BotRequest_BotCode {
	return p.botType
}

func (p *BotTurnPlayer) SetBotType(b pb.BotRequest_BotCode) {
	p.botType = b
}

func (p *BotTurnPlayer) SetSimThreads(t int) {
	p.simThreads = t
}

func (p *BotTurnPlayer) SetMinSimPlies(t int) {
	p.minSimPlies = t
}
