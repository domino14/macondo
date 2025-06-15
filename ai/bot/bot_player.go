package bot

import (
	"context"
	"errors"
	"sort"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/preendgame"
	"github.com/domino14/macondo/rangefinder"
	"github.com/domino14/macondo/turnplayer"
	"github.com/rs/zerolog/log"
)

type BotConfig struct {
	config.Config
	PEGAdjustmentFile    string
	LeavesFile           string
	MinSimPlies          int
	StochasticStaticEval bool
	// If UseOppRacksInAnalysis is true, will use opponent rack info for simulation/pre-endgames/etc
	UseOppRacksInAnalysis bool
}

type BotTurnPlayer struct {
	aiturnplayer.AIStaticTurnPlayer
	botType     pb.BotRequest_BotCode
	endgamer    *negamax.Solver
	preendgamer *preendgame.Solver
	simmer      *montecarlo.Simmer
	simmerCalcs []equity.EquityCalculator
	simThreads  int
	minSimPlies int
	cfg         *BotConfig

	inferencer            *rangefinder.RangeFinder
	lastCalculatedDetails string
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
		cfg:                conf,
	}

	// If it is a simming bot, add more fields.
	if hasSimming(botType) {
		log.Info().Msg("adding fields for simmer")
		leaveFile := "" // use default
		if p.Rules().BoardName() == board.SuperCrosswordGameLayout {
			leaveFile = "super-leaves.klv2"
		}
		c, err := equity.NewCombinedStaticCalculator(
			p.LexiconName(), p.Config(), leaveFile, equity.PEGAdjustmentFilename)
		if err != nil {
			return nil, err
		}
		btp.simmer = &montecarlo.Simmer{}
		btp.simmerCalcs = []equity.EquityCalculator{c}
		if conf.MinSimPlies > 0 {
			btp.SetMinSimPlies(conf.MinSimPlies)
		}
	}
	if HasEndgame(botType) {
		log.Info().Msg("adding fields for endgame")
		btp.endgamer = &negamax.Solver{}
	}
	if HasPreendgame(botType) {
		log.Info().Msg("adding fields for pre-endgame")
		btp.preendgamer = &preendgame.Solver{}
	}
	if HasInfer(botType) {
		log.Info().Msg("adding fields for rangefinder")
		btp.inferencer = &rangefinder.RangeFinder{}
	}

	return btp, nil
}

func (p *BotTurnPlayer) GenerateMoves(numPlays int) []*move.Move {
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
	if numPlays == 1 {
		// Plays aren't sorted yet
		sort.Slice(plays, func(i, j int) bool {
			return plays[i].Equity() > plays[j].Equity()
		})
		return []*move.Move{filter(p.Config(), p.Game, curRack, plays, p.botType, p.LexiconName())}
	}

	return p.TopPlays(plays, numPlays)
}

type moveEval struct {
	move *move.Move
	eval float32
	idx  int
}

func (p *BotTurnPlayer) BestPlay(ctx context.Context) (*move.Move, error) {
	if hasSimming(p.botType) || HasEndgame(p.botType) || HasInfer(p.botType) || HasPreendgame(p.botType) {
		return eliteBestPlay(ctx, p)
	}
	if p.botType == pb.BotRequest_FAST_ML_BOT {
		// Fast ML bot uses a different method
		moves := p.GenerateMoves(50)

		if len(moves) == 1 {
			return moves[0], nil
		}
		var lc *equity.ExhaustiveLeaveCalculator
		for _, c := range p.Calculators() {
			if e, ok := c.(*equity.ExhaustiveLeaveCalculator); ok {
				lc = e
				break
			}
		}
		if lc == nil {
			return nil, errors.New("no ExhaustiveLeaveCalculator found for fast ML bot")
		}

		resp, err := p.MLEvaluateMoves(moves, lc)
		if err != nil {
			log.Error().Err(err).Msg("Failed to evaluate moves for fast ML bot")
			return nil, err
		}
		pairs := make([]moveEval, len(moves))
		for i, m := range moves {
			pairs[i] = moveEval{
				move: m,
				eval: resp[i],
				idx:  i + 1, // Store original index for reference
			}
		}
		// Sort by evaluation in descending order
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].eval > pairs[j].eval
		})
		return pairs[0].move, nil
	}
	return p.GenerateMoves(1)[0], nil
}

// Returns a string summary of details from a previous call to BestPlay.
func (p *BotTurnPlayer) BestPlayDetails(ctx context.Context) string {
	if hasSimming(p.botType) || HasEndgame(p.botType) || HasInfer(p.botType) || HasPreendgame(p.botType) {
		return p.lastCalculatedDetails
	} else {
		return "(No summary)"
	}
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
