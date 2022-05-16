package runner

import (
	"sort"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/runner"
	"github.com/domino14/macondo/strategy"
	"github.com/rs/zerolog/log"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// Game with an AI player available for move generation.
type AIGameRunner struct {
	runner.GameRunner

	aiplayer player.AIPlayer
	gen      movegen.MoveGenerator
	cfg      *config.Config
}

func NewAIGameRunner(conf *config.Config, opts *runner.GameOptions, players []*pb.PlayerInfo, botType pb.BotRequest_BotCode) (*AIGameRunner, error) {
	opts.SetDefaults(conf)
	rules, err := NewAIGameRules(
		conf, opts.BoardLayoutName, opts.Lexicon.Name, opts.Lexicon.Distribution)
	if err != nil {
		return nil, err
	}
	g, err := runner.NewGameRunnerFromRules(opts, players, rules)
	if err != nil {
		return nil, err
	}
	return addAIFields(g, conf, botType)
}

func NewAIGameRunnerFromGame(g *game.Game, conf *config.Config, botType pb.BotRequest_BotCode) (*AIGameRunner, error) {
	gr := runner.GameRunner{*g}
	return addAIFields(&gr, conf, botType)
}

func addAIFields(g *runner.GameRunner, conf *config.Config, botType pb.BotRequest_BotCode) (*AIGameRunner, error) {
	strategy, err := strategy.NewExhaustiveLeaveStrategy(
		g.LexiconName(),
		g.Alphabet(),
		conf,
		strategy.LeaveFilename,
		strategy.PEGAdjustmentFilename)
	if err != nil {
		return nil, err
	}

	gd, err := gaddag.Get(conf, g.LexiconName())
	if err != nil {
		return nil, err
	}

	aiplayer := player.NewRawEquityPlayer(strategy, botType)
	gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())

	ret := &AIGameRunner{*g, aiplayer, gen, conf}
	return ret, nil
}

func (g *AIGameRunner) MoveGenerator() movegen.MoveGenerator {
	return g.gen
}

func (g *AIGameRunner) AssignEquity(plays []*move.Move, oppRack *alphabet.Rack) {
	g.aiplayer.AssignEquity(plays, g.Board(), g.Bag(), oppRack)
}

func (g *AIGameRunner) AIPlayer() player.AIPlayer {
	return g.aiplayer
}

func NewAIGameRules(cfg *config.Config, boardLayoutName string,
	lexiconName string, letterDistributionName string) (*game.GameRules, error) {

	// assume bot can only play classic for now. Modify if we can teach this
	// bot to play other variants.
	return game.NewBasicGameRules(
		cfg, lexiconName, boardLayoutName, letterDistributionName,
		game.CrossScoreAndSet,
		game.VarClassic)
}

func (g *AIGameRunner) GenerateMoves(numPlays int) []*move.Move {
	return GenerateMoves(&g.Game, g.aiplayer, g.gen, g.cfg, numPlays)
}

func GenerateMoves(g *game.Game, aiplayer player.AIPlayer, gen movegen.MoveGenerator,
	cfg *config.Config, numPlays int) []*move.Move {
	curRack := g.RackFor(g.PlayerOnTurn())
	oppRack := g.RackFor(g.NextPlayer())

	gen.GenAll(curRack, g.Bag().TilesRemaining() >= 7)

	plays := gen.Plays()

	aiplayer.AssignEquity(plays, g.Board(), g.Bag(), oppRack)
	if numPlays == 1 {
		// Plays aren't sorted yet
		sort.Slice(plays, func(i, j int) bool {
			return plays[j].Equity() < plays[i].Equity()
		})
		log.Debug().Msgf("botType: %v", aiplayer.GetBotType().String())
		return []*move.Move{filter(cfg, g, curRack, plays, aiplayer.GetBotType())}
	}

	return aiplayer.TopPlays(plays, numPlays)
}
