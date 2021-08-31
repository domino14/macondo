package runner

import (
	"sort"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/runner"
	"github.com/domino14/macondo/strategy"

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
		conf, board.CrosswordGameLayout,
		opts.Lexicon.Name, opts.Lexicon.Distribution)
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

	// Assign equity to plays, and return the top ones.
	aiplayer.AssignEquity(plays, g.Board(), g.Bag(), oppRack)

	if numPlays == 1 && aiplayer.GetBotType() != pb.BotRequest_HASTY_BOT {
		// Plays aren't sorted yet
		sort.Slice(plays, func(i, j int) bool {
			return plays[j].Equity() < plays[i].Equity()
		})
		// Filters the plays here based on bot type
		dist := g.Bag().LetterDistribution()
		// XXX: this should be cached:
		subChooseCombos := createSubCombos(dist)
		var wordsFormed []string
		var wordsNumCombinations []uint64
		for idx, play := range plays {
			if play.Action() == move.MoveTypePlay {
				machineWords, err := g.Board().FormedWords(play)
				if err != nil {
					log.Err(err).Msg("formed-words-error")
					break
				}
				wordsFormed = make([]string, len(machineWords))
				wordsNumCombinations = make([]uint64, len(machineWords))
				for i, mw := range machineWords {
					word := mw.UserVisible(g.Alphabet())
					wordsFormed[i] = word
					wordsNumCombinations[i] = combinations(dist, subChooseCombos, word, true)
				}
			}

			allowed, err := BotTypeMoveFilterMap[aiplayer.GetBotType()](cfg, wordsFormed, wordsNumCombinations, aiplayer.GetBotType())
			if err != nil {
				log.Err(err).Msg("bot-type-move-filter")
				break
			}
			if allowed {
				return []*move.Move{play}
			}
			// If we are all the way at the end we must pick a play, so pick
			// the worst play and move on.
			if idx == len(plays)-1 {
				return []*move.Move{play}
			}
		}
	}

	return aiplayer.TopPlays(plays, numPlays)
}
