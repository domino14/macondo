package runner

import (
	"strings"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
	"github.com/rs/zerolog/log"
)

// Basic game. Set racks, make moves

type GameRunner struct {
	game.Game
}

// NewGameRunnerFromRules is a good entry point
func NewGameRunnerFromRules(opts *GameOptions, players []*pb.PlayerInfo, rules *game.GameRules) (*GameRunner, error) {
	g, err := game.NewGame(rules, players)
	if err != nil {
		return nil, err
	}
	if opts.FirstIsAssigned {
		g.SetNextFirst(opts.GoesFirst)
	} else {
		// game determines it.
		g.SetNextFirst(-1)
	}
	g.StartGame()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	g.SetChallengeRule(opts.ChallengeRule)
	ret := &GameRunner{*g}
	return ret, nil
}

func (g *GameRunner) SetPlayerRack(playerid int, letters string) error {
	rack := alphabet.RackFromString(letters, g.Alphabet())
	return g.SetRackFor(playerid, rack)
}

func (g *GameRunner) SetCurrentRack(letters string) error {
	return g.SetPlayerRack(g.PlayerOnTurn(), letters)
}

func (g *GameRunner) NewPassMove(playerid int) (*move.Move, error) {
	rack := g.RackFor(playerid)
	m := move.NewPassMove(rack.TilesOn(), g.Alphabet())
	return m, nil
}

func (g *GameRunner) NewChallengeMove(playerid int) (*move.Move, error) {
	rack := g.RackFor(playerid)
	m := move.NewChallengeMove(rack.TilesOn(), g.Alphabet())
	return m, nil
}

func (g *GameRunner) NewExchangeMove(playerid int, letters string) (*move.Move, error) {
	alph := g.Alphabet()
	rack := g.RackFor(playerid)
	tiles, err := alphabet.ToMachineWord(letters, alph)
	if err != nil {
		return nil, err
	}
	leaveMW, err := game.Leave(rack.TilesOn(), tiles)
	if err != nil {
		return nil, err
	}
	m := move.NewExchangeMove(tiles, leaveMW, alph)
	return m, nil
}

func (g *GameRunner) NewPlacementMove(playerid int, coords string, word string) (*move.Move, error) {
	coords = strings.ToUpper(coords)
	rack := g.RackFor(playerid).String()
	return g.CreateAndScorePlacementMove(coords, word, rack)
}

func (g *GameRunner) MoveFromEvent(evt *pb.GameEvent) *move.Move {
	return game.MoveFromEvent(evt, g.Alphabet(), g.Board())
}

func (g *GameRunner) IsPlaying() bool {
	return g.Playing() == pb.PlayState_PLAYING
}

// Game with an AI player available for move generation.
type AIGameRunner struct {
	GameRunner

	aiplayer player.AIPlayer
	gen      movegen.MoveGenerator
	cfg      *config.Config
}

func NewAIGameRunner(conf *config.Config, opts *GameOptions, players []*pb.PlayerInfo, botType pb.BotRequest_BotCode) (*AIGameRunner, error) {
	opts.SetDefaults(conf)
	rules, err := NewAIGameRules(
		conf, board.CrosswordGameLayout,
		opts.Lexicon.Name, opts.Lexicon.Distribution)
	if err != nil {
		return nil, err
	}
	g, err := NewGameRunnerFromRules(opts, players, rules)
	if err != nil {
		return nil, err
	}
	return addAIFields(g, conf, botType)
}

func NewAIGameRunnerFromGame(g *game.Game, conf *config.Config, botType pb.BotRequest_BotCode) (*AIGameRunner, error) {
	gr := GameRunner{*g}
	return addAIFields(&gr, conf, botType)
}

func addAIFields(g *GameRunner, conf *config.Config, botType pb.BotRequest_BotCode) (*AIGameRunner, error) {
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

func (g *AIGameRunner) GenerateMoves(numPlays int) []*move.Move {
	curRack := g.RackFor(g.PlayerOnTurn())
	oppRack := g.RackFor(g.NextPlayer())

	g.gen.GenAll(curRack, g.Bag().TilesRemaining() >= 7)

	plays := g.gen.Plays()

	// Assign equity to plays, and return the top ones.
	g.aiplayer.AssignEquity(plays, g.Board(), g.Bag(), oppRack)

	if numPlays == 1 {
		// Filters the plays here based on bot type
		dist := g.Bag().LetterDistribution()
		subChooseCombos := createSubCombos(dist)
		for _, play := range plays {
			machineWords, err := g.Board().FormedWords(play)
			if err != nil {
				log.Err(err).Msg("formed-words-error")
				break
			}
			wordsFormed := make([]string, len(machineWords))
			wordsNumCombinations := make([]uint64, len(machineWords))
			for i, mw := range machineWords {
				word := mw.UserVisible(g.Alphabet())
				wordsFormed[i] = word
				wordsNumCombinations[i] = combinations(dist, subChooseCombos, word, true)
			}
			allowed, err := BotTypeMoveFilterMap[g.aiplayer.GetBotType()](g.cfg, wordsFormed, wordsNumCombinations, g.aiplayer.GetBotType())
			if err != nil {
				log.Err(err).Msg("bot-type-move-filter")
				break
			}
			if allowed {
				return []*move.Move{play}
			}
		}
	}

	return g.aiplayer.TopPlays(plays, numPlays)
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
