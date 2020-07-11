package runner

import (
	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
	"github.com/rs/zerolog/log"
	"strings"
)

type GameOptions struct {
	Lexicon       string
	ChallengeRule pb.ChallengeRule
}

func (opts *GameOptions) SetDefaults(config *config.Config) {
	if opts.Lexicon == "" {
		opts.Lexicon = config.DefaultLexicon
		log.Info().Msgf("using default lexicon %v", opts.Lexicon)
	}
}

// Basic game. Set racks, make moves

type GameRunner struct {
	*game.Game

	config  *config.Config
	options *GameOptions
}

func NewGameRunner(conf *config.Config, opts *GameOptions, players []*pb.PlayerInfo) (*GameRunner, error) {
	opts.SetDefaults(conf)
	rules, err := game.NewGameRules(
		conf, board.CrosswordGameBoard, opts.Lexicon,
		conf.DefaultLetterDistribution)
	if err != nil {
		return nil, err
	}

	g, err := game.NewGame(rules, players)
	if err != nil {
		return nil, err
	}
	g.StartGame()
	g.SetBackupMode(game.InteractiveGameplayMode)
	// Set challenge rule to double by default. This can be overridden.
	g.SetChallengeRule(opts.ChallengeRule)
	ret := &GameRunner{g, conf, opts}
	return ret, nil
}

func (g *GameRunner) SetPlayerRack(playerid int, letters string) error {
	rack := alphabet.RackFromString(letters, g.Alphabet())
	return g.SetRackFor(playerid, rack)
}

func (g *GameRunner) SetCurrentRack(letters string) error {
	return g.SetPlayerRack(g.PlayerOnTurn(), letters)
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

// Game with an AI player available for move generation.
type AIGameRunner struct {
	*GameRunner

	aiplayer player.AIPlayer
	gen      movegen.MoveGenerator
}

func NewAIGameRunner(conf *config.Config, opts *GameOptions, players []*pb.PlayerInfo) (*AIGameRunner, error) {
	g, err := NewGameRunner(conf, opts, players)
	if err != nil {
		return nil, err
	}

	strategy, err := strategy.NewExhaustiveLeaveStrategy(
		g.Gaddag().LexiconName(),
		g.Gaddag().GetAlphabet(),
		g.config.StrategyParamsPath,
		strategy.LeaveFilename)
	if err != nil {
		return nil, err
	}

	aiplayer := player.NewRawEquityPlayer(strategy)
	gen := movegen.NewGordonGenerator(
		g.Gaddag(), g.Board(), g.Bag().LetterDistribution())

	ret := &AIGameRunner{g, aiplayer, gen}
	return ret, nil
}

func (g *AIGameRunner) GenerateMoves(numPlays int) []*move.Move {
	curRack := g.RackFor(g.PlayerOnTurn())
	oppRack := g.RackFor(g.NextPlayer())

	g.gen.GenAll(curRack, g.Bag().TilesRemaining() >= 7)

	plays := g.gen.Plays()

	// Assign equity to plays, and return the top ones.
	g.aiplayer.AssignEquity(plays, g.Board(), g.Bag(), oppRack)
	return g.aiplayer.TopPlays(plays, numPlays)
}
