package rangefinder

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/matryer/is"
	"github.com/rs/zerolog"
)

var DefaultConfig = config.DefaultConfig()

func defaultSimCalculators(lexiconName string) []equity.EquityCalculator {
	c, err := equity.NewCombinedStaticCalculator(
		lexiconName, &DefaultConfig, equity.LeaveFilename, equity.PEGAdjustmentFilename)
	if err != nil {
		panic(err)
	}
	return []equity.EquityCalculator{c}
}

func TestInfer(t *testing.T) {

	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	lex := "NWL18"
	players := []*macondo.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewBasicGameRules(&DefaultConfig, lex, board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)

	// This will start the game and deal a random rack to players:
	game.StartGame()
	game.SetPlayerOnTurn(0)

	game.SetRackFor(0, alphabet.RackFromString("PHEW", game.Alphabet()))
	_, err = game.PlayScoringMove("H6", "PHEW", true)
	is.NoErr(err)

	calcs := defaultSimCalculators(lex)

	rangeFinder := &RangeFinder{}
	rangeFinder.Init(game, calcs, &DefaultConfig)
	rangeFinder.PrepareFinder()
	timeout, cancel := context.WithTimeout(
		context.Background(), 5*time.Second)
	defer cancel()

	err = rangeFinder.Infer(timeout)
	is.NoErr(err)
	fmt.Println(rangeFinder.iterationCount)
	fmt.Println(len(rangeFinder.inferences))
	fmt.Println("analyze inferences")
	Analyze(rangeFinder.inferences, rangeFinder.origGame.Alphabet(),
		rangeFinder.inferenceBagMap)
}

func TestInferSingle(t *testing.T) {
	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.DebugLevel)

	players := []*macondo.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewBasicGameRules(&DefaultConfig, "NWL20", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)

	// This will start the game and deal a random rack to players:
	game.StartGame()
	game.SetPlayerOnTurn(0)

	game.SetRackFor(0, alphabet.RackFromString("PHEW", game.Alphabet()))
	_, err = game.PlayScoringMove("8F", "PHEW", true)
	is.NoErr(err)

	is.Equal(game.PlayerOnTurn(), 1)

	calcs := defaultSimCalculators("NWL20")

	rangeFinder := &RangeFinder{}
	rangeFinder.Init(game, calcs, &DefaultConfig)
	rangeFinder.PrepareFinder()

	is.Equal(rangeFinder.gameCopies[0].PlayerOnTurn(), 0)

	_, err = rangeFinder.inferSingle(0)
	is.NoErr(err)
}
