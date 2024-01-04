package rangefinder

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

var DefaultConfig = config.DefaultConfig()

func defaultSimCalculators(lexiconName string) []equity.EquityCalculator {
	c, err := equity.NewCombinedStaticCalculator(
		lexiconName, &DefaultConfig, "", equity.PEGAdjustmentFilename)
	if err != nil {
		panic(err)
	}
	return []equity.EquityCalculator{c}
}

func TestInferTilePlay(t *testing.T) {

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

	game.SetRackFor(0, tilemapping.RackFromString("PHEW", game.Alphabet()))
	_, err = game.PlayScoringMove("H6", "PHEW", true)
	is.NoErr(err)

	calcs := defaultSimCalculators(lex)

	rangeFinder := &RangeFinder{}
	rangeFinder.Init(game, calcs, &DefaultConfig)

	f, err := os.Create("/tmp/inferlog")
	is.NoErr(err)
	defer f.Close()
	rangeFinder.logStream = f

	rangeFinder.PrepareFinder(nil)
	timeout, cancel := context.WithTimeout(
		context.Background(), 5*time.Second)
	defer cancel()

	err = rangeFinder.Infer(timeout)
	is.NoErr(err)
	fmt.Println(rangeFinder.iterationCount)
	fmt.Println(len(rangeFinder.inferences))
	fmt.Println("analyze inferences")
	fmt.Println(rangeFinder.AnalyzeInferences(true))
	fmt.Println(rangeFinder.AnalyzeInferences(false))

}

func TestInferExchange(t *testing.T) {

	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	lex := "NWL18"
	players := []*macondo.PlayerInfo{
		{Nickname: "Joel", RealName: "Joel"},
		{Nickname: "Nigel", RealName: "Nigel"},
	}
	rules, err := game.NewBasicGameRules(&DefaultConfig, lex, board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)

	// This will start the game and deal a random rack to players:
	game.StartGame()
	game.SetPlayerOnTurn(0)

	game.SetRackFor(0, tilemapping.RackFromString("AENSTUU", game.Alphabet()))

	uu, err := tilemapping.ToMachineLetters("UU", game.Alphabet())
	is.NoErr(err)
	aenst, err := tilemapping.ToMachineLetters("AENST", game.Alphabet())
	is.NoErr(err)

	m := move.NewExchangeMove(uu, aenst, game.Alphabet())
	err = game.PlayMove(m, true, 0)
	is.NoErr(err)

	calcs := defaultSimCalculators(lex)

	rangeFinder := &RangeFinder{}
	rangeFinder.Init(game, calcs, &DefaultConfig)

	f, err := os.Create("/tmp/inferlog")
	is.NoErr(err)
	defer f.Close()
	rangeFinder.logStream = f

	// Nigel's rack was AELNOQT.
	aelnoqt, err := tilemapping.ToMachineLetters("AELNOQT", game.Alphabet())
	is.NoErr(err)
	err = rangeFinder.PrepareFinder(aelnoqt)
	is.NoErr(err)
	timeout, cancel := context.WithTimeout(
		context.Background(), 5*time.Second)
	defer cancel()

	err = rangeFinder.Infer(timeout)
	is.NoErr(err)
	fmt.Println(rangeFinder.iterationCount)
	fmt.Println(len(rangeFinder.inferences))
	fmt.Println("analyze inferences")
	fmt.Println(rangeFinder.AnalyzeInferences(true))
	fmt.Println(rangeFinder.AnalyzeInferences(false))
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

	game.SetRackFor(0, tilemapping.RackFromString("PHEW", game.Alphabet()))
	_, err = game.PlayScoringMove("8F", "PHEW", true)
	is.NoErr(err)

	is.Equal(game.PlayerOnTurn(), 1)

	calcs := defaultSimCalculators("NWL20")

	rangeFinder := &RangeFinder{}
	rangeFinder.Init(game, calcs, &DefaultConfig)
	rangeFinder.PrepareFinder(nil)

	is.Equal(rangeFinder.gameCopies[0].PlayerOnTurn(), 0)

	_, err = rangeFinder.inferSingle(0, 0, nil)
	is.NoErr(err)
}
