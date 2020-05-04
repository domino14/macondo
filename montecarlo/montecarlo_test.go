package montecarlo

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/matryer/is"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/movegen"
	pb "github.com/domino14/macondo/rpc/api/proto"
	"github.com/domino14/macondo/strategy"
)

var LexiconDir = os.Getenv("LEXICON_PATH")

const (
	Epsilon = 1e-6
)

var DefaultConfig = &config.Config{
	StrategyParamsPath:        os.Getenv("STRATEGY_PARAMS_PATH"),
	LexiconPath:               os.Getenv("LEXICON_PATH"),
	DefaultLexicon:            "NWL18",
	DefaultLetterDistribution: "English",
}

func TestMain(m *testing.M) {
	for _, lex := range []string{"NWL18"} {
		gdgPath := filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
	os.Exit(m.Run())
}

func TestSimSingleIteration(t *testing.T) {
	is := is.New(t)
	plies := 2

	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewGameRules(DefaultConfig, board.CrosswordGameBoard,
		"NWL18", "English")
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)
	game.StartGame()

	strategy := strategy.NewExhaustiveLeaveStrategy(rules.Gaddag().LexiconName(),
		rules.Gaddag().GetAlphabet(), DefaultConfig.StrategyParamsPath, "")
	generator := movegen.NewGordonGenerator(rules.Gaddag(), game.Board(), rules.LetterDistribution())

	// This will deal a random rack to players:
	game.StartGame()
	// Overwrite the first rack
	game.SetRackFor(0, alphabet.RackFromString("AAADERW", rules.Gaddag().GetAlphabet()))
	generator.GenAll(game.RackFor(0), false)
	oldOppRack := game.RackFor(1).String()
	plays := generator.Plays()[:10]
	simmer := &Simmer{}
	simmer.Init(game, player.NewRawEquityPlayer(strategy))
	simmer.makeGameCopies()
	simmer.PrepareSim(plies, plays)

	simmer.simSingleIteration(plies, 0, 1, nil)

	// Board should be reset back to empty after the simulation.
	is.True(simmer.gameCopies[0].Board().IsEmpty())
	is.Equal(simmer.gameCopies[0].Turn(), 0)
	is.Equal(simmer.gameCopies[0].RackFor(0).String(), "AAADERW")
	// The original game shouldn't change at all.
	is.Equal(game.RackFor(1).String(), oldOppRack)
	is.True(game.Board().IsEmpty())

	simmer.sortPlaysByEquity()
	fmt.Println(simmer.printStats())
}

func TestLongerSim(t *testing.T) {
	// t.Skip()
	is := is.New(t)
	plies := 2

	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewGameRules(DefaultConfig, board.CrosswordGameBoard,
		"NWL18", "English")
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)
	game.StartGame()

	strategy := strategy.NewExhaustiveLeaveStrategy(rules.Gaddag().LexiconName(),
		rules.Gaddag().GetAlphabet(), DefaultConfig.StrategyParamsPath, "")

	generator := movegen.NewGordonGenerator(rules.Gaddag(), game.Board(), rules.LetterDistribution())
	// This will start the game and deal a random rack to players:
	game.StartGame()
	// Overwrite rack we are simming for. This is the prototypical Maven sim rack.
	// AWA should sim best.
	game.SetRackFor(0, alphabet.RackFromString("AAADERW", rules.Gaddag().GetAlphabet()))
	aiplayer := player.NewRawEquityPlayer(strategy)
	generator.GenAll(game.RackFor(0), false)
	aiplayer.AssignEquity(generator.Plays(), game.Board(), game.Bag(),
		game.RackFor(1))
	plays := aiplayer.TopPlays(generator.Plays(), 10)
	simmer := &Simmer{}
	simmer.Init(game, aiplayer)

	timeout, cancel := context.WithTimeout(
		context.Background(), 15*time.Second)
	defer cancel()

	f, err := os.Create("/tmp/simlog")
	is.NoErr(err)
	defer f.Close()
	simmer.logStream = f
	simmer.SetThreads(3)
	simmer.PrepareSim(plies, plays)
	simmer.Simulate(timeout)

	// Board should be reset back to empty after the simulation.
	is.True(game.Board().IsEmpty())
	fmt.Println(simmer.printStats())
	fmt.Println("Total iterations", simmer.iterationCount)
	// AWA wins (note that the print above also sorts the plays by equity)
	is.Equal(simmer.plays[0].play.Tiles().UserVisible(rules.Gaddag().GetAlphabet()), "AWA")
	is.Equal(simmer.gameCopies[0].Turn(), 0)
}

// func TestDrawingAssumptions(t *testing.T) {
// 	// Test that we are actually drawing from a sane bag.
// 	// is := is.New(t)
// 	plies := 2
// 	gd, err := GaddagFromLexicon("NWL18")
// 	if err != nil {
// 		t.Errorf("Expected error to be nil, got %v", err)
// 	}
// 	dist := alphabet.EnglishLetterDistribution()

// 	game := &mechanics.XWordGame{}
// 	game.Init(gd, dist)
// 	strategy := strategy.NewExhaustiveLeaveStrategy(game.Bag(), gd.LexiconName(),
// 		gd.GetAlphabet(), LeaveFile)
// 	generator := movegen.NewGordonGenerator(game, strategy)

// 	// Deal out racks.
// 	game.StartGame()
// 	game.SetRackFor(0, alphabet.RackFromString("AAADERW", gd.GetAlphabet()))

// 	simmer := &Simmer{}
// 	simmer.Init(generator, game)
// 	generator.GenAll(game.RackFor(0))
// 	plays := generator.Plays()[:10]
// 	simmer.resetStats(plies, len(plays))
// 	simmer.plays = plays

// 	simmer.simSingleIteration(plays, plies)

// }
