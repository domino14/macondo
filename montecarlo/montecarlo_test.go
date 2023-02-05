package montecarlo

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/matryer/is"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/player"
	airunner "github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
	"github.com/domino14/macondo/testcommon"
)

const (
	Epsilon = 1e-6
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	testcommon.CreateGaddags(DefaultConfig, []string{"NWL18"})
	os.Exit(m.Run())
}

func TestSimSingleIteration(t *testing.T) {
	is := is.New(t)
	plies := 2

	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout, game.VarClassic,
		"NWL18", "English")
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)

	strategy, err := strategy.NewExhaustiveLeaveStrategy(rules.LexiconName(),
		game.Alphabet(), &DefaultConfig, strategy.LeaveFilename, strategy.PEGAdjustmentFilename)
	is.NoErr(err)

	gd, err := gaddag.Get(game.Config(), game.LexiconName())
	is.NoErr(err)

	generator := movegen.NewGordonGenerator(gd, game.Board(), rules.LetterDistribution())

	// This will deal a random rack to players:
	game.StartGame()
	game.SetPlayerOnTurn(0)
	// Overwrite the first rack
	game.SetRackFor(0, alphabet.RackFromString("AAADERW", game.Alphabet()))
	generator.GenAll(game.RackFor(0), false)
	oldOppRack := game.RackFor(1).String()
	plays := generator.Plays()[:10]
	simmer := &Simmer{}
	simmer.Init(game, player.NewRawEquityPlayer(strategy, pb.BotRequest_HASTY_BOT), &DefaultConfig)
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

func BenchmarkSim(b *testing.B) {
	is := is.New(b)
	plies := 2

	cgpstr := "C14/O2TOY9/mIRADOR8/F4DAB2PUGH1/I5GOOEY3V/T4XI2MALTHA/14N/6GUM3OWN/7PEW2DOE/9EF1DOR/2KUNA1J1BEVELS/3TURRETs2S2/7A4T2/7N7/7S7 EEEIILZ/ 336/298 0 lex NWL20;"

	game, err := cgp.ParseCGP(&DefaultConfig, cgpstr)
	is.NoErr(err)
	game.RecalculateBoard()
	strategy, err := strategy.NewExhaustiveLeaveStrategy(game.Rules().LexiconName(),
		game.Alphabet(), &DefaultConfig, strategy.LeaveFilename, strategy.PEGAdjustmentFilename)
	is.NoErr(err)

	gd, err := gaddag.Get(game.Config(), game.LexiconName())
	is.NoErr(err)

	generator := movegen.NewGordonGenerator(gd, game.Board(), game.Rules().LetterDistribution())

	generator.GenAll(game.RackFor(0), false)
	plays := generator.Plays()[:10]

	simmer := &Simmer{}
	simmer.Init(game, player.NewRawEquityPlayer(strategy, pb.BotRequest_HASTY_BOT), &DefaultConfig)
	simmer.SetThreads(1)
	simmer.PrepareSim(plies, plays)
	log.Debug().Msg("About to start")
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	b.ResetTimer()
	// benchmark 2022-08-20 on monolith (12th gen Intel computer)
	// 362	   3448347 ns/op	    7980 B/op	      60 allocs/op
	for i := 0; i < b.N; i++ {
		simmer.simSingleIteration(plies, 0, i+1, nil)
	}
}

func TestLongerSim(t *testing.T) {
	// t.Skip()
	is := is.New(t)
	plies := 2

	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := airunner.NewAIGameRules(&DefaultConfig, board.CrosswordGameLayout, game.VarClassic,
		"NWL18", "English")
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)

	strategy, err := strategy.NewExhaustiveLeaveStrategy(rules.LexiconName(),
		game.Alphabet(), &DefaultConfig, strategy.LeaveFilename, strategy.PEGAdjustmentFilename)
	is.NoErr(err)

	gd, err := gaddag.Get(game.Config(), game.LexiconName())
	is.NoErr(err)

	generator := movegen.NewGordonGenerator(gd, game.Board(), rules.LetterDistribution())
	// This will start the game and deal a random rack to players:
	game.StartGame()
	game.SetPlayerOnTurn(0)
	// Overwrite rack we are simming for. This is the prototypical Maven sim rack.
	// AWA should sim best.
	// Note we changed the rack here from AAADERW to AAAENSW because the test kept failing
	// because of the fairly new word ADWARE.
	game.SetRackFor(0, alphabet.RackFromString("AAAENSW", game.Alphabet()))
	aiplayer := player.NewRawEquityPlayer(strategy, pb.BotRequest_HASTY_BOT)
	aiplayer.SetMovegen(generator)
	generator.GenAll(game.RackFor(0), false)
	aiplayer.AssignEquity(generator.Plays(), game.Board(), game.Bag(),
		game.RackFor(1))
	plays := aiplayer.TopPlays(context.Background(), generator.Plays(), 10)
	simmer := &Simmer{}
	simmer.Init(game, aiplayer, &DefaultConfig)

	timeout, cancel := context.WithTimeout(
		context.Background(), 20*time.Second)
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
	is.Equal(simmer.plays[0].play.Tiles().UserVisible(game.Alphabet()), "AWA")
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
