package game_test

import (
	"os"
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	"github.com/matryer/is"
)

var DefaultConfig = &config.Config{
	StrategyParamsPath:        os.Getenv("STRATEGY_PARAMS_PATH"),
	LexiconPath:               os.Getenv("LEXICON_PATH"),
	DefaultLexicon:            "NWL18",
	DefaultLetterDistribution: "English",
}

// Since the easiest way to create a history is with the gcgio package,
// we use package game_test above to avoid an import loop.
func TestNewFromHistoryIncomplete1(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewGameRules(DefaultConfig, board.CrosswordGameBoard,
		"NWL18", "English")
	gameHistory, err := gcgio.ParseGCG("../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	game, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)

	is.Equal(game.RackFor(0).String(), "GMU")
}

func TestNewFromHistoryIncomplete2(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewGameRules(DefaultConfig, board.CrosswordGameBoard,
		"NWL18", "English")
	gameHistory, err := gcgio.ParseGCG("../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	game, err := game.NewFromHistory(gameHistory, rules, 6)
	is.NoErr(err)

	is.Equal(game.RackFor(0).String(), "BDDDOST")
}

func TestNewFromHistoryIncomplete3(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewGameRules(DefaultConfig, board.CrosswordGameBoard,
		"NWL18", "English")
	gameHistory, err := gcgio.ParseGCG("../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	game, err := game.NewFromHistory(gameHistory, rules, 7)
	is.NoErr(err)

	is.Equal(game.RackFor(1).String(), "EGOOY")
}

func TestNewFromHistoryIncomplete4(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewGameRules(DefaultConfig, board.CrosswordGameBoard,
		"NWL18", "English")
	gameHistory, err := gcgio.ParseGCG("../gcgio/testdata/incomplete_elise.gcg")
	is.NoErr(err)
	is.Equal(len(gameHistory.Turns), 20)

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)
	err = g.PlayToTurn(20)
	is.NoErr(err)

	// The Elise GCG is very malformed. Besides having play-through letters
	// instead of `.` signs, it also uses the capital version of these
	// letters (see COMFIT for example in the gcg), and specifies
	// the last known racks incorrectly. Rack2 is impossible given the
	// pool, so let's just ignore these two tests and find a better
	// use of #rack1 and #rack2
	// is.Equal(game.RackLettersFor(0), "AEEIILZ")
	// is.Equal(game.RackLettersFor(1), "AAAELQS")

	// Since it's malformed we end the game. Oops; do not load this GCG!
	is.Equal(g.Playing(), game.StateGameOver)
}

func TestNewFromHistoryIncomplete5(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewGameRules(DefaultConfig, board.CrosswordGameBoard,
		"NWL18", "English")
	gameHistory, err := gcgio.ParseGCG("../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	is.Equal(len(gameHistory.Turns), 20)

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)
	is.True(g != nil)
	err = g.PlayToTurn(20)
	is.NoErr(err)
	is.Equal(g.Playing(), game.StatePlaying)
}

func TestNewFromHistoryIncomplete6(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewGameRules(DefaultConfig, board.CrosswordGameBoard,
		"NWL18", "English")
	gameHistory, err := gcgio.ParseGCG("../gcgio/testdata/incomplete_3.gcg")
	is.NoErr(err)
	is.Equal(len(gameHistory.Turns), 20)

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)
	is.True(g != nil)
	err = g.PlayToTurn(20)
	is.NoErr(err)
	is.True(g.Playing() == game.StatePlaying)
	is.Equal(g.RackLettersFor(0), "AEEIILZ")
}
