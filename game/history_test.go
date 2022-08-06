package game_test

import (
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/matryer/is"
)

var DefaultConfig = config.DefaultConfig()

// Since the easiest way to create a history is with the gcgio package,
// we use package game_test above to avoid an import loop.
func TestNewFromHistoryIncomplete1(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	game, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)

	is.Equal(game.RackFor(0).String(), "GMU")
}

func TestNewFromHistoryIncomplete2(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	game, err := game.NewFromHistory(gameHistory, rules, 6)
	is.NoErr(err)

	is.Equal(game.RackFor(0).String(), "BDDDOST")
}

func TestNewFromHistoryIncomplete3(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	game, err := game.NewFromHistory(gameHistory, rules, 7)
	is.NoErr(err)

	is.Equal(game.RackFor(1).String(), "EGOOY")
}

func TestNewFromHistoryIncomplete4(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../gcgio/testdata/incomplete_elise.gcg")
	is.NoErr(err)
	is.Equal(len(gameHistory.Events), 20)

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
	is.Equal(g.Playing(), pb.PlayState_GAME_OVER)
}

func TestNewFromHistoryIncomplete5(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	is.Equal(len(gameHistory.Events), 20)

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)
	is.True(g != nil)
	err = g.PlayToTurn(20)
	is.NoErr(err)
	is.Equal(g.Playing(), pb.PlayState_PLAYING)
}

func TestNewFromHistoryIncomplete6(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../gcgio/testdata/incomplete_3.gcg")
	is.NoErr(err)
	is.Equal(len(gameHistory.Events), 20)

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)
	is.True(g != nil)
	err = g.PlayToTurn(20)
	is.NoErr(err)
	is.True(g.Playing() == pb.PlayState_PLAYING)
	is.Equal(g.RackLettersFor(0), "AEEIILZ")
}

func TestNewFromHistoryIncomplete7(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		&DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "../gcgio/testdata/incomplete4.gcg")
	is.NoErr(err)
	is.Equal(len(gameHistory.Events), 5)

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)
	is.True(g != nil)
	err = g.PlayToTurn(5)
	is.NoErr(err)
	is.True(g.Playing() == pb.PlayState_PLAYING)
	is.Equal(g.RackLettersFor(1), "AEEIILZ")
}
