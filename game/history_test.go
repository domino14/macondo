package game_test

import (
	"strings"
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
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	game, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)

	is.Equal(game.RackFor(0).String(), "GMU")
}

func TestNewFromHistoryIncomplete2(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	game, err := game.NewFromHistory(gameHistory, rules, 6)
	is.NoErr(err)

	is.Equal(game.RackFor(0).String(), "BDDDOST")
}

func TestNewFromHistoryIncomplete3(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../gcgio/testdata/incomplete.gcg")
	is.NoErr(err)
	game, err := game.NewFromHistory(gameHistory, rules, 7)
	is.NoErr(err)

	is.Equal(game.RackFor(1).String(), "EGOOY")
}

func TestNewFromHistoryIncomplete4(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../gcgio/testdata/incomplete_elise.gcg")
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
	is.Equal(g.Playing(), pb.PlayState_WAITING_FOR_FINAL_PASS)
}

func TestNewFromHistoryIncomplete5(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../gcgio/testdata/incomplete.gcg")
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
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../gcgio/testdata/incomplete_3.gcg")
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
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "../gcgio/testdata/incomplete4.gcg")
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

func TestNewFromHistoryFirstMoveIsPass(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)

	hist := &pb.GameHistory{
		Players: []*pb.PlayerInfo{
			{Nickname: "p0", UserId: "u0"},
			{Nickname: "p1", UserId: "u1"},
		},
		Events: []*pb.GameEvent{
			{Type: pb.GameEvent_PASS, PlayerIndex: 0, Rack: "AEINSTU"},
		},
		Lexicon: "CSW19",
	}

	g, err := game.NewFromHistory(hist, rules, 1)
	is.NoErr(err)
	is.Equal(g.Playing(), pb.PlayState_PLAYING)
	is.Equal(g.RackLettersFor(0), "AEINSTU")
}

// A pass can be recorded without its rack, when an annotator does not know
// it. The passer must still hold tiles, at that turn and after it, or the
// game reads as waiting for a final pass.
func TestNewFromHistoryPassWithoutRack(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(
		DefaultConfig, "", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")
	is.NoErr(err)
	gameHistory, err := gcgio.ParseGCGFromReader(DefaultConfig, strings.NewReader(
		"#player1 alice Alice\n#player2 bob Bob\n"+
			">alice: ACDEFGT 8H CAT +10 10\n"+
			">bob: EIIOUUV - +0 0\n"+
			">alice: DEFGIOR 9G DOG +11 21\n"+
			">bob: EIIOUUV - +0 0\n"))
	is.NoErr(err)
	// Take bob's racks out, as a GCG that leaves them out has none.
	gameHistory.Events[1].Rack = ""
	gameHistory.Events[3].Rack = ""

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)
	for turn := 0; turn <= len(gameHistory.Events); turn++ {
		is.NoErr(g.PlayToTurn(turn))
		is.Equal(g.Playing(), pb.PlayState_PLAYING)
		is.Equal(g.RackFor(0).NumTiles(), uint8(7))
		is.Equal(g.RackFor(1).NumTiles(), uint8(7))
	}
	// CAT and DOG are on the board, and both racks are full.
	is.Equal(g.Bag().TilesRemaining(), 100-6-14)
}
