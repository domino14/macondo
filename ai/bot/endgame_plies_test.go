package bot

import (
	"context"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// One tile on each rack, bag empty: the endgame plies estimate is 1, which
// the multi-threaded searches used to reject ("use at least 2 plies"), so
// the bot returned no move at all whenever it solved with 2+ threads.
func TestEndgameOneTileEach(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL23", board.CrosswordGameLayout,
		"English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	hist, err := gcgio.ParseGCG(DefaultConfig, "../../gcgio/testdata/one_tile_each.gcg")
	is.NoErr(err)
	for _, threads := range []int{1, 2, 4} {
		g, err := game.NewFromHistory(hist, rules, len(hist.Events))
		is.NoErr(err)
		is.Equal(g.Bag().TilesRemaining(), 0)
		is.Equal(g.RackLettersFor(g.PlayerOnTurn()), "I")
		is.Equal(g.RackLettersFor(g.NextPlayer()), "R")
		btp, err := NewBotTurnPlayerFromGame(g, &BotConfig{Config: *DefaultConfig}, pb.BotRequest_SIMMING_BOT)
		is.NoErr(err)
		btp.SetEndgameThreads(threads)
		m, err := btp.BestPlay(context.Background())
		is.NoErr(err)
		is.Equal(m.ShortDescription(), " 9K .I") // goes out
	}
}
