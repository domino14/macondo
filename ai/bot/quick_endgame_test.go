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

// A close endgame (vs_frentz, turn 22: bag empty) must be played by the
// quick search when configured, and left to the static player when the
// spread gate excludes it or the option is off.
func TestQuickEndgameMove(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW19", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	hist, err := gcgio.ParseGCG(DefaultConfig, "../../gcgio/testdata/vs_frentz.gcg")
	is.NoErr(err)

	newBotOfType := func(bt pb.BotRequest_BotCode, plies, margin int) (*BotTurnPlayer, *game.Game) {
		g, err := game.NewFromHistory(hist, rules, 22)
		is.NoErr(err)
		is.Equal(g.Bag().TilesRemaining(), 0)
		conf := &BotConfig{Config: *DefaultConfig, QuickEndgamePlies: plies, QuickEndgameMargin: margin}
		btp, err := NewBotTurnPlayerFromGame(g, conf, bt)
		is.NoErr(err)
		return btp, g
	}
	newBot := func(plies, margin int) (*BotTurnPlayer, *game.Game) {
		return newBotOfType(pb.BotRequest_HASTY_BOT, plies, margin)
	}

	// Off: the static player decides, no solves.
	btp, g := newBot(0, 0)
	m, err := btp.BestPlay(context.Background())
	is.NoErr(err)
	is.Equal(btp.QuickEndgameSolves, 0)
	is.NoErr(g.PlayMove(m, false, 0)) // legal

	// On, no gate: the quick search decides and its move is legal.
	btp, g = newBot(2, 0)
	m, err = btp.BestPlay(context.Background())
	is.NoErr(err)
	is.Equal(btp.QuickEndgameSolves, 1)
	is.NoErr(g.PlayMove(m, false, 0))

	// On, gated tighter than the current spread: static player again.
	btp, g = newBot(2, 1)
	spread := g.SpreadFor(g.PlayerOnTurn())
	if spread > 1 || spread < -1 {
		_, err = btp.BestPlay(context.Background())
		is.NoErr(err)
		is.Equal(btp.QuickEndgameSolves, 0)
	}

	// A simming bot without an endgame engine uses it too.
	btp, g = newBotOfType(pb.BotRequest_SIMMING_BOT_NO_EG, 2, 0)
	m, err = btp.BestPlay(context.Background())
	is.NoErr(err)
	is.Equal(btp.QuickEndgameSolves, 1)
	is.NoErr(g.PlayMove(m, false, 0))
}

// A simming bot sims the top 100 plays unless BotConfig.SimPlays says
// otherwise (autoplay -simplays1/2: as many plays as FastMlBot ranks).
func TestSimPlays(t *testing.T) {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW19", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	hist, err := gcgio.ParseGCG(DefaultConfig, "../../gcgio/testdata/vs_frentz.gcg")
	is.NoErr(err)
	for _, tc := range []struct{ conf, want int }{{0, 100}, {50, 50}} {
		g, err := game.NewFromHistory(hist, rules, 10)
		is.NoErr(err)
		conf := &BotConfig{Config: *DefaultConfig, SimPlays: tc.conf}
		btp, err := NewBotTurnPlayerFromGame(g, conf, pb.BotRequest_SIMMING_BOT_NO_EG)
		is.NoErr(err)
		is.Equal(btp.simPlaysOrDefault(), tc.want)
	}
}
