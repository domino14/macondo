package bot

import (
	"os"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/turnplayer"
)

var DefaultConfig = config.DefaultConfig()

// TestWordSmogFullGames plays complete bot-vs-bot WordSmog games. It's a soak
// test for the parts a single-position test can't reach: incremental cross-set
// updates after every move, the alpha generator on a crowded board, endgame
// positions with a small rack, and the legality of every word the bot picks.
func TestWordSmogFullGames(t *testing.T) {
	is := is.New(t)
	if os.Getenv("MACONDO_DATA_PATH") == "" {
		t.Skip("MACONDO_DATA_PATH not set")
	}
	if testing.Short() {
		t.Skip("skipping full-game WordSmog soak test in short mode")
	}

	opts := &turnplayer.GameOptions{
		Variant:       game.VarWordSmog,
		ChallengeRule: pb.ChallengeRule_DOUBLE,
	}
	opts.Lexicon = &turnplayer.Lexicon{Name: "CSW21", Distribution: "english"}

	players := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "player one"},
		{Nickname: "p2", RealName: "player two"},
	}

	conf := &BotConfig{Config: *DefaultConfig}

	const numGames = 3
	for i := 0; i < numGames; i++ {
		p, err := NewBotTurnPlayer(conf, opts, players, pb.BotRequest_HASTY_BOT)
		if err != nil {
			t.Skip("could not create a WordSmog bot (is CSW21.kad present?): " + err.Error())
		}
		p.StartGame()
		p.SetBackupMode(game.NoBackup)

		turns := 0
		for p.Playing() == pb.PlayState_PLAYING {
			m := p.GenerateMoves(1)[0]
			if m.Action() == move.MoveTypePlay {
				// Every word the bot forms must be legal WordSmog.
				words, err := p.Board().FormedWords(m)
				is.NoErr(err)
				if err := p.ValidateWords(p.Lexicon(), words); err != nil {
					t.Fatalf("bot played an illegal WordSmog move %s: %v",
						m.ShortDescription(), err)
				}
			}
			is.NoErr(p.PlayMove(m, true, 0))
			turns++
			if turns > 200 {
				t.Fatal("game did not end")
			}
		}
		is.True(p.PointsFor(0) > 0 || p.PointsFor(1) > 0)
	}
}
