package crosscheck

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

var autoplayGames = flag.Int("crosscheck.autoplay", 10, "seeded games per case in the autoplay cross-check")

// TestAutoplayCrossCheck plays seeded games between static players in both
// engines and compares them move by move: the same seed must deal the same
// tiles, and the same bots must make the same plays, to the end. Pairs mode
// plays them from a paired bag (each seat drawing from its own end).
func TestAutoplayCrossCheck(t *testing.T) {
	e := startEngine(t)
	defer e.Quit()
	cfg := config.DefaultConfig()
	dataPath := cfg.GetString(config.ConfigDataPath)
	if dataPath == "" {
		t.Skip("no data path; set MACONDO_DATA_PATH")
	}
	if err := e.Send("setoption name data-path value " + dataPath); err != nil {
		t.Fatal(err)
	}
	const master = 20261125
	for _, bc := range boardCases {
		if bc.peg {
			continue
		}
		for _, pairs := range []bool{false, true} {
			name := bc.lexicon + "-" + bc.layout
			if pairs {
				name += "-pairs"
			}
			t.Run(name, func(t *testing.T) {
				if _, err := os.Stat(filepath.Join(dataPath, "lexica", "gaddag", bc.lexicon+".kwg")); err != nil {
					t.Skipf("no %s word graph", bc.lexicon)
				}
				dim := 15
				if bc.layout == board.SuperCrosswordGameLayout {
					dim = 21
				}
				empty := strings.TrimSuffix(strings.Repeat(fmt.Sprintf("%d/", dim), dim), "/")
				if err := e.Send(fmt.Sprintf("position cgp %s / 0/0 0 lex %s; ld %s; bdn %s;", empty, bc.lexicon, bc.dist, bc.layout)); err != nil {
					t.Fatal(err)
				}
				cmd := fmt.Sprintf("autoplay games %d seed %d moves", *autoplayGames, master)
				if pairs {
					cmd += " pairs"
				}
				lines, err := e.Command(cmd, "autoplayok")
				if err != nil {
					t.Fatal(err)
				}
				zig := map[int][]string{}
				for _, l := range lines[:len(lines)-1] {
					var gi int
					var rest string
					if strings.HasPrefix(l, "move ") || strings.HasPrefix(l, "game ") {
						f := strings.SplitN(l, " ", 3)
						fmt.Sscan(f[1], &gi)
						rest = f[0] + " " + f[2]
						if f[0] == "move" {
							rest = normalizeEquity(t, rest)
						}
					} else {
						t.Fatalf("bad autoplay line %q", l)
					}
					zig[gi] = append(zig[gi], rest)
				}
				for gi := 0; gi < *autoplayGames; gi++ {
					unit := gi
					if pairs {
						unit = gi / 2
					}
					want := goStaticGame(t, cfg, bc, automatic.DeriveSeed(master, unit), pairs)
					got := zig[gi]
					for i := range max(len(want), len(got)) {
						var w, g string
						if i < len(want) {
							w = want[i]
						}
						if i < len(got) {
							g = got[i]
						}
						if w != g {
							t.Fatalf("game %d, line %d: Go %q, Zig %q", gi, i, w, g)
						}
					}
				}
			})
		}
	}
}

// normalizeEquity rewrites "move <m> score <s> equity <e>" with the equity
// as Go formats it.
func normalizeEquity(t *testing.T, s string) string {
	f := strings.Fields(s)
	var eq float64
	if _, err := fmt.Sscan(f[len(f)-1], &eq); err != nil {
		t.Fatalf("bad equity in %q", s)
	}
	f[len(f)-1] = fmtFloat(eq)
	return strings.Join(f, " ")
}

// goStaticGame plays one seeded game between two HastyBots the way the
// autoplay runner does (GenBestStaticTurn, one bot per seat) and returns
// its moves and result as the Zig engine writes them.
func goStaticGame(t *testing.T, cfg *config.Config, bc boardCase, seed [32]byte, pairs bool) []string {
	t.Helper()
	rules, err := game.NewBasicGameRules(cfg, bc.lexicon, bc.layout, bc.dist, game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Fatal(err)
	}
	players := []*pb.PlayerInfo{{Nickname: "p1", RealName: "p1"}, {Nickname: "p2", RealName: "p2"}}
	g, err := game.NewGame(rules, players)
	if err != nil {
		t.Fatal(err)
	}
	g.SetPairedBagMode(pairs)
	g.SeedBag(seed)
	g.StartGame()
	leaves := ""
	if bc.layout == board.SuperCrosswordGameLayout {
		leaves = "super-leaves.klv2"
	}
	var bots [2]*bot.BotTurnPlayer
	for i := range bots {
		bots[i], err = bot.NewBotTurnPlayerFromGame(g, &bot.BotConfig{Config: *cfg, LeavesFile: leaves}, pb.BotRequest_HASTY_BOT)
		if err != nil {
			t.Fatal(err)
		}
	}
	var out []string
	for g.Playing() == pb.PlayState_PLAYING {
		onturn := g.PlayerOnTurn()
		m := turnplayer.GenBestStaticTurn(g, bots[onturn], onturn)
		out = append(out, fmt.Sprintf("move %s score %d equity %s", ucgiMove(m, g.Alphabet()), m.Score(), fmtFloat(m.Equity())))
		chosen := new(move.Move)
		chosen.CopyFrom(m)
		if err := g.PlayMove(chosen, false, 0); err != nil {
			t.Fatal(err)
		}
	}
	out = append(out, fmt.Sprintf("game score %d %d turns %d", g.PointsFor(0), g.PointsFor(1), g.Turn()))
	return out
}
