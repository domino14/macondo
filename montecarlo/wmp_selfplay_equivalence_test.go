package montecarlo

import (
	"os"
	"testing"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	wmppkg "github.com/domino14/macondo/wmp"
)

// TestWMPSelfPlayEquivalence plays seeded games with the generator a sim's
// rollouts use (best play only, the word map, every shortcut on) and at each
// turn asks two other generators for the best play of the same position: the
// same one with the word map's shortcuts switched off, and one with no word
// map at all. All three must name the same play with the same equity.
//
// It needs CSW21 and its word map; without them it is skipped.
func TestWMPSelfPlayEquivalence(t *testing.T) {
	const lexicon = "CSW21"
	wmpPath := DefaultConfig.GetString("data-path") + "/lexica/" + lexicon + ".wmp"
	if _, err := os.Stat(wmpPath); err != nil {
		t.Skipf("no word map at %s", wmpPath)
	}
	w, err := wmppkg.LoadFromFile(lexicon, wmpPath)
	if err != nil {
		t.Fatal(err)
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lexicon,
		board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Skipf("no %s: %v", lexicon, err)
	}
	calcs, _ := defaultSimCalculators(lexicon)
	players := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "Player1"},
		{Nickname: "p2", RealName: "Player2"},
	}
	games := 500
	if testing.Short() {
		games = 20
	}
	turns := 0
	for seedIdx := 0; seedIdx < games; seedIdx++ {
		g, err := game.NewGame(rules, players)
		if err != nil {
			t.Fatal(err)
		}
		var seed [32]byte
		seed[0], seed[1], seed[2] = byte(seedIdx), byte(seedIdx>>8), 0x51
		g.SeedBag(seed)
		g.StartGame()

		// fast: as in a rollout. plain: the word map with its shortcuts
		// off. kwg: no word map.
		type variant struct {
			name string
			p    *aiturnplayer.AIStaticTurnPlayer
		}
		var variants []variant
		for _, name := range []string{"fast", "plain", "kwg"} {
			p, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(g, DefaultConfig, calcs)
			if err != nil {
				t.Fatal(err)
			}
			gen := p.MoveGenerator().(*movegen.GordonGenerator)
			gen.SetGame(g)
			switch name {
			case "fast":
				gen.SetWMP(w)
				gen.SetWMPSelfCheck(true)
			case "plain":
				gen.SetWMP(w)
				gen.SetWMPSubrackLeavePrune(false)
				gen.SetExchangesFirst(false)
			case "kwg":
				gen.SetExchangesFirst(false)
			}
			variants = append(variants, variant{name, p})
		}
		for g.Playing() == pb.PlayState_PLAYING {
			onTurn := g.PlayerOnTurn()
			var best *move.Move
			for _, v := range variants {
				m := aiturnplayer.GenBestStaticTurn(g, v.p, onTurn)
				if best == nil {
					best = &move.Move{}
					best.CopyFrom(m)
					continue
				}
				if m.ShortDescription() != best.ShortDescription() ||
					m.Score() != best.Score() || m.Equity() != best.Equity() {
					t.Fatalf("game %d turn %d rack %s: fast found %s (%d, %.4f), %s found %s (%d, %.4f)\n%s",
						seedIdx, g.Turn(), g.RackLettersFor(onTurn),
						best.ShortDescription(), best.Score(), best.Equity(),
						v.name, m.ShortDescription(), m.Score(), m.Equity(),
						g.ToCGP(false))
				}
			}
			if err := g.PlayMove(best, false, 0); err != nil {
				t.Fatal(err)
			}
			turns++
		}
	}
	t.Logf("%d games, %d turns, the three generators agreed on every one", games, turns)
}
