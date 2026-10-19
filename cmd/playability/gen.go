package main

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/playability"
)

type worker struct {
	g   *game.Game
	btp *bot.BotTurnPlayer
	ev  playability.Evaluator
}

func newConfig(lexicon, ld string) *config.Config {
	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, lexicon)
	cfg.Set(config.ConfigDefaultLetterDistribution, ld)
	return cfg
}

func newWorker(cfg *config.Config, g *game.Game) (*worker, error) {
	if g == nil {
		rules, err := game.NewBasicGameRules(cfg, cfg.GetString(config.ConfigDefaultLexicon),
			board.CrosswordGameLayout, cfg.GetString(config.ConfigDefaultLetterDistribution),
			game.CrossScoreAndSet, game.VarClassic)
		if err != nil {
			return nil, err
		}
		g, err = game.NewGame(rules, []*pb.PlayerInfo{{Nickname: "p1"}, {Nickname: "p2"}})
		if err != nil {
			return nil, err
		}
	}
	// PLAY_BOT swaps the bot (e.g. NO_LEAVE_BOT), for diagnostics only.
	code := pb.BotRequest_HASTY_BOT
	if b := os.Getenv("PLAY_BOT"); b != "" {
		v, ok := pb.BotRequest_BotCode_value[b]
		if !ok {
			return nil, fmt.Errorf("unknown PLAY_BOT %q", b)
		}
		code = pb.BotRequest_BotCode(v)
	}
	btp, err := bot.NewBotTurnPlayerFromGame(g, &bot.BotConfig{Config: *cfg}, code)
	if err != nil {
		return nil, err
	}
	return &worker{g: g, btp: btp}, nil
}

func gameSeed(shard, gi int) [32]byte {
	return sha256.Sum256([]byte(fmt.Sprintf("playability/%d/%d", shard, gi)))
}

// playShard plays one shard's games and returns the accumulated regrets.
func (w *worker) playShard(idx, games int, check bool, mismatches *atomic.Int64) (*playability.Shard, error) {
	sh := playability.NewShard(idx)
	for gi := 0; gi < games; gi++ {
		w.g.SeedBag(gameSeed(idx, gi))
		w.g.StartGame()
		for w.g.Playing() == pb.PlayState_PLAYING {
			pi := w.g.PlayerOnTurn()
			bag := w.g.Bag().TilesRemaining()
			plays := w.btp.GenerateMoves(math.MaxInt32)
			best := plays[0]
			wrs, ars := w.ev.Regrets(w.g.Board(), plays)
			ph := playability.Phase(bag)
			sh.Turns++
			sh.PhaseN[ph]++
			for _, wr := range wrs {
				st := sh.Words[wr.Word]
				if st == nil {
					st = &playability.Stat{}
					sh.Words[wr.Word] = st
				}
				st.Add(wr.Regret, wr.Main, ph)
			}
			for _, ar := range ars {
				st := sh.Alphas[ar.Alpha]
				if st == nil {
					st = &playability.Stat{}
					sh.Alphas[ar.Alpha] = st
				}
				st.Add(ar.Regret, false, ph)
			}
			bestEq := best.Equity()
			if check {
				// Confirm the full-list best equals what HastyBot's own top-play
				// path would choose. This regenerates, so the plays slice above
				// is dead from here on.
				top := turnplayer.GenBestStaticTurn(w.g, w.btp, pi)
				if math.Abs(top.Equity()-bestEq) > 1e-6 {
					mismatches.Add(1)
				}
				plays = w.btp.GenerateMoves(math.MaxInt32)
				best = plays[0]
			}
			if err := w.g.PlayMove(best, false, 0); err != nil {
				return nil, err
			}
		}
		sh.Games++
	}
	return sh, nil
}

func runGen(args []string) error {
	fs := flag.NewFlagSet("gen", flag.ExitOnError)
	out := fs.String("out", "", "output directory for shards")
	shards := fs.Int("shards", 100, "total number of shards wanted (existing ones are skipped)")
	gamesPer := fs.Int("games-per-shard", 2000, "games per shard (keep fixed for a run)")
	workers := fs.Int("workers", 3, "worker goroutines")
	lexicon := fs.String("lexicon", "NWL23", "lexicon")
	ld := fs.String("ld", "english", "letter distribution")
	check := fs.Bool("check", false, "verify the best move against HastyBot's top-play path")
	fs.Parse(args)
	if *out == "" {
		return fmt.Errorf("-out is required")
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		return err
	}
	cfg := newConfig(*lexicon, *ld)

	var todo []int
	for i := 0; i < *shards; i++ {
		if _, err := os.Stat(playability.ShardPath(*out, i)); err != nil {
			todo = append(todo, i)
		}
	}
	log.Info().Int("todo", len(todo)).Int("done", *shards-len(todo)).Msg("shards")
	jobs := make(chan int)
	go func() {
		for _, i := range todo {
			jobs <- i
		}
		close(jobs)
	}()
	var mism atomic.Int64
	var gamesDone atomic.Int64
	start := time.Now()
	var wg sync.WaitGroup
	errc := make(chan error, *workers)
	for wi := 0; wi < *workers; wi++ {
		wk, err := newWorker(cfg, nil)
		if err != nil {
			return err
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				sh, err := wk.playShard(idx, *gamesPer, *check, &mism)
				if err != nil {
					errc <- err
					return
				}
				if err := playability.WriteAtomic(playability.ShardPath(*out, idx), sh); err != nil {
					errc <- err
					return
				}
				n := gamesDone.Add(sh.Games)
				el := time.Since(start).Seconds()
				log.Info().Int("shard", idx).Int64("games", n).
					Float64("games/s", float64(n)/el).
					Float64("turns/game", float64(sh.Turns)/float64(sh.Games)).
					Int("words", len(sh.Words)).Int64("mismatch", mism.Load()).Msg("shard-done")
			}
		}()
	}
	wg.Wait()
	close(errc)
	return <-errc
}
