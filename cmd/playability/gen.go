package main

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/playability"
)

type worker struct {
	g   *game.Game
	btp *bot.BotTurnPlayer
	gen *movegen.GordonGenerator
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
	btp, err := bot.NewBotTurnPlayerFromGame(g, &bot.BotConfig{Config: *cfg}, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return nil, err
	}
	gen := btp.MoveGenerator().(*movegen.GordonGenerator)
	gen.SetGame(g)
	gen.SetPlayRecorder(movegen.AllPlaysRecorder)
	return &worker{g: g, btp: btp, gen: gen}, nil
}

func gameSeed(shard, gi int) [32]byte {
	return sha256.Sum256([]byte(fmt.Sprintf("playability/%d/%d", shard, gi)))
}

// playShard plays one shard's games and returns the accumulated regrets.
func (w *worker) playShard(idx, games int, posRate float64, check bool, mismatches *atomic.Int64) (*playability.Shard, error) {
	sh := playability.NewShard(idx)
	rng := rand.New(rand.NewPCG(uint64(idx), 0x9e3779b9))
	tm := w.g.Alphabet()
	for gi := 0; gi < games; gi++ {
		w.g.SeedBag(gameSeed(idx, gi))
		w.g.StartGame()
		var prevLeave [2]tilemapping.MachineWord
		for w.g.Playing() == pb.PlayState_PLAYING {
			pi := w.g.PlayerOnTurn()
			bag := w.g.Bag().TilesRemaining()
			rack := w.g.RackFor(pi)
			if posRate > 0 && rng.Float64() < posRate {
				sh.Positions = append(sh.Positions, playability.Position{
					CGP:   w.g.ToCGP(false),
					Leave: prevLeave[pi].UserVisible(tm),
					Drawn: int(rack.NumTiles()) - len(prevLeave[pi]),
					Bag:   bag,
				})
			}
			w.gen.SetPlayRecorder(movegen.AllPlaysRecorder)
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
			leave := append(tilemapping.MachineWord(nil), best.Leave()...)
			if check {
				// Confirm the full-list best equals what HastyBot's own top-play
				// path would choose. This regenerates, so the plays slice above
				// is dead from here on.
				top := turnplayer.GenBestStaticTurn(w.g, w.btp, pi)
				if math.Abs(top.Equity()-bestEq) > 1e-6 {
					mismatches.Add(1)
				}
				w.gen.SetPlayRecorder(movegen.AllPlaysRecorder)
				plays = w.btp.GenerateMoves(math.MaxInt32)
				best = plays[0]
			}
			if err := w.g.PlayMove(best, false, 0); err != nil {
				return nil, err
			}
			prevLeave[pi] = leave
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
	posRate := fs.Float64("pos-rate", 0.01, "fraction of turns saved as positions for synth")
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
				sh, err := wk.playShard(idx, *gamesPer, *posRate, *check, &mism)
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
