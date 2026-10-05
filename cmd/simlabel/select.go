package main

import (
	"bufio"
	"compress/gzip"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/cespare/xxhash"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/turnplayer"
)

type selRow struct {
	player, game, rack, play string
	turn, opening            int
}

type selGame struct {
	tp      *turnplayer.BaseTurnPlayer
	target  int
	pos     *Position
	mover   int
	skipped bool
}

func openMaybeGzip(path string) (io.ReadCloser, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if !strings.HasSuffix(path, ".gz") {
		return f, nil
	}
	zr, err := gzip.NewReader(bufio.NewReaderSize(f, 1<<20))
	if err != nil {
		f.Close()
		return nil, err
	}
	return struct {
		io.Reader
		io.Closer
	}{zr, f}, nil
}

func selectMain(args []string) {
	fs := flag.NewFlagSet("select", flag.ExitOnError)
	var turnsPath, tag, outPath, valPath, lexicon string
	var holdoutMod, maxTurn, minUnseen, threads int
	var rate float64
	var seed uint64
	fs.StringVar(&turnsPath, "turns", "", "autoplay turn log (.txt or .txt.gz)")
	fs.StringVar(&tag, "tag", "", "short name of the log, prefixed to every key (required)")
	fs.StringVar(&outPath, "out", "positions.jsonl", "training positions")
	fs.StringVar(&valPath, "val", "positions-val.jsonl", "held-out positions (games whose ID hashes to 0 mod -holdout-mod)")
	fs.StringVar(&lexicon, "lexicon", "NWL23", "lexicon")
	fs.IntVar(&holdoutMod, "holdout-mod", 20, "held-out split, as the streamed training's (0 = none)")
	fs.Float64Var(&rate, "rate", 1, "fraction of games to take a position from")
	fs.IntVar(&maxTurn, "max-turn", 30, "latest turn a position is drawn from")
	fs.IntVar(&minUnseen, "min-unseen", 9, "skip positions with fewer unseen tiles (pre-endgame and endgame)")
	fs.IntVar(&threads, "threads", 8, "replay workers")
	fs.Uint64Var(&seed, "seed", 1, "seed for the turn drawn from each game")
	fs.Parse(args)
	if turnsPath == "" || tag == "" {
		log.Fatal().Msg("-turns and -tag are required")
	}
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	cfg := config.DefaultConfig()

	in, err := openMaybeGzip(turnsPath)
	if err != nil {
		log.Fatal().Err(err).Msg("turn log")
	}
	defer in.Close()
	outF, err := os.Create(outPath)
	if err != nil {
		log.Fatal().Err(err).Msg("out")
	}
	defer outF.Close()
	valF, err := os.Create(valPath)
	if err != nil {
		log.Fatal().Err(err).Msg("val")
	}
	defer valF.Close()
	var mu sync.Mutex
	outW, valW := bufio.NewWriter(outF), bufio.NewWriter(valF)
	defer outW.Flush()
	defer valW.Flush()
	nOut, nVal := 0, 0
	emit := func(p *Position, held bool) {
		b, _ := json.Marshal(p)
		mu.Lock()
		defer mu.Unlock()
		if held {
			valW.Write(append(b, '\n'))
			nVal++
		} else {
			outW.Write(append(b, '\n'))
			nOut++
		}
	}

	jobs := make([]chan selRow, threads)
	var wg sync.WaitGroup
	for i := range jobs {
		jobs[i] = make(chan selRow, 4096)
		wg.Add(1)
		go func(ch <-chan selRow) {
			defer wg.Done()
			live := map[string]*selGame{}
			for r := range ch {
				if err := selFeed(cfg, lexicon, tag, live, r, maxTurn, minUnseen, seed, func(p *Position) {
					h := xxhash.Sum64String(p.Game)
					emit(p, holdoutMod > 0 && h%uint64(holdoutMod) == 0)
				}); err != nil {
					log.Fatal().Err(err).Msg("replay")
				}
			}
		}(jobs[i])
	}

	cr := csv.NewReader(bufio.NewReaderSize(in, 1<<20))
	cr.FieldsPerRecord = -1
	cr.TrimLeadingSpace = true
	hdr, err := cr.Read()
	if err != nil {
		log.Fatal().Err(err).Msg("header")
	}
	col := map[string]int{}
	for i, h := range hdr {
		col[h] = i
	}
	for _, need := range []string{"playerID", "gameID", "turn", "rack", "play"} {
		if _, ok := col[need]; !ok {
			log.Fatal().Msgf("turn log lacks column %q", need)
		}
	}
	openCol, hasOpen := col["openingplies"]
	threshold := uint64(rate * float64(^uint64(0)>>1))
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal().Err(err).Msg("turn row")
		}
		if rec[0] == "playerID" { // header of a concatenated log
			continue
		}
		g := rec[col["gameID"]]
		h := xxhash.Sum64String(g)
		// Rate sampling on a second, independent hash so it does not
		// correlate with the held-out split.
		if rate < 1 && (xxhash.Sum64String("rate:"+g)>>1) > threshold {
			continue
		}
		tn, _ := strconv.Atoi(rec[col["turn"]])
		op := 0
		if hasOpen && openCol < len(rec) {
			op, _ = strconv.Atoi(rec[openCol])
		}
		jobs[h%uint64(threads)] <- selRow{player: rec[col["playerID"]], game: g, turn: tn,
			rack: strings.TrimSpace(rec[col["rack"]]), play: strings.TrimSpace(rec[col["play"]]), opening: op}
	}
	for _, ch := range jobs {
		close(ch)
	}
	wg.Wait()
	fmt.Fprintf(os.Stderr, "%d training positions, %d held out\n", nOut, nVal)
}

// targetTurn draws the turn whose position a game contributes: uniform over
// the bot-played turns K+1..maxTurn (K = sampled opening plies).
func targetTurn(gameID string, opening, maxTurn int, seed uint64) int {
	lo := opening + 1
	if lo > maxTurn {
		return -1
	}
	h := xxhash.Sum64String(fmt.Sprintf("turn:%d:%s", seed, gameID))
	return lo + int(h%uint64(maxTurn-lo+1))
}

func newSelGame(cfg *config.Config, lexicon string, first, other string) (*turnplayer.BaseTurnPlayer, error) {
	rules, err := game.NewBasicGameRules(cfg, lexicon, board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		return nil, err
	}
	return turnplayer.BaseTurnPlayerFromRules(
		&turnplayer.GameOptions{Variant: game.VarClassic, BoardLayoutName: board.CrosswordGameLayout},
		[]*pb.PlayerInfo{{Nickname: first, RealName: first}, {Nickname: other, RealName: other}}, rules)
}

func selFeed(cfg *config.Config, lexicon, tag string, live map[string]*selGame, r selRow,
	maxTurn, minUnseen int, seed uint64, emit func(*Position)) error {
	sg := live[r.game]
	if sg == nil {
		other := "p2"
		if r.player == "p2" {
			other = "p1"
		}
		tp, err := newSelGame(cfg, lexicon, r.player, other)
		if err != nil {
			return err
		}
		sg = &selGame{tp: tp, target: targetTurn(r.game, r.opening, maxTurn, seed)}
		sg.skipped = sg.target < 0
		live[r.game] = sg
	}
	g := sg.tp.Game
	onturn := g.PlayerOnTurn()
	if g.NickOnTurn() != r.player {
		return fmt.Errorf("game %s turn %d: %s to move but the log says %s", r.game, r.turn, g.NickOnTurn(), r.player)
	}
	if err := g.SetRackFor(onturn, tilemapping.RackFromString(r.rack, g.Alphabet())); err != nil {
		return err
	}
	if !sg.skipped && r.turn == sg.target {
		unseen := g.Bag().TilesRemaining() + int(g.RackFor(1-onturn).NumTiles())
		if unseen < minUnseen || g.Bag().TilesRemaining() == 0 {
			sg.skipped = true
		} else {
			sg.pos = &Position{Key: fmt.Sprintf("%s:%s:%d", tag, r.game, r.turn), Game: r.game, Turn: r.turn,
				CGP: hideOppRack(g.ToCGP(false)), Played: r.play, Spread: g.SpreadFor(onturn), Unseen: unseen}
			sg.mover = onturn
		}
	}
	m, err := sg.tp.ParseMove(onturn, false, strings.Fields(stats.Normalize(r.play)), false)
	if err != nil {
		return fmt.Errorf("game %s turn %d: parse %q: %w", r.game, r.turn, r.play, err)
	}
	if err := g.PlayMove(m, false, 0); err != nil {
		return fmt.Errorf("game %s turn %d: play %q: %w", r.game, r.turn, r.play, err)
	}
	if st := g.Playing(); st == pb.PlayState_GAME_OVER || st == pb.PlayState_WAITING_FOR_FINAL_PASS {
		if st == pb.PlayState_WAITING_FOR_FINAL_PASS {
			p := g.PlayerOnTurn()
			g.ThrowRacksInFor(p)
			rack := tilemapping.NewRack(g.Alphabet())
			rack.Set(g.Bag().Peek())
			if err := g.SetRackForOnly(p, rack); err != nil {
				return err
			}
			if err := g.PlayMove(move.NewPassMove(g.RackFor(p).TilesOn(), g.Alphabet()), false, 0); err != nil {
				return err
			}
		}
		if sg.pos != nil {
			sg.pos.Final = g.SpreadFor(sg.mover)
			switch {
			case sg.pos.Final > 0:
				sg.pos.Result = 1
			case sg.pos.Final < 0:
				sg.pos.Result = -1
			}
			emit(sg.pos)
		}
		delete(live, r.game)
	}
	return nil
}

// hideOppRack blanks the opponent's rack in a CGP ("ours/theirs" -> "ours/"),
// as a bot sees the position. (ToCGP(true) does this too but needs the game
// history, which a log replay does not keep.)
func hideOppRack(c string) string {
	f := strings.SplitN(c, " ", 3)
	if len(f) < 3 {
		return c
	}
	if i := strings.Index(f[1], "/"); i >= 0 {
		f[1] = f[1][:i+1]
	}
	return strings.Join(f, " ")
}
