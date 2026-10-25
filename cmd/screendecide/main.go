// screendecide asks a bot what it would play at positions from a logged
// autoplay run. Each position is rebuilt from the log with the game's own
// seed, so a seeded bot that is configured like the logged one replays the
// logged move exactly, and a variant (set by environment) differs only where
// the variant changes its mind. divjudge -mode screen then judges those
// differences.
//
//	screendecide -turns base.txt -player p1 -n 4000 -out moves.jsonl
package main

import (
	"context"
	"encoding/base64"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/turnplayer"
)

type position struct {
	gid, first string
	rows       [][]string
	i          int // row index of the decision
}

type record struct {
	Key       string  `json:"key"` // gameID|first|row
	GameID    string  `json:"gameID"`
	First     string  `json:"first"`
	Row       int     `json:"row"`
	Mover     string  `json:"mover"`
	Rack      string  `json:"rack"`
	Bag       int     `json:"bag"`
	OppPlayed int     `json:"oppPlayed"` // tiles in the opponent's last play, -1 exchange/pass/none
	OppScore  int     `json:"oppScore"`
	Logged    string  `json:"logged"`
	Play      string  `json:"play"`
	Score     int     `json:"score"`
	Tiles     int     `json:"tiles"`
	Leave     string  `json:"leave"`
	Equity    float64 `json:"equity"`
	Same      bool    `json:"same"`
	Secs      float64 `json:"secs"`
	Err       string  `json:"err,omitempty"`
}

func readPositions(path, player string) ([]position, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	byGame := map[string][][]string{}
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if rec[0] == "playerID" || len(rec) < 12 {
			continue
		}
		byGame[rec[1]] = append(byGame[rec[1]], rec)
	}
	var out []position
	for gid, rows := range byGame {
		for _, h := range automatic.SplitTurnHalves(rows) {
			for i, row := range h {
				// Skip the endgame: tilesremaining is logged after each move's
				// draw, so the previous row's is the bag this move sees.
				if row[0] != player || i == 0 || atoi(h[i-1][10]) == 0 {
					continue
				}
				out = append(out, position{gid: gid, first: h[0][0], rows: h, i: i})
			}
		}
	}
	// Map order is random; sort so a sample is the same every run.
	sort.Slice(out, func(a, b int) bool {
		if out[a].gid != out[b].gid {
			return out[a].gid < out[b].gid
		}
		if out[a].first != out[b].first {
			return out[a].first < out[b].first
		}
		return out[a].i < out[b].i
	})
	return out, nil
}

func atoi(s string) int { n, _ := strconv.Atoi(s); return n }

func decide(cfg *config.Config, rules *game.GameRules, conf bot.BotConfig, code pb.BotRequest_BotCode,
	p position) record {

	row := p.rows[p.i]
	rec := record{Key: fmt.Sprintf("%s|%s|%d", p.gid, p.first, p.i), GameID: p.gid, First: p.first,
		Row: p.i, Mover: row[0], Rack: row[3], Logged: strings.TrimSpace(row[4]), OppPlayed: -1}
	seedB, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(p.gid, "seed:"))
	if err != nil || len(seedB) != 32 {
		rec.Err = "game ID is not a seed"
		return rec
	}
	var seed [32]byte
	copy(seed[:], seedB)

	players := []*pb.PlayerInfo{
		{Nickname: p.rows[0][0], RealName: p.rows[0][0]},
		{Nickname: p.rows[1][0], RealName: p.rows[1][0]},
	}
	g, err := turnplayer.BaseTurnPlayerFromRules(&turnplayer.GameOptions{
		BoardLayoutName: board.CrosswordGameLayout, Variant: game.VarClassic}, players, rules)
	if err != nil {
		rec.Err = err.Error()
		return rec
	}
	g.SetPairedBagMode(true)
	g.SeedBag(seed)
	g.StartGame()
	g.SetUidFromSeed(seed)
	for j := 0; j < p.i; j++ {
		if err := automatic.PlayLoggedRow(g, p.rows[j], true); err != nil {
			rec.Err = fmt.Sprintf("replay row %d: %v", j+1, err)
			return rec
		}
	}
	me := 0
	if g.History().Players[1].Nickname == row[0] {
		me = 1
	}
	for j := p.i - 1; j >= 0; j-- {
		if p.rows[j][0] != row[0] {
			rec.OppScore = atoi(p.rows[j][5])
			if !strings.HasPrefix(p.rows[j][4], "(") {
				rec.OppPlayed = atoi(p.rows[j][7])
			}
			break
		}
	}
	// Seat the racks the logged game had: the mover's from this row, the
	// opponent's from their next row (nothing they hold depends on this move).
	racks := make([]*tilemapping.Rack, 2)
	racks[me] = tilemapping.RackFromString(row[3], g.Alphabet())
	oppRack := ""
	for j := p.i + 1; j < len(p.rows); j++ {
		if p.rows[j][0] != row[0] {
			oppRack = p.rows[j][3]
			break
		}
	}
	if oppRack != "" {
		racks[1-me] = tilemapping.RackFromString(oppRack, g.Alphabet())
		err = g.Game.SetRacksForBoth(racks)
	} else {
		err = g.SetRackFor(me, racks[me])
	}
	if err != nil {
		rec.Err = "seating racks: " + err.Error()
		return rec
	}
	rec.Bag = g.Bag().TilesRemaining()
	if rec.Bag == 0 {
		rec.Err = "bag empty"
		return rec
	}

	btp, err := bot.NewBotTurnPlayerFromGame(g.Game, &conf, code)
	if err != nil {
		rec.Err = err.Error()
		return rec
	}
	t0 := time.Now()
	m, err := btp.BestPlay(context.Background())
	rec.Secs = time.Since(t0).Seconds()
	if err != nil {
		rec.Err = "best play: " + err.Error()
		return rec
	}
	rec.Play = m.ShortDescription()
	rec.Score = m.Score()
	rec.Tiles = m.TilesPlayed()
	rec.Leave = m.LeaveString()
	rec.Equity = m.Equity()
	rec.Same = strings.TrimSpace(rec.Play) == rec.Logged
	return rec
}

func main() {
	turns := flag.String("turns", "", "per-turn log of the baseline run")
	player := flag.String("player", "p1", "decide this player's moves")
	botCode := flag.String("bot", "SIMMING_INFER_BOT_NO_EG", "bot code")
	plies := flag.Int("plies", 2, "minSimPlies")
	budget := flag.Int("budget", 200, "inferenceBudget")
	tau := flag.Float64("tau", 0, "inferenceTau (0 = engine default)")
	minBag := flag.Int("minbag", 1, "only positions where the mover sees at least this many tiles in the bag")
	maxBag := flag.Int("maxbag", 100, "... and at most this many")
	oracle := flag.Bool("oracle", false, "tell the bot the opponent's true leave (oracleInference)")
	n := flag.Int("n", 0, "sample this many positions (0 = all)")
	sample := flag.Int64("sample", 1, "sampling seed")
	maxOpp := flag.Int("maxopp", 7, "only positions after an opponent placement of at most this many tiles (-1 = any, also exchanges)")
	minOpp := flag.Int("minopp", 1, "... and at least this many")
	out := flag.String("out", "decide.jsonl", "output")
	parallel := flag.Int("parallel", 8, "positions at once (each single-threaded)")
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)

	code, ok := pb.BotRequest_BotCode_value[*botCode]
	if !ok {
		log.Fatal().Str("bot", *botCode).Msg("unknown bot code")
	}
	pos, err := readPositions(*turns, *player)
	if err != nil {
		log.Fatal().Err(err).Msg("turns")
	}
	if *maxOpp >= 0 {
		kept := pos[:0]
		for _, p := range pos {
			prev := p.rows[p.i-1]
			if prev[0] == p.rows[p.i][0] || strings.HasPrefix(prev[4], "(") {
				continue
			}
			if t := atoi(prev[7]); t >= *minOpp && t <= *maxOpp {
				kept = append(kept, p)
			}
		}
		pos = kept
	}
	{
		kept := pos[:0]
		for _, p := range pos {
			if b := atoi(p.rows[p.i-1][10]); b >= *minBag && b <= *maxBag {
				kept = append(kept, p)
			}
		}
		pos = kept
	}
	if *n > 0 && *n < len(pos) {
		rng := rand.New(rand.NewSource(*sample))
		rng.Shuffle(len(pos), func(a, b int) { pos[a], pos[b] = pos[b], pos[a] })
		pos = pos[:*n]
	}
	fmt.Fprintf(os.Stderr, "%d positions\n", len(pos))

	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, "NWL23")
	cfg.Set(config.ConfigDefaultLetterDistribution, "english")
	rules, err := game.NewBasicGameRules(cfg, "NWL23", board.CrosswordGameLayout,
		"english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		log.Fatal().Err(err).Msg("rules")
	}
	conf := bot.BotConfig{Config: *cfg, MinSimPlies: *plies, SimThreads: 1, InferenceBudget: *budget,
		OracleInference: *oracle, InferenceTau: *tau}

	f, err := os.Create(*out)
	if err != nil {
		log.Fatal().Err(err).Msg("out")
	}
	defer f.Close()
	var mu sync.Mutex
	enc := json.NewEncoder(f)
	ch := make(chan position)
	var wg sync.WaitGroup
	var done, same atomic.Int64
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range ch {
				rec := decide(cfg, rules, conf, pb.BotRequest_BotCode(code), p)
				mu.Lock()
				enc.Encode(rec)
				mu.Unlock()
				if rec.Same {
					same.Add(1)
				}
				if d := done.Add(1); d%100 == 0 {
					fmt.Fprintf(os.Stderr, "%d/%d (%d same as logged)\n", d, len(pos), same.Load())
				}
			}
		}()
	}
	for _, p := range pos {
		ch <- p
	}
	close(ch)
	wg.Wait()
	fmt.Fprintf(os.Stderr, "done: %d positions, %d same as logged\n", done.Load(), same.Load())
}
