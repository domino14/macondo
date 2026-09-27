// mlreads finds the positions in an autoplay match where the ML bot chose a
// different move from the static-equity (HastyBot) move for the same
// position, and records both with their equities and the game's outcome, so
// the interesting disagreements can be adjudicated by a sim and shown.
//
//	mlreads -turns tf-nwl23s-v-hasty-pairs.txt -games games-tf-nwl23s-v-hasty-pairs.txt \
//	    -bot p1 -lexicon NWL23 -max-games 30000 > reads.jsonl
//
// Every line of the output is one disagreement:
//
//	{"game":..., "half":0, "turn":7, "bag":52, "rack":"AEINRST", "cgp":"...",
//	 "ml":"8D TRAINEE", "ml_eq":31.2, "hasty":"H4 RETAINS", "hasty_eq":38.9,
//	 "gap":7.7, "ml_spread":12, "ml_won":true}
//
// The turn log is the autoplay per-turn CSV; a paired run lists the two
// halves of a pair under the same game ID, one after the other, each
// starting again at turn 1. The games file supplies each half's result.
package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/turnplayer"
)

// dumpSpec names one position whose top-50 candidates get their net input
// vectors written out, for evaluating the model offline.
type dumpSpec struct {
	game       string
	half, turn int
	prefix     string
	leaves     *equity.ExhaustiveLeaveCalculator
}

type turnRow struct {
	player string
	game   string
	turn   int
	rack   string
	play   string
	equity float64
	bag    int // tiles remaining after the ply
}

type halfResult struct {
	mlSpread int
	first    string // nickname of the bot that moved first
}

type disagreement struct {
	Game     string  `json:"game"`
	Half     int     `json:"half"`
	Turn     int     `json:"turn"`
	Bag      int     `json:"bag"` // tiles in the bag before the move
	Rack     string  `json:"rack"`
	CGP      string  `json:"cgp"`
	ML       string  `json:"ml"`
	MLEq     float64 `json:"ml_eq"`
	Hasty    string  `json:"hasty"`
	HastyEq  float64 `json:"hasty_eq"`
	Gap      float64 `json:"gap"`
	MLScore  int     `json:"ml_score"`  // before the move
	OppScore int     `json:"opp_score"` // before the move
	MLSpread int     `json:"ml_spread"`
	MLWon    bool    `json:"ml_won"`
	MLTurns  int     `json:"ml_turns"` // the ML bot's move count in the game
}

// readGames maps game ID -> the halves' results for the ML bot, in play order.
func readGames(path, mlName string) (map[string][]halfResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	hdr, err := r.Read()
	if err != nil {
		return nil, err
	}
	col := map[string]int{}
	for i, h := range hdr {
		col[h] = i
	}
	var oppName string
	for h := range col {
		if strings.HasSuffix(h, "_score") && !strings.HasPrefix(h, mlName) {
			oppName = strings.TrimSuffix(h, "_score")
		}
	}
	if _, ok := col[mlName+"_score"]; !ok || oppName == "" {
		return nil, fmt.Errorf("games file lacks %s_score / opponent score columns", mlName)
	}
	out := map[string][]halfResult{}
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		ml, _ := strconv.Atoi(rec[col[mlName+"_score"]])
		opp, _ := strconv.Atoi(rec[col[oppName+"_score"]])
		out[rec[0]] = append(out[rec[0]], halfResult{mlSpread: ml - opp, first: rec[col["first"]]})
	}
	return out, nil
}

// replay holds one half of a game being replayed move by move.
type replay struct {
	tp   *turnplayer.BaseTurnPlayer
	g    *game.Game
	ai   *aiturnplayer.AIStaticTurnPlayer
	hist []*move.Move // every move played so far, oldest first
	half int
	// nickname -> player index
	idx map[string]int
}

func newReplay(cfg *config.Config, lexicon string, calcs []equity.EquityCalculator, firstNick, otherNick string, half int) (*replay, error) {
	rules, err := game.NewBasicGameRules(cfg, lexicon, board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		return nil, err
	}
	tp, err := turnplayer.BaseTurnPlayerFromRules(
		&turnplayer.GameOptions{Variant: game.VarClassic, BoardLayoutName: board.CrosswordGameLayout},
		[]*pb.PlayerInfo{{Nickname: firstNick, RealName: firstNick}, {Nickname: otherNick, RealName: otherNick}}, rules)
	if err != nil {
		return nil, err
	}
	ai, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(tp.Game, cfg, calcs)
	if err != nil {
		return nil, err
	}
	return &replay{tp: tp, g: tp.Game, ai: ai, half: half, idx: map[string]int{firstNick: 0, otherNick: 1}}, nil
}

type worker struct {
	cfg     *config.Config
	lexicon string
	calcs   []equity.EquityCalculator
	mlNick  string
	games   map[string][]halfResult
	out     chan<- disagreement
	// per game ID: the current half's replay and how many ML moves so far
	live map[string]*replay
	// disagreements of the current half, emitted once the half's move count is known
	pending map[string][]disagreement
	mlMoves map[string]int
	games_  int
	dump    *dumpSpec
}

func (w *worker) feed(t turnRow) error {
	rp := w.live[t.game]
	if rp == nil || t.turn == 1 && rp.g.Turn() > 0 {
		// A new half. Emit the previous half's rows first.
		if rp != nil {
			w.flush(t.game, rp)
		}
		half := 0
		if rp != nil {
			half = rp.half + 1
		}
		other := "p2"
		if t.player == "p2" {
			other = "p1"
		}
		var err error
		rp, err = newReplay(w.cfg, w.lexicon, w.calcs, t.player, other, half)
		if err != nil {
			return err
		}
		w.live[t.game] = rp
		w.mlMoves[t.game] = 0
	}
	g := rp.g
	onturn := g.PlayerOnTurn()
	if g.NickOnTurn() != t.player {
		return fmt.Errorf("game %s half %d turn %d: %s to move but the log says %s", t.game, rp.half, t.turn, g.NickOnTurn(), t.player)
	}
	if err := g.SetRackFor(onturn, tilemapping.RackFromString(t.rack, g.Alphabet())); err != nil {
		return err
	}
	m, err := rp.tp.ParseMove(onturn, false, strings.Fields(stats.Normalize(t.play)), false)
	if err != nil {
		return fmt.Errorf("game %s turn %d: parse %q: %w", t.game, t.turn, t.play, err)
	}
	if t.player == w.mlNick {
		w.mlMoves[t.game]++
		if d := w.dump; d != nil && d.game == t.game && d.half == rp.half && d.turn == t.turn {
			if err := dumpCandidates(rp, d, m); err != nil {
				return err
			}
		}
		if g.Bag().TilesRemaining() > 0 {
			best := aiturnplayer.GenBestStaticTurn(g, rp.ai, onturn)
			bestDesc, bestEq := best.ShortDescription(), best.Equity()
			mlDesc := m.ShortDescription()
			if bestDesc != mlDesc {
				w.pending[t.game] = append(w.pending[t.game], disagreement{
					Game: t.game, Half: rp.half, Turn: t.turn, Bag: g.Bag().TilesRemaining(),
					Rack: t.rack, CGP: g.ToCGP(false),
					MLScore: g.PointsFor(onturn), OppScore: g.PointsFor(1 - onturn),
					ML: mlDesc, MLEq: t.equity, Hasty: bestDesc, HastyEq: bestEq, Gap: bestEq - t.equity,
				})
			}
		}
	}
	if err := g.PlayMove(m, false, 0); err != nil {
		return fmt.Errorf("game %s turn %d: play %q: %w", t.game, t.turn, t.play, err)
	}
	rp.hist = append(rp.hist, m)
	return nil
}

// dumpCandidates writes the top-50 static-equity candidates at the current
// position (plus the played move if it is not among them) with the exact
// input vectors the bot would send to the net: <prefix>.json (moves) and
// <prefix>-planes.bin / <prefix>-scalars.bin (float32 rows).
func dumpCandidates(rp *replay, d *dumpSpec, played *move.Move) error {
	rp.ai.MoveGenerator().SetPlayRecorder(movegen.AllPlaysRecorder)
	cands := rp.ai.GenerateMoves(50)
	copies := make([]*move.Move, 0, len(cands)+1)
	found := false
	for _, c := range cands {
		mc := &move.Move{}
		mc.CopyFrom(c)
		copies = append(copies, mc)
		if c.ShortDescription() == played.ShortDescription() {
			found = true
		}
	}
	if !found {
		copies = append(copies, played)
	}
	planes, scalars, err := rp.g.MLVectorsForMoves(copies, d.leaves, rp.hist)
	if err != nil {
		return err
	}
	type cand struct {
		Move   string  `json:"move"`
		Equity float64 `json:"equity"`
		Score  int     `json:"score"`
		Leave  string  `json:"leave"`
		Played bool    `json:"played"`
	}
	out := struct {
		CGP   string `json:"cgp"`
		Rack  string `json:"rack"`
		Cands []cand `json:"cands"`
	}{CGP: rp.g.ToCGP(false), Rack: rp.g.RackLettersFor(rp.g.PlayerOnTurn())}
	for _, c := range copies {
		out.Cands = append(out.Cands, cand{Move: c.ShortDescription(), Equity: c.Equity(), Score: c.Score(),
			Leave: c.LeaveString(), Played: c.ShortDescription() == played.ShortDescription()})
	}
	js, _ := json.MarshalIndent(out, "", " ")
	if err := os.WriteFile(d.prefix+".json", js, 0644); err != nil {
		return err
	}
	if err := writeFloats(d.prefix+"-planes.bin", planes); err != nil {
		return err
	}
	return writeFloats(d.prefix+"-scalars.bin", scalars)
}

func writeFloats(path string, v []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		u := math.Float32bits(x)
		buf[4*i], buf[4*i+1], buf[4*i+2], buf[4*i+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
	}
	_, err = f.Write(buf)
	return err
}

func (w *worker) flush(gameID string, rp *replay) {
	res := w.games[gameID]
	for _, d := range w.pending[gameID] {
		if rp.half < len(res) {
			d.MLSpread = res[rp.half].mlSpread
			d.MLWon = d.MLSpread > 0
		}
		d.MLTurns = w.mlMoves[gameID]
		w.out <- d
	}
	delete(w.pending, gameID)
	w.games_++
}

func main() {
	var turnsPath, gamesPath, mlNick, mlName, lexicon string
	var maxGames, threads int
	var dumpGame, dumpPrefix string
	var dumpHalf, dumpTurn int
	flag.StringVar(&dumpGame, "dump-game", "", "with -dump-half/-dump-turn/-dump-out: write the top-50 candidates' net inputs at that position")
	flag.IntVar(&dumpHalf, "dump-half", 0, "")
	flag.IntVar(&dumpTurn, "dump-turn", 0, "")
	flag.StringVar(&dumpPrefix, "dump-out", "dump", "output prefix")
	flag.StringVar(&turnsPath, "turns", "", "autoplay per-turn log")
	flag.StringVar(&gamesPath, "games", "", "autoplay per-game log")
	flag.StringVar(&mlNick, "bot", "p1", "nickname of the ML bot in the turn log (p1 = botcode1)")
	flag.StringVar(&mlName, "bot-name", "FastMlBot", "the ML bot's name in the games file columns")
	flag.StringVar(&lexicon, "lexicon", "NWL23", "lexicon")
	flag.IntVar(&maxGames, "max-games", 0, "stop after this many distinct game IDs (0 = all)")
	flag.IntVar(&threads, "threads", 4, "replay workers")
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)

	cfg := config.DefaultConfig()
	games, err := readGames(gamesPath, mlName)
	if err != nil {
		log.Fatal().Err(err).Msg("games file")
	}
	calc, err := equity.NewCombinedStaticCalculator(lexicon, cfg, "", "")
	if err != nil {
		log.Fatal().Err(err).Msg("equity")
	}
	calcs := []equity.EquityCalculator{calc}
	var dump *dumpSpec
	if dumpGame != "" {
		leaves, err := equity.NewExhaustiveLeaveCalculator(lexicon, cfg, "")
		if err != nil {
			log.Fatal().Err(err).Msg("leaves")
		}
		dump = &dumpSpec{game: dumpGame, half: dumpHalf, turn: dumpTurn, prefix: dumpPrefix, leaves: leaves}
	}

	f, err := os.Open(turnsPath)
	if err != nil {
		log.Fatal().Err(err).Msg("turn log")
	}
	defer f.Close()

	out := make(chan disagreement, 1024)
	jobs := make([]chan turnRow, threads)
	var wg sync.WaitGroup
	for i := range jobs {
		jobs[i] = make(chan turnRow, 4096)
		w := &worker{cfg: cfg, lexicon: lexicon, calcs: calcs, mlNick: mlNick, games: games, out: out,
			live: map[string]*replay{}, pending: map[string][]disagreement{}, mlMoves: map[string]int{}, dump: dump}
		wg.Add(1)
		go func(ch <-chan turnRow) {
			defer wg.Done()
			for t := range ch {
				if err := w.feed(t); err != nil {
					log.Fatal().Err(err).Msg("replay")
				}
			}
			for id, rp := range w.live {
				w.flush(id, rp)
			}
		}(jobs[i])
	}
	go func() {
		wg.Wait()
		close(out)
	}()

	go func() {
		r := csv.NewReader(bufio.NewReaderSize(f, 1<<20))
		r.FieldsPerRecord = -1
		r.TrimLeadingSpace = true
		if _, err := r.Read(); err != nil {
			log.Fatal().Err(err).Msg("header")
		}
		seen := map[string]bool{}
		for {
			rec, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal().Err(err).Msg("turn row")
			}
			if !seen[rec[1]] {
				if maxGames > 0 && len(seen) >= maxGames {
					break
				}
				seen[rec[1]] = true
			}
			tn, _ := strconv.Atoi(rec[2])
			eq, _ := strconv.ParseFloat(rec[9], 64)
			bag, _ := strconv.Atoi(rec[10])
			t := turnRow{player: rec[0], game: rec[1], turn: tn, rack: strings.TrimSpace(rec[3]),
				play: strings.TrimSpace(rec[4]), equity: eq, bag: bag}
			// One worker per game ID, so a game's turns stay in order.
			h := 0
			for _, c := range rec[1] {
				h = h*31 + int(c)
			}
			if h < 0 {
				h = -h
			}
			jobs[h%threads] <- t
		}
		for _, ch := range jobs {
			close(ch)
		}
	}()

	enc := json.NewEncoder(os.Stdout)
	n := 0
	for d := range out {
		if err := enc.Encode(d); err != nil {
			log.Fatal().Err(err).Msg("write")
		}
		n++
	}
	fmt.Fprintf(os.Stderr, "%d disagreements written\n", n)
}
