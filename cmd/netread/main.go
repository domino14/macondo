// netread asks the value net how likely each leave the opponent might have
// kept makes the play they actually made, at positions from an autoplay turn
// log. It is the expensive half of a net-based opponent read, done once per
// position so that experiments on the CPU machines (which have no GPU) can
// load the result instead of running the net.
//
//	netread -turns base.txt -keys d3-placebo.jsonl -out reads.jsonl.gz
//
// A position is a key from screendecide's output, gid|first|row: the move at
// row `row` of the half that `first` started. The read is of the opponent's
// move one row earlier. For every leave they might have kept (all of them
// when there are at most -budget, otherwise -budget draws from the unseen
// pool), the opponent is given the rack played tiles + leave, their top -k
// static plays are generated, and the net scores those plays and the one
// they made, from their side. Raw values are written, not probabilities, so
// any likelihood shape or temperature can be applied afterwards.
//
// Every output line is one position:
//
//	{"key":..., "bag":31, "oppPlay":"8D TRAINEE", "k":5, "mode":"enum",
//	 "trueLeave":"AEINS", "trueIdx":12, "leaves":["AEINS",...],
//	 "w":[...], "v":[[actual, alt1, alt2, ...], ...]}
//
// w is each leave's prior weight (probability when enumerated, draw count
// when sampled); v[j][0] is the net's value of the actual play holding leave
// j and the rest are its alternatives' values, best static equity first; s,
// when the model has a spread head, is the same for its output.
// Exchanges, passes and plays that keep no tiles are recorded with a "skip"
// reason and no leaves.
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

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

type opts struct {
	k, budget int
}

type record struct {
	Key       string      `json:"key"`
	Bag       int         `json:"bag"` // tiles in the bag before the opponent's move
	OppPlay   string      `json:"oppPlay"`
	K         int         `json:"k"` // tiles the opponent kept
	Mode      string      `json:"mode,omitempty"`
	Distinct  int         `json:"distinct,omitempty"` // distinct leaves possible
	TrueLeave string      `json:"trueLeave"`
	TrueIdx   int         `json:"trueIdx"` // index in Leaves, -1 if not among them
	Leaves    []string    `json:"leaves,omitempty"`
	W         []float64   `json:"w,omitempty"`
	V         [][]float32 `json:"v,omitempty"`
	S         [][]float32 `json:"s,omitempty"` // the spread head, same layout, when the model has one
	Secs      float64     `json:"secs"`
	Skip      string      `json:"skip,omitempty"`
	Err       string      `json:"err,omitempty"`
}

// splitTurnHalves cuts a game's rows wherever the turn counter stops
// climbing: a game pair logs its two seatings under one game ID.
func splitTurnHalves(rows [][]string) [][][]string {
	halves := [][][]string{}
	start := 0
	for i := 1; i < len(rows); i++ {
		prev, err1 := strconv.Atoi(rows[i-1][2])
		cur, err2 := strconv.Atoi(rows[i][2])
		if err1 != nil || err2 != nil {
			continue
		}
		if cur <= prev {
			halves = append(halves, rows[start:i])
			start = i
		}
	}
	return append(halves, rows[start:])
}

// job is one half of a game and the rows in it to read, ascending.
type job struct {
	gid   string
	half  [][]string
	first string
	rows  []int
}

type worker struct {
	cfg     *config.Config
	lexicon string
	calcs   []equity.EquityCalculator
	leaves  *equity.ExhaustiveLeaveCalculator
	o       opts
}

func (w *worker) run(j job, out chan<- record) {
	other := "p2"
	if j.first == "p2" {
		other = "p1"
	}
	fail := func(from int, err error) {
		for _, i := range j.rows {
			if i >= from {
				out <- record{Key: key(j, i), TrueIdx: -1, Err: err.Error()}
			}
		}
	}
	rules, err := game.NewBasicGameRules(w.cfg, w.lexicon, board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		fail(0, err)
		return
	}
	tp, err := turnplayer.BaseTurnPlayerFromRules(
		&turnplayer.GameOptions{Variant: game.VarClassic, BoardLayoutName: board.CrosswordGameLayout},
		[]*pb.PlayerInfo{{Nickname: j.first, RealName: j.first}, {Nickname: other, RealName: other}}, rules)
	if err != nil {
		fail(0, err)
		return
	}
	ai, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(tp.Game, w.cfg, w.calcs)
	if err != nil {
		fail(0, err)
		return
	}
	ai.MoveGenerator().SetPlayRecorder(movegen.AllPlaysRecorder)
	g := tp.Game
	var hist []*move.Move
	played := 0 // rows applied so far
	for _, i := range j.rows {
		// The read is made before row i-1, the opponent's move.
		for played < i-1 {
			m, err := playRow(tp, j.half[played])
			if err != nil {
				fail(i, fmt.Errorf("row %d: %w", played, err))
				return
			}
			hist = append(hist, m)
			played++
		}
		out <- w.read(tp, ai, hist, j, i)
		if g.Playing() != pb.PlayState_PLAYING {
			return
		}
	}
}

func key(j job, i int) string { return fmt.Sprintf("%s|%s|%d", j.gid, j.first, i) }

func playRow(tp *turnplayer.BaseTurnPlayer, row []string) (*move.Move, error) {
	g := tp.Game
	if g.NickOnTurn() != row[0] {
		return nil, fmt.Errorf("%s to move but the log says %s", g.NickOnTurn(), row[0])
	}
	onturn := g.PlayerOnTurn()
	if err := g.SetRackFor(onturn, tilemapping.RackFromString(strings.TrimSpace(row[3]), g.Alphabet())); err != nil {
		return nil, err
	}
	m, err := tp.ParseMove(onturn, false, strings.Fields(stats.Normalize(strings.TrimSpace(row[4]))), false)
	if err != nil {
		return nil, fmt.Errorf("parse %q: %w", row[4], err)
	}
	if err := g.PlayMove(m, false, 0); err != nil {
		return nil, fmt.Errorf("play %q: %w", row[4], err)
	}
	return m, nil
}

func (w *worker) read(tp *turnplayer.BaseTurnPlayer, ai *aiturnplayer.AIStaticTurnPlayer,
	hist []*move.Move, j job, i int) (rec record) {

	start := time.Now()
	g := tp.Game
	alph := g.Alphabet()
	oppRow, myRow := j.half[i-1], j.half[i]
	rec = record{Key: key(j, i), OppPlay: strings.TrimSpace(oppRow[4]), TrueIdx: -1,
		TrueLeave: sortedWord(strings.TrimSpace(oppRow[8]), alph)}
	defer func() { rec.Secs = time.Since(start).Seconds() }()
	if g.NickOnTurn() != oppRow[0] || myRow[0] == oppRow[0] {
		rec.Err = "the opponent is not on turn"
		return rec
	}
	opp := g.PlayerOnTurn()
	trueRack := tilemapping.RackFromString(strings.TrimSpace(oppRow[3]), alph)
	myRack := tilemapping.RackFromString(strings.TrimSpace(myRow[3]), alph)
	seat := func(oppRack *tilemapping.Rack) error {
		racks := make([]*tilemapping.Rack, 2)
		racks[opp], racks[1-opp] = oppRack, myRack
		return g.SetRacksForBoth(racks)
	}
	if err := seat(trueRack); err != nil {
		rec.Err = "racks: " + err.Error()
		return rec
	}
	rec.Bag = g.Bag().TilesRemaining()
	fields := strings.Fields(stats.Normalize(rec.OppPlay))
	actual, err := tp.ParseMove(opp, false, fields, false)
	if err != nil {
		rec.Err = "parse: " + err.Error()
		return rec
	}
	switch actual.Action() {
	case move.MoveTypePlay:
	case move.MoveTypeExchange:
		rec.Skip = "exchange"
		return rec
	default:
		rec.Skip = "pass"
		return rec
	}
	var playedTiles []tilemapping.MachineLetter
	for _, t := range actual.Tiles() {
		if t != 0 { // 0 marks a played-through square
			playedTiles = append(playedTiles, t.IntrinsicTileIdx())
		}
	}
	rec.K = int(trueRack.NumTiles()) - len(playedTiles)
	if rec.K <= 0 {
		rec.Skip = "kept nothing"
		return rec
	}
	// The pool the leave came from: everything the opponent could not see
	// but its own rack, minus the tiles it played.
	pool := make([]int, alph.NumLetters())
	for ml, c := range g.Bag().PeekMap() {
		pool[ml] += int(c)
	}
	for _, t := range trueRack.TilesOn() {
		pool[t]++
	}
	for _, t := range playedTiles {
		pool[t]--
		if pool[t] < 0 {
			rec.Err = "played tiles not in the pool"
			return rec
		}
	}

	leaves, weights, distinct, mode := w.candidateLeaves(pool, rec.K, rec.Key)
	rec.Mode, rec.Distinct = mode, distinct
	for li, l := range leaves {
		rack := tilemapping.NewRack(alph)
		rack.Set(append(append([]tilemapping.MachineLetter{}, playedTiles...), l...))
		if err := seat(rack); err != nil {
			rec.Err = "seat leave: " + err.Error()
			return rec
		}
		cands := ai.GenerateMoves(w.o.k)
		moves := make([]*move.Move, 0, len(cands)+1)
		mine, err := tp.ParseMove(opp, false, fields, false) // its leave is this rack's
		if err != nil {
			rec.Err = "parse under leave: " + err.Error()
			return rec
		}
		moves = append(moves, mine)
		desc := mine.ShortDescription()
		for _, c := range cands {
			if c.ShortDescription() == desc {
				continue
			}
			mc := &move.Move{}
			mc.CopyFrom(c) // the generator reuses its move objects
			moves = append(moves, mc)
		}
		outs, err := g.MLEvaluateMoves(moves, w.leaves, hist)
		if err != nil {
			rec.Err = "net: " + err.Error()
			return rec
		}
		v := make([]float32, len(moves))
		for mi := range moves {
			v[mi] = float32(math.Round(float64(outs.Value[mi])*1e4) / 1e4)
		}
		if len(outs.Spread) == len(moves) {
			sp := make([]float32, len(moves))
			for mi := range moves {
				sp[mi] = float32(math.Round(float64(outs.Spread[mi])*1e4) / 1e4)
			}
			rec.S = append(rec.S, sp)
		}
		word := tilemapping.MachineWord(l).UserVisible(alph)
		if word == rec.TrueLeave {
			rec.TrueIdx = li
		}
		rec.Leaves = append(rec.Leaves, word)
		rec.W = append(rec.W, weights[li])
		rec.V = append(rec.V, v)
	}
	return rec
}

func sortedWord(s string, alph *tilemapping.TileMapping) string {
	mw, err := tilemapping.ToMachineWord(s, alph)
	if err != nil {
		return s
	}
	sort.Slice(mw, func(a, b int) bool { return mw[a] < mw[b] })
	return mw.UserVisible(alph)
}

// candidateLeaves lists every multiset of k tiles from the pool with its
// hypergeometric probability when there are at most budget of them, and
// otherwise budget random draws, merged, with their counts as weights.
// Leaves come back sorted by tile.
func (w *worker) candidateLeaves(pool []int, k int, key string) ([][]tilemapping.MachineLetter, []float64, int, string) {
	distinct := countMultisets(pool, k)
	if distinct <= w.o.budget {
		var leaves [][]tilemapping.MachineLetter
		var logw []float64
		cur := make([]tilemapping.MachineLetter, 0, k)
		var rec func(t, left int, lw float64)
		rec = func(t, left int, lw float64) {
			if left == 0 {
				leaves = append(leaves, append([]tilemapping.MachineLetter{}, cur...))
				logw = append(logw, lw)
				return
			}
			if t == len(pool) {
				return
			}
			for c := 0; c <= min(pool[t], left); c++ {
				for range c {
					cur = append(cur, tilemapping.MachineLetter(t))
				}
				rec(t+1, left-c, lw+logChoose(pool[t], c))
				cur = cur[:len(cur)-c]
			}
		}
		rec(0, k, 0)
		total := 0.0
		for _, lw := range logw {
			total += math.Exp(lw)
		}
		ws := make([]float64, len(logw))
		for i, lw := range logw {
			ws[i] = math.Exp(lw) / total
		}
		return leaves, ws, distinct, "enum"
	}
	h := fnv.New64a()
	h.Write([]byte(key))
	rng := rand.New(rand.NewPCG(h.Sum64(), 0x6e6574726561640a))
	var bag []tilemapping.MachineLetter
	for t, c := range pool {
		for range c {
			bag = append(bag, tilemapping.MachineLetter(t))
		}
	}
	counts := map[string]int{}
	byKey := map[string][]tilemapping.MachineLetter{}
	for range w.o.budget {
		for x := range k { // partial Fisher-Yates: the first k are the draw
			y := x + rng.IntN(len(bag)-x)
			bag[x], bag[y] = bag[y], bag[x]
		}
		l := append([]tilemapping.MachineLetter{}, bag[:k]...)
		sort.Slice(l, func(a, b int) bool { return l[a] < l[b] })
		s := string(l)
		counts[s]++
		byKey[s] = l
	}
	keys := make([]string, 0, len(counts))
	for s := range counts {
		keys = append(keys, s)
	}
	sort.Strings(keys)
	leaves := make([][]tilemapping.MachineLetter, len(keys))
	ws := make([]float64, len(keys))
	for i, s := range keys {
		leaves[i], ws[i] = byKey[s], float64(counts[s])
	}
	return leaves, ws, distinct, "sample"
}

// countMultisets counts the multisets of k tiles the pool allows.
func countMultisets(pool []int, k int) int {
	ways := make([]int, k+1) // ways[j]: multisets of j tiles from the letters so far
	ways[0] = 1
	for _, n := range pool {
		next := make([]int, k+1)
		for j := 0; j <= k; j++ {
			for c := 0; c <= min(n, j); c++ {
				next[j] += ways[j-c]
			}
		}
		ways = next
	}
	return ways[k]
}

func logChoose(n, c int) float64 {
	a, _ := math.Lgamma(float64(n + 1))
	b, _ := math.Lgamma(float64(c + 1))
	d, _ := math.Lgamma(float64(n - c + 1))
	return a - b - d
}

func readKeys(path string, limit int) (map[string]bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	keys := map[string]bool{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		var r struct {
			Key string `json:"key"`
			Err string `json:"err"`
		}
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, err
		}
		if r.Key != "" && r.Err == "" {
			keys[r.Key] = true
			if limit > 0 && len(keys) >= limit {
				break
			}
		}
	}
	return keys, sc.Err()
}

func readJobs(path string, keys map[string]bool) ([]job, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(bufio.NewReaderSize(f, 1<<20))
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
	var jobs []job
	for gid, rows := range byGame {
		for _, h := range splitTurnHalves(rows) {
			j := job{gid: gid, half: h, first: h[0][0]}
			for i := 1; i < len(h); i++ {
				if keys[key(j, i)] {
					j.rows = append(j.rows, i)
				}
			}
			if len(j.rows) > 0 {
				jobs = append(jobs, j)
			}
		}
	}
	sort.Slice(jobs, func(a, b int) bool {
		if jobs[a].gid != jobs[b].gid {
			return jobs[a].gid < jobs[b].gid
		}
		return jobs[a].first < jobs[b].first
	})
	return jobs, nil
}

func main() {
	turnsPath := flag.String("turns", "", "autoplay per-turn log the positions come from")
	keysPath := flag.String("keys", "", "screendecide output whose keys (gid|first|row) name the positions")
	outPath := flag.String("out", "", "output JSONL (gzipped if it ends in .gz)")
	lexicon := flag.String("lexicon", "NWL23", "lexicon")
	k := flag.Int("k", 20, "static alternatives per leave, besides the actual play")
	budget := flag.Int("budget", 1000, "list every leave when there are at most this many, otherwise draw this many")
	threads := flag.Int("threads", 8, "positions read at once")
	limit := flag.Int("n", 0, "read only the first n keys (0 = all)")
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	if *turnsPath == "" || *keysPath == "" || *outPath == "" {
		fmt.Fprintln(os.Stderr, "netread -turns LOG -keys DECISIONS -out READS")
		os.Exit(2)
	}

	cfg := config.DefaultConfig()
	keys, err := readKeys(*keysPath, *limit)
	if err != nil {
		log.Fatal().Err(err).Msg("keys")
	}
	jobs, err := readJobs(*turnsPath, keys)
	if err != nil {
		log.Fatal().Err(err).Msg("turn log")
	}
	calc, err := equity.NewCombinedStaticCalculator(*lexicon, cfg, "", "")
	if err != nil {
		log.Fatal().Err(err).Msg("equity")
	}
	leaves, err := equity.NewExhaustiveLeaveCalculator(*lexicon, cfg, "")
	if err != nil {
		log.Fatal().Err(err).Msg("leaves")
	}
	o := opts{k: *k, budget: *budget}

	f, err := os.Create(*outPath)
	if err != nil {
		log.Fatal().Err(err).Msg("out")
	}
	var wr io.Writer = f
	var gz *gzip.Writer
	if strings.HasSuffix(*outPath, ".gz") {
		gz = gzip.NewWriter(f)
		wr = gz
	}
	bw := bufio.NewWriterSize(wr, 1<<20)

	out := make(chan record, 256)
	ch := make(chan job)
	var wg sync.WaitGroup
	for range *threads {
		wk := &worker{cfg: cfg, lexicon: *lexicon, calcs: []equity.EquityCalculator{calc}, leaves: leaves, o: o}
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range ch {
				wk.run(j, out)
			}
		}()
	}
	go func() {
		for _, j := range jobs {
			ch <- j
		}
		close(ch)
		wg.Wait()
		close(out)
	}()

	enc := json.NewEncoder(bw)
	n, skipped, failed := 0, 0, 0
	start := time.Now()
	for r := range out {
		if err := enc.Encode(r); err != nil {
			log.Fatal().Err(err).Msg("write")
		}
		n++
		switch {
		case r.Err != "":
			failed++
		case r.Skip != "":
			skipped++
		}
		if n%100 == 0 {
			fmt.Fprintf(os.Stderr, "%d/%d positions (%d skipped, %d failed), %.0f s\n", n, len(keys), skipped, failed, time.Since(start).Seconds())
		}
	}
	if err := bw.Flush(); err != nil {
		log.Fatal().Err(err).Msg("flush")
	}
	if gz != nil {
		gz.Close()
	}
	f.Close()
	fmt.Fprintf(os.Stderr, "done: %d positions (%d skipped, %d failed) in %.0f s\n", n, skipped, failed, time.Since(start).Seconds())
}
