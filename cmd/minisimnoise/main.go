// minisimnoise measures how noisy the inference mini-sim is. For sampled
// inference positions it measures a few candidate leaves (the true one and
// random decoys) repeatedly with different seeds, under several mini-sim
// settings, and writes every measurement's per-play log-odds, so the noise in
// log P(play | leave) can be compared with the differences between leaves.
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/rs/zerolog"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/rangefinder"
)

type setting struct {
	Name   string `json:"name"`
	Iters  int    `json:"iters"`
	Moves  int    `json:"moves"`
	Equal  bool   `json:"equal"`
	Repeat int    `json:"repeat"`
}

type record struct {
	GameID    string    `json:"gameID"`
	Half      int       `json:"half"`
	Turn      int       `json:"turn"`
	Bag       int       `json:"bag"`
	LeaveLen  int       `json:"leaveLen"`
	Leave     string    `json:"leave"`
	IsTruth   bool      `json:"isTruth"`
	Setting   string    `json:"setting"`
	Rep       int       `json:"rep"`
	LogOdds   []float64 `json:"logOdds"`
	Samples   []int     `json:"samples"`
	TargetIdx int       `json:"targetIdx"`
	Micros    int64     `json:"us"`
	Err       string    `json:"err,omitempty"`
}

func main() {
	turnlog := flag.String("turnlog", "", "per-turn log from a finished run")
	out := flag.String("out", "noise.jsonl", "output")
	perBucket := flag.Int("per-bucket", 12, "positions per bag bucket")
	decoys := flag.Int("decoys", 3, "random leaves per position besides the truth")
	parallel := flag.Int("parallel", 16, "positions at once")
	settingsArg := flag.String("settings", "bai200:200:10:0:6,eq200:200:10:1:6,eq500:500:10:1:6,eq1000:1000:10:1:6,eq2000:2000:10:1:4",
		"name:iters:moves:equal(0/1):repeats, comma-separated")
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.Disabled)

	var settings []setting
	for _, f := range strings.Split(*settingsArg, ",") {
		var st setting
		var eq int
		parts := strings.Split(f, ":")
		if len(parts) != 5 {
			fmt.Fprintf(os.Stderr, "bad setting %q\n", f)
			os.Exit(2)
		}
		st.Name = parts[0]
		fmt.Sscan(parts[1], &st.Iters)
		fmt.Sscan(parts[2], &st.Moves)
		fmt.Sscan(parts[3], &eq)
		fmt.Sscan(parts[4], &st.Repeat)
		st.Equal = eq == 1
		settings = append(settings, st)
	}

	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, "NWL23")
	cfg.Set(config.ConfigDefaultLetterDistribution, "english")
	corpus, _, err := automatic.LoadCorpusVerbose(cfg, *turnlog, "NWL23", "english", "",
		automatic.CorpusFilter{Placements: true, AllLifts: true})
	if err != nil {
		panic(err)
	}
	// Even numbers of positions from each phase, picked at random.
	buckets := [][2]int{{1, 7}, {8, 20}, {21, 45}, {46, 100}}
	rng := rand.New(rand.NewSource(20261014))
	rng.Shuffle(len(corpus), func(i, j int) { corpus[i], corpus[j] = corpus[j], corpus[i] })
	var picked []*automatic.CorpusPosition
	for _, b := range buckets {
		n := 0
		for _, p := range corpus {
			if n < *perBucket && p.TilesRemaining >= b[0] && p.TilesRemaining <= b[1] {
				picked = append(picked, p)
				n++
			}
		}
	}
	fmt.Fprintf(os.Stderr, "%d positions\n", len(picked))

	calc, err := equity.NewCombinedStaticCalculator("NWL23", cfg, "", equity.PEGAdjustmentFilename)
	if err != nil {
		panic(err)
	}
	calcs := []equity.EquityCalculator{calc}

	f, err := os.Create(*out)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	var mu sync.Mutex
	enc := json.NewEncoder(w)
	emit := func(r record) {
		mu.Lock()
		enc.Encode(r)
		mu.Unlock()
	}

	work := make(chan *automatic.CorpusPosition)
	var done atomic.Int64
	var wg sync.WaitGroup
	for t := 0; t < *parallel; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for pos := range work {
				measurePosition(pos, cfg, calcs, settings, *decoys, emit)
				fmt.Fprintf(os.Stderr, "  %d/%d\n", done.Add(1), len(picked))
			}
		}()
	}
	for _, p := range picked {
		work <- p
	}
	close(work)
	wg.Wait()
}

func measurePosition(pos *automatic.CorpusPosition, cfg *config.Config, calcs []equity.EquityCalculator,
	settings []setting, decoys int, emit func(record)) {

	base := record{GameID: pos.GameID, Half: pos.Half, Turn: pos.Turn, Bag: pos.TilesRemaining,
		LeaveLen: pos.LeaveLen}
	g := pos.Game.CopyWithHistory()
	if sd, ok := seedFromGameID(pos.GameID); ok {
		g.SeedBag(sd)
	}
	rf := &rangefinder.RangeFinder{}
	rf.Init(g, calcs, cfg)
	rf.SetThreads(1)
	myRack := g.RackFor(g.PlayerOnTurn()).TilesOn()
	if err := rf.PrepareFinder([]tilemapping.MachineLetter(myRack)); err != nil {
		base.Err = err.Error()
		emit(base)
		return
	}
	alph := g.Alphabet()
	truth := tilemapping.RackFromString(pos.TrueLeave, alph).TilesOn()
	leaves := [][]tilemapping.MachineLetter{truth}
	pool := append([]tilemapping.MachineLetter{}, g.Bag().Peek()...)
	pool = append(pool, g.RackFor(1-g.PlayerOnTurn()).TilesOn()...)
	drng := rand.New(rand.NewSource(int64(pos.Turn)*7919 + int64(pos.Half)))
	for d := 0; d < decoys && len(pool) >= len(truth); d++ {
		drng.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
		l := append([]tilemapping.MachineLetter{}, pool[:len(truth)]...)
		sort.Slice(l, func(i, j int) bool { return l[i] < l[j] })
		leaves = append(leaves, l)
	}
	for li, leave := range leaves {
		for _, st := range settings {
			for rep := 0; rep < st.Repeat; rep++ {
				h := sha256.New()
				fmt.Fprintf(h, "%s|%d|%d|%d|%d", pos.GameID, pos.Half, pos.Turn, li, rep)
				var seed [32]byte
				copy(seed[:], h.Sum(nil))
				r := base
				r.Leave = tilemapping.MachineWord(leave).UserVisible(alph)
				r.IsTruth = li == 0
				r.Setting = st.Name
				r.Rep = rep
				m, err := rf.MeasureLeaveMiniSim(leave, seed, st.Iters, st.Moves, st.Equal)
				if err != nil {
					r.Err = err.Error()
				} else {
					r.LogOdds, r.Samples, r.TargetIdx = m.LogOdds, m.Samples, m.TargetIdx
					r.Micros = m.Elapsed.Microseconds()
				}
				emit(r)
			}
		}
	}
}

func seedFromGameID(gid string) ([32]byte, bool) {
	var out [32]byte
	raw, ok := strings.CutPrefix(gid, "seed:")
	if !ok {
		return out, false
	}
	b, err := base64.RawURLEncoding.DecodeString(raw)
	if err != nil || len(b) != 32 {
		return out, false
	}
	copy(out[:], b)
	return out, true
}

var _ = game.RackTileLimit
