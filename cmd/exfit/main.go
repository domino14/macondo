// exfit tabulates, for every logged exchange (or, with -kind placement, tile
// placement) that was inferred, the static
// leave value of each leave the exchanger could have kept against its
// hypergeometric prior. It is the data for fitting a model of the form
//
//	P(kept = L) ∝ prior(L) · exp(β · value(L))
//
// offline, without a single sim: an exchanger keeps the best leave they can,
// and the leave table already knows which leaves are good.
//
// Each output line is one exchange: the true kept leave's value and log prior,
// and the prior mass of every possible kept leave binned by value. The bins
// are narrow enough (0.1 points) that the partition function computed from
// them is exact to well within the noise of a fit.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"

	"github.com/rs/zerolog"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/rangefinder"
	"github.com/domino14/word-golib/tilemapping"
)

const binWidth = 0.1

type record struct {
	GameID     string          `json:"gameID"`
	Half       int             `json:"half"`
	Turn       int             `json:"turn"`
	Play       string          `json:"play"`
	TrueLeave  string          `json:"trueLeave"`
	Kept       int             `json:"kept"`
	Played     int             `json:"played"` // tiles placed or thrown
	Score      int             `json:"score"`
	Bag        int             `json:"bag"`
	TrueValue  float64         `json:"trueValue"`
	TrueLogPri float64         `json:"trueLogPrior"`
	Leaves     int             `json:"leaves"`
	Hist       map[int]float64 `json:"hist"` // value bin -> prior mass
	LoggedLift *float64        `json:"loggedLift,omitempty"`
	Err        string          `json:"err,omitempty"`
}

func main() {
	turnlog := flag.String("turnlog", "", "per-turn log from a finished run (required)")
	lexicon := flag.String("lexicon", "NWL23", "lexicon")
	out := flag.String("out", "exfit.jsonl", "output")
	limit := flag.Int("positions", 0, "cap on positions (0 = all)")
	parallel := flag.Int("parallel", runtime.NumCPU(), "positions at once")
	kind := flag.String("kind", "exchange", "exchange or placement")
	flag.Parse()
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	if *turnlog == "" {
		flag.Usage()
		os.Exit(2)
	}
	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, *lexicon)
	cfg.Set(config.ConfigDefaultLetterDistribution, "english")

	corpus, skipped, err := automatic.LoadCorpusVerbose(cfg, *turnlog, *lexicon, "english", "",
		automatic.CorpusFilter{Exchanges: *kind == "exchange", Placements: *kind == "placement",
			AllLifts: true, Limit: *limit})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "%d positions (%d games skipped)\n", len(corpus), skipped)

	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	var mu sync.Mutex
	work := make(chan *automatic.CorpusPosition)
	var wg sync.WaitGroup
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			calc, err := equity.NewCombinedStaticCalculator(*lexicon, cfg, "", equity.PEGAdjustmentFilename)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			for pos := range work {
				rec := tabulate(pos, calc, cfg)
				mu.Lock()
				enc.Encode(rec)
				mu.Unlock()
			}
		}()
	}
	for _, pos := range corpus {
		work <- pos
	}
	close(work)
	wg.Wait()
}

func tabulate(pos *automatic.CorpusPosition, calc *equity.CombinedStaticCalculator,
	cfg *config.Config) record {

	rec := record{GameID: pos.GameID, Half: pos.Half, Turn: pos.Turn, Play: pos.OppPlay,
		TrueLeave: pos.TrueLeave, Kept: pos.LeaveLen, Bag: pos.TilesRemaining,
		Played: len([]rune(pos.OppRack)) - pos.LeaveLen, Score: pos.OppScore}
	if pos.HasLoggedResult {
		l := pos.LoggedLiftBits
		rec.LoggedLift = &l
	}
	g := pos.Game
	rf := &rangefinder.RangeFinder{}
	rf.Init(g, []equity.EquityCalculator{calc}, cfg)
	if err := rf.PrepareFinder(g.RackFor(g.PlayerOnTurn()).TilesOn()); err != nil {
		rec.Err = err.Error()
		return rec
	}
	bag := rf.BagMap()
	n := 0
	for _, c := range bag {
		n += int(c)
	}
	k := pos.LeaveLen
	logNK := logBinomial(n, k)

	truth := tilemapping.RackFromString(pos.TrueLeave, g.Alphabet())
	tt := truth.TilesOn()
	rec.TrueValue = calc.LeaveValue(sorted(tt))
	rec.TrueLogPri = -logNK
	for i, c := range truth.LetArr {
		if c > 0 {
			rec.TrueLogPri += logBinomial(int(bag[i]), int(c))
		}
	}

	rec.Hist = map[int]float64{}
	buf := make([]tilemapping.MachineLetter, 0, k)
	var recurse func(minIdx, remaining int, logP float64)
	recurse = func(minIdx, remaining int, logP float64) {
		if remaining == 0 {
			v := calc.LeaveValue(tilemapping.MachineWord(buf))
			rec.Hist[int(math.Round(v/binWidth))] += math.Exp(logP - logNK)
			rec.Leaves++
			return
		}
		for i := minIdx; i < len(bag); i++ {
			if bag[i] == 0 {
				continue
			}
			prev := len(buf)
			for cnt := 1; cnt <= int(bag[i]) && cnt <= remaining; cnt++ {
				buf = append(buf, tilemapping.MachineLetter(i))
				recurse(i+1, remaining-cnt, logP+logBinomial(int(bag[i]), cnt))
			}
			buf = buf[:prev]
		}
	}
	recurse(0, k, 0)
	return rec
}

func sorted(t []tilemapping.MachineLetter) tilemapping.MachineWord {
	w := make(tilemapping.MachineWord, len(t))
	copy(w, t)
	for i := 1; i < len(w); i++ {
		for j := i; j > 0 && w[j] < w[j-1]; j-- {
			w[j], w[j-1] = w[j-1], w[j]
		}
	}
	return w
}

func logBinomial(n, k int) float64 {
	if k < 0 || k > n {
		return math.Inf(-1)
	}
	a, _ := math.Lgamma(float64(n + 1))
	b, _ := math.Lgamma(float64(k + 1))
	c, _ := math.Lgamma(float64(n - k + 1))
	return a - b - c
}
