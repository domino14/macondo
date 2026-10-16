package main

import (
	"flag"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/playability"
	"github.com/domino14/macondo/wmp"
)

// SumStat accumulates regrets over the synthetic samples of one word or
// alphagram.
type SumStat struct {
	Sum  float64
	Sum2 float64
	Hits int64 // samples with regret > 0
}

func (s *SumStat) add(r float64) {
	s.Sum += r
	s.Sum2 += r * r
	if r > 0 {
		s.Hits++
	}
}

// Stratum is estimator B for one alphagram over one band of the event "the
// rack covers c of the word's letters": stratum 0 is the full word (c =
// min(len,7)), stratum j is exactly j short. Per turn, the stratum's expected
// regret is E[p * r], where p is the chance the mover's draw lands in the band at
// a saved turn and r is the regret given that it did. It is estimated as
// mean(p) over M uniformly chosen saved turns, times mean(r) over K turns drawn
// from those M in proportion to p, each re-dealt conditionally on the band.
type Stratum struct {
	M     int64
	PSum  float64
	PSum2 float64
	K     int64
	Alpha SumStat
	Words map[playability.Key]*SumStat
}

// SynthResult sums the strata, which are disjoint.
type SynthResult struct {
	Strata []*Stratum
}

// PerTurn returns the per-turn estimate for a word (or the alphagram, w == ""),
// with a delta-method standard error, summed over strata.
func (r *SynthResult) PerTurn(w playability.Key) (est, se float64, hits int64) {
	v := 0.0
	for _, st := range r.Strata {
		if st.M == 0 || st.K == 0 {
			continue
		}
		s := &st.Alpha
		if w != "" {
			s = st.Words[w]
		}
		mp := st.PSum / float64(st.M)
		vp := math.Max(0, st.PSum2/float64(st.M)-mp*mp) / float64(st.M)
		mr := s.Sum / float64(st.K)
		vr := math.Max(0, s.Sum2/float64(st.K)-mr*mr) / float64(st.K)
		est += mp * mr
		v += mr*mr*vp + mp*mp*vr
		hits += s.Hits
	}
	return est, math.Sqrt(v), hits
}

// PMean is the chance a turn's rack qualifies for any stratum.
func (r *SynthResult) PMean() float64 {
	p := 0.0
	for _, st := range r.Strata {
		if st.M > 0 {
			p += st.PSum / float64(st.M)
		}
	}
	return p
}

func (r *SynthResult) K() int64 {
	var k int64
	for _, st := range r.Strata {
		k += st.K
	}
	return k
}

type SynthChunk struct {
	Index   int
	Slack   int
	Results map[playability.Key]*SynthResult
}

func synthChunkPath(dir string, idx int) string {
	return filepath.Join(dir, fmt.Sprintf("synth-%06d.gob.gz", idx))
}

// lexiconClasses returns the lexicon's words grouped by alphagram.
func lexiconClasses(cfg *config.Config, lexicon string) (map[playability.Key][]playability.Key, *tilemapping.TileMapping, error) {
	k, err := kwg.GetKWG(cfg.WGLConfig(), lexicon)
	if err != nil {
		return nil, nil, err
	}
	classes := map[playability.Key][]playability.Key{}
	for _, mw := range wmp.ExtractWordsFromKWG(k, 15) {
		w := playability.KeyOf(mw)
		a := playability.AlphaKey(w)
		classes[a] = append(classes[a], w)
	}
	return classes, k.GetAlphabet(), nil
}

func loadPositions(dir string, max int) ([]playability.Position, error) {
	files, err := filepath.Glob(filepath.Join(dir, "shard-*.gob.gz"))
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	var pos []playability.Position
	for _, f := range files {
		var sh playability.Shard
		if err := playability.ReadGob(f, &sh); err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		pos = append(pos, sh.Positions...)
		if max > 0 && len(pos) >= max {
			return pos[:max], nil
		}
	}
	return pos, nil
}

func runSynth(args []string) error {
	fs := flag.NewFlagSet("synth", flag.ExitOnError)
	in := fs.String("in", "", "gen shard directory (positions come from here)")
	out := fs.String("out", "", "output directory for synth chunks")
	k := fs.Int("k", 300, "re-dealt turns (movegens) per alphagram")
	mPos := fs.Int("m", 3000, "saved turns per chunk over which each alphagram's draw probability is averaged")
	minLen := fs.Int("minlen", 7, "shortest alphagram")
	maxLen := fs.Int("maxlen", 15, "longest alphagram")
	chunk := fs.Int("chunk", 200, "alphagrams per chunk (keep fixed for a run)")
	maxChunks := fs.Int("max-chunks", 0, "stop after this many chunks (0 = all)")
	stride := fs.Int("stride", 1, "only do every stride-th chunk (quick coverage pass)")
	slack := fs.Int("slack", 1, "strata: the full word, then racks 1..slack letters short of it")
	maxPos := fs.Int("max-positions", 500000, "positions to load")
	workers := fs.Int("workers", 3, "worker goroutines")
	lexicon := fs.String("lexicon", "NWL23", "lexicon")
	ld := fs.String("ld", "english", "letter distribution")
	only := fs.String("words", "", "comma-separated words: do just their alphagrams (testing)")
	fs.Parse(args)
	if *in == "" || *out == "" {
		return fmt.Errorf("-in and -out are required")
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		return err
	}
	cfg := newConfig(*lexicon, *ld)
	positions, err := loadPositions(*in, *maxPos)
	if err != nil {
		return err
	}
	if len(positions) == 0 {
		return fmt.Errorf("no positions in %s", *in)
	}
	classes, tm, err := lexiconClasses(cfg, *lexicon)
	if err != nil {
		return err
	}
	var alphas []playability.Key
	if *only != "" {
		for _, w := range strings.Split(*only, ",") {
			mw, err := tilemapping.ToMachineWord(strings.ToUpper(w), tm)
			if err != nil {
				return err
			}
			a := playability.AlphaKey(playability.KeyOf(mw))
			if _, ok := classes[a]; !ok {
				return fmt.Errorf("%s is not in the lexicon", w)
			}
			alphas = append(alphas, a)
		}
	} else {
		for a := range classes {
			if len(a) >= *minLen && len(a) <= *maxLen {
				alphas = append(alphas, a)
			}
		}
	}
	sort.Slice(alphas, func(i, j int) bool {
		if len(alphas[i]) != len(alphas[j]) {
			return len(alphas[i]) < len(alphas[j])
		}
		return alphas[i] < alphas[j]
	})
	nChunks := (len(alphas) + *chunk - 1) / *chunk
	var todo []int
	for c := 0; c < nChunks; c += *stride {
		if _, err := os.Stat(synthChunkPath(*out, c)); err != nil {
			todo = append(todo, c)
		}
		if *maxChunks > 0 && len(todo) >= *maxChunks {
			break
		}
	}
	log.Info().Int("positions", len(positions)).Int("alphagrams", len(alphas)).
		Int("chunks", nChunks).Int("todo", len(todo)).Msg("synth")

	jobs := make(chan int)
	go func() {
		for _, c := range todo {
			jobs <- c
		}
		close(jobs)
	}()
	start := time.Now()
	var done atomic.Int64
	var wg sync.WaitGroup
	errc := make(chan error, *workers)
	for wi := 0; wi < *workers; wi++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range jobs {
				lo, hi := c**chunk, min((c+1)**chunk, len(alphas))
				res, err := synthChunk(cfg, positions, alphas[lo:hi], classes, c, *mPos, *k, *slack)
				if err != nil {
					errc <- err
					return
				}
				if err := playability.WriteAtomic(synthChunkPath(*out, c), res); err != nil {
					errc <- err
					return
				}
				n := done.Add(1)
				log.Info().Int("chunk", c).Int("len", len(alphas[lo])).Int64("done", n).
					Int("todo", len(todo)).
					Float64("chunks/h", float64(n)/time.Since(start).Hours()).Msg("chunk-done")
			}
		}()
	}
	wg.Wait()
	close(errc)
	return <-errc
}

// prepared is a saved turn reduced to what the conditioned draw needs.
type prepared struct {
	unseen  []int
	leaveCt []int
	leave   []tilemapping.MachineLetter
	drawn   int
}

func prepare(cfg *config.Config, pos playability.Position) (*prepared, error) {
	parsed, err := cgp.ParseCGP(cfg, pos.CGP)
	if err != nil {
		return nil, err
	}
	g := parsed.Game
	tm := g.Alphabet()
	nl := int(tm.NumLetters())
	leave, err := tilemapping.ToMachineLetters(pos.Leave, tm)
	if err != nil {
		return nil, err
	}
	pr := &prepared{unseen: make([]int, nl), leaveCt: make([]int, nl), leave: leave, drawn: pos.Drawn}
	for ml, n := range g.Bag().PeekMap() {
		if ml < nl {
			pr.unseen[ml] += int(n)
		}
	}
	for p := 0; p < 2; p++ {
		for _, t := range g.RackFor(p).TilesOn() {
			pr.unseen[t.IntrinsicTileIdx()]++
		}
	}
	for _, t := range leave {
		pr.leaveCt[t.IntrinsicTileIdx()]++
		pr.unseen[t.IntrinsicTileIdx()]--
	}
	return pr, nil
}

type synthJob struct {
	ai, stratum int
	draw        []int
}

func synthChunk(cfg *config.Config, positions []playability.Position, alphas []playability.Key,
	classes map[playability.Key][]playability.Key, idx, m, k, slack int) (*SynthChunk, error) {

	h := fnv.New64a()
	fmt.Fprintf(h, "synth/%d", idx)
	rng := rand.New(rand.NewPCG(h.Sum64(), uint64(slack)))
	res := &SynthChunk{Index: idx, Slack: slack, Results: map[playability.Key]*SynthResult{}}

	// The chunk's M turns, shared by all its alphagrams.
	m = min(m, len(positions))
	posIdx := make([]int, m)
	preps := make([]*prepared, m)
	for i := range posIdx {
		posIdx[i] = rng.IntN(len(positions))
		var err error
		if preps[i], err = prepare(cfg, positions[posIdx[i]]); err != nil {
			return nil, err
		}
	}

	jobs := map[int][]synthJob{} // by index into posIdx
	cum := make([]float64, m)
	for ai, a := range alphas {
		r := &SynthResult{}
		res.Results[a] = r
		full := min(len(a), 7)
		for sj := 0; sj <= slack && full-sj >= 1; sj++ {
			st := &Stratum{Words: map[playability.Key]*SumStat{}}
			for _, w := range classes[a] {
				st.Words[w] = &SumStat{}
			}
			r.Strata = append(r.Strata, st)
			need, most := full-sj, full-sj
			if sj == 0 {
				most = 99
			}
			s := 0.0
			for i, pr := range preps {
				p := playability.NewCondDraw(pr.unseen, pr.leaveCt, pr.drawn, []byte(a), need, most).Prob
				st.M++
				st.PSum += p
				st.PSum2 += p * p
				s += p
				cum[i] = s
			}
			if s == 0 {
				continue
			}
			for j := 0; j < k; j++ {
				i := sort.SearchFloat64s(cum, rng.Float64()*s)
				for i < m-1 && cum[i] == 0 {
					i++
				}
				pr := preps[i]
				cd := playability.NewCondDraw(pr.unseen, pr.leaveCt, pr.drawn, []byte(a), need, most)
				jobs[i] = append(jobs[i], synthJob{ai: ai, stratum: sj, draw: cd.Sample(rng)})
			}
		}
	}

	order := make([]int, 0, len(jobs))
	for i := range jobs {
		order = append(order, i)
	}
	sort.Ints(order)
	var ev playability.Evaluator
	debug := os.Getenv("SYNTH_DEBUG") != ""
	for _, i := range order {
		parsed, err := cgp.ParseCGP(cfg, positions[posIdx[i]].CGP)
		if err != nil {
			return nil, err
		}
		g := parsed.Game
		// A parsed board has no cross-sets or anchors until asked for them.
		g.RecalculateBoard()
		wk, err := newWorker(cfg, g)
		if err != nil {
			return nil, err
		}
		tm := g.Alphabet()
		pi := g.PlayerOnTurn()
		for _, jb := range jobs[i] {
			a := alphas[jb.ai]
			r := res.Results[a].Strata[jb.stratum]
			r.K++
			rackTiles := append([]tilemapping.MachineLetter(nil), preps[i].leave...)
			for _, t := range jb.draw {
				rackTiles = append(rackTiles, tilemapping.MachineLetter(t))
			}
			rack := tilemapping.NewRack(tm)
			rack.Set(rackTiles)
			if err := g.SetRackFor(pi, rack); err != nil {
				return nil, err
			}
			plays := wk.btp.GenerateMoves(math.MaxInt32)
			b := g.Board()
			ra := ev.RegretFor(b, plays, func(f playability.Key) bool {
				return len(f) == len(a) && playability.AlphaKey(f) == a
			})
			if debug {
				fmt.Fprintf(os.Stderr, "%s rack=%s best=%s eq=%.1f regret=%.1f\n", playability.Visible(a, tm),
					rack.String(), plays[0].ShortDescription(), plays[0].Equity(), ra)
			}
			r.Alpha.add(ra)
			for w, st := range r.Words {
				if ra == 0 {
					st.add(0) // no anagram in the best move: no single word can matter
					continue
				}
				st.add(ev.RegretFor(b, plays, func(f playability.Key) bool { return f == w }))
			}
		}
	}
	return res, nil
}
