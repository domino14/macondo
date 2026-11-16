package montecarlo

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/rs/zerolog"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/montecarlo/bai"
)

// The stopping lab replays stopping rules over rollout tables
// (sim_table_test.go). A replay draws its iterations from a table at random,
// with replacement, so one table gives any number of sims of any length,
// and the table's own means are exactly the means of what a replay samples:
// the truth each sim's pick is judged against.
//
//	STOPLAB_DIR=dir STOPLAB_OUT=file [STOPLAB_REPS=20 STOPLAB_WORKERS=16
//	  STOPLAB_K=15,50 STOPLAB_ALGOS=...]
//
// One line is written per replay.

type labRun struct {
	samples int
	pick    int
	status  string
}

type labTable struct {
	*simTable
	k                         int
	meanWin, meanEq, meanUtil []float64
	similar                   []bool
}

// labUtility is MAGPIE's default sim utility: win probability blended with
// a sigmoid of the final spread (here spread plus leftover).
func labUtility(win, eq float32, initialSpread int) float64 {
	return labUtilityWeighted(win, eq, initialSpread, 0.5)
}

// labUtilityWeighted is the utility with another weight on the spread term
// (the win probability's weight is 1).
func labUtilityWeighted(win, eq float32, initialSpread int, spreadWeight float64) float64 {
	x := (float64(eq) + float64(initialSpread)) / 100
	var sig float64
	if x >= 0 {
		sig = 1 / (1 + math.Exp(-x))
	} else {
		e := math.Exp(x)
		sig = e / (1 + e)
	}
	return (float64(win) + spreadWeight*sig) / (1 + spreadWeight)
}

func loadLabTable(path string) (*labTable, error) {
	tab, err := readSimTable(path)
	if err != nil {
		return nil, err
	}
	parsed, err := cgp.ParseCGP(DefaultConfig, tab.CGP)
	if err != nil {
		return nil, err
	}
	parsed.RecalculateBoard()
	calcs, _ := defaultSimCalculators(tab.Lexicon)
	p, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(parsed.Game, DefaultConfig, calcs)
	if err != nil {
		return nil, err
	}
	tab.moves = p.GenerateMoves(tab.K)
	if len(tab.moves) != tab.K {
		return nil, fmt.Errorf("%s: %d moves, table has %d", path, len(tab.moves), tab.K)
	}
	for i, m := range tab.moves {
		if m.ShortDescription() != tab.Plays[i] {
			return nil, fmt.Errorf("%s: move %d is %s, table has %s", path, i, m.ShortDescription(), tab.Plays[i])
		}
	}
	return &labTable{simTable: tab}, nil
}

// restrict makes the table's first k plays the candidates.
func (lt *labTable) restrict(k int) {
	k = min(k, lt.K)
	lt.k = k
	lt.meanWin = make([]float64, k)
	lt.meanEq = make([]float64, k)
	lt.meanUtil = make([]float64, k)
	for i := 0; i < k; i++ {
		var w, e, u float64
		for j := 0; j < lt.N; j++ {
			w += float64(lt.win[i][j])
			e += float64(lt.eq[i][j])
			u += labUtility(lt.win[i][j], lt.eq[i][j], lt.InitialSpread)
		}
		lt.meanWin[i], lt.meanEq[i], lt.meanUtil[i] = w/float64(lt.N), e/float64(lt.N), u/float64(lt.N)
	}
	keys := make([]string, k)
	for i := range keys {
		keys[i] = similarPlayKey(lt.moves[i])
	}
	lt.similar = make([]bool, k*k)
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			lt.similar[i*k+j] = i != j && keys[i] == keys[j]
		}
	}
}

// labPlays builds the sim's play list for a replay.
func (lt *labTable) labPlays() (*Simmer, []*SimmedPlay) {
	plays := make([]*SimmedPlay, lt.k)
	for i := range plays {
		plays[i] = &SimmedPlay{play: lt.moves[i]}
	}
	s := &Simmer{simmedPlays: &SimmedPlays{plays: append([]*SimmedPlay(nil), plays...)}}
	return s, plays
}

func labWinner(s *Simmer, plays []*SimmedPlay) int {
	s.sortPlaysByWinRate(true)
	best := s.simmedPlays.plays[0]
	for i, p := range plays {
		if p == best {
			return i
		}
	}
	return -1
}

// runAutostop replays Macondo's stopping rule: every play still in the sim
// is rolled out once an iteration, and the rule is consulted every
// checkInterval iterations. A negative condition means no rule: uniform
// sampling for fixedIters iterations. A positive capIters replaces the
// rule's cap on a sim's length.
func (lt *labTable) runAutostop(perm []int32, sc StoppingCondition, fixedIters, capIters int) labRun {
	s, plays := lt.labPlays()
	a := newAutostopper()
	a.stoppingCondition = sc
	if capIters > 0 {
		// The rule stops a sim past iterationsCutoff + plies*perPlyStopScaling.
		a.iterationsCutoff = capIters
		a.perPlyStopScaling = 0
	}
	samples := 0
	status := "table"
	for it := 1; it <= lt.N; it++ {
		col := perm[it-1]
		for i, p := range plays {
			if p.ignore.Load() {
				continue
			}
			p.winPctStats.Push(float64(lt.win[i][col]))
			p.equityStats.Push(float64(lt.eq[i][col]))
			samples++
		}
		if fixedIters > 0 {
			if it >= fixedIters {
				status = "fixed"
				break
			}
			continue
		}
		if uint64(it)%a.stopConditionCheckInterval == 0 && a.shouldStop(uint64(it), s.simmedPlays, lt.Plies) {
			status = "stop"
			if it > a.iterationsCutoff+lt.Plies*a.perPlyStopScaling {
				status = "limit"
			}
			break
		}
	}
	return labRun{samples: samples, pick: labWinner(s, plays), status: status}
}

type labBAIConfig struct {
	delta    float64
	minimum  int
	utility  bool // MAGPIE's blended utility; else win probability
	paired   bool
	similar  bool
	limitPer int // sample limit, in samples per play
	// spreadWeight, if not 0, replaces the utility's weight on spread.
	spreadWeight float64
	roundSize    int
}

// runBAI replays best-arm identification. An arm's samples are the table's
// rollouts of that play in the replay's order, so equal ordinals of two arms
// are the same iteration, as BAI's paired statistics assume.
func (lt *labTable) runBAI(perm []int32, c labBAIConfig) labRun {
	opts := bai.Options{
		Delta:         c.delta,
		SampleMinimum: c.minimum,
		Cutoff:        0.005 / 100,
		Threads:       1,
		Paired:        c.paired,
		RoundSize:     c.roundSize,
	}
	if c.similar {
		k := lt.k
		opts.Similar = func(a, b int) bool { return lt.similar[a*k+b] }
	}
	opts.SampleLimit = c.limitPer * lt.k
	exhausted := false
	res, err := bai.Run(context.Background(), lt.k, opts, func(_, arm int, ordinal uint64) (float64, error) {
		if ordinal >= uint64(len(perm)) {
			exhausted = true
			return 0, fmt.Errorf("draws exhausted")
		}
		col := perm[ordinal]
		if c.utility {
			w := c.spreadWeight
			if w == 0 {
				w = 0.5
			}
			return labUtilityWeighted(lt.win[arm][col], lt.eq[arm][col], lt.InitialSpread, w), nil
		}
		return float64(lt.win[arm][col]), nil
	})
	if exhausted {
		return labRun{samples: -1, pick: -1, status: "exhausted"}
	}
	if err != nil {
		panic(err)
	}
	return labRun{samples: res.Samples, pick: res.Best, status: res.Status.String()}
}

type labAlgo struct {
	name string
	run  func(lt *labTable, perm []int32) labRun
}

func labAlgos() []labAlgo {
	var algos []labAlgo
	levels := []struct {
		name  string
		sc    StoppingCondition
		delta float64
	}{{"90", Stop90, 0.10}, {"95", Stop95, 0.05}, {"99", Stop99, 0.01}, {"999", Stop999, 0.001}}
	// Caps on a sim's length, in iterations (Macondo) or samples per play
	// (BAI); 0 is Macondo's default, 2000 + 625 a ply.
	caps := []int{0, 500, 1000, 1500, 2500}
	for _, l := range levels {
		for _, limit := range caps {
			if limit != 0 && l.name != "99" {
				continue
			}
			l, limit := l, limit
			name := "macondo-" + l.name
			if limit != 0 {
				name += fmt.Sprintf("-cap%d", limit)
			}
			algos = append(algos, labAlgo{name, func(lt *labTable, perm []int32) labRun {
				return lt.runAutostop(perm, l.sc, 0, limit)
			}})
		}
	}
	for _, n := range []int{64, 128, 256, 512, 1024, 2048, 4096, 8192} {
		n := n
		algos = append(algos, labAlgo{fmt.Sprintf("uniform-%d", n), func(lt *labTable, perm []int32) labRun {
			return lt.runAutostop(perm, StopNone, n, 0)
		}})
	}
	add := func(level string, delta float64, minimum int, utility, paired bool, limit int) {
		c := labBAIConfig{delta: delta, minimum: minimum, utility: utility, paired: paired, similar: true, limitPer: limit}
		name := fmt.Sprintf("bai-%s-min%d", level, minimum)
		if !utility {
			name += "-win"
		}
		if !paired {
			name += "-unpaired"
		}
		if limit != 0 {
			name += fmt.Sprintf("-cap%d", limit)
		}
		algos = append(algos, labAlgo{name, func(lt *labTable, perm []int32) labRun {
			c := c
			if c.limitPer == 0 {
				c.limitPer = defaultIterationsCutoff + lt.Plies*defaultPerPlyStopScaling
			}
			return lt.runBAI(perm, c)
		}})
	}
	for _, l := range levels {
		for _, minimum := range []int{50, 128, 500} {
			for _, limit := range caps {
				// Every level and minimum at the default cap; the other
				// caps at 99% only.
				if limit != 0 && l.name != "99" {
					continue
				}
				if limit != 0 && limit <= minimum {
					continue
				}
				add(l.name, l.delta, minimum, true, true, limit)
			}
		}
	}
	// What the pieces are worth, at 99% and a minimum of 128: win
	// probability alone as the reward, and no paired statistics.
	for _, limit := range caps {
		add("99", 0.01, 128, false, true, limit)
		if limit == 0 || limit == 1500 {
			add("99", 0.01, 128, true, false, limit)
		}
	}
	// Other weights on the utility's spread term, and bigger rounds (a round
	// is laid out from the statistics before it, so a round as big as a
	// many-threaded machine needs adapts less often).
	for _, limit := range []int{0, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000} {
		sweepOnly := limit != 0 && limit != 1000 && limit != 1500
		for _, w := range []float64{0.02, 0.1, 0.25} {
			if sweepOnly && w != 0.1 {
				continue
			}
			c := labBAIConfig{delta: 0.01, minimum: 128, utility: true, paired: true, similar: true, limitPer: limit, spreadWeight: w}
			name := fmt.Sprintf("bai-99-min128-w%g", w)
			if limit != 0 {
				name += fmt.Sprintf("-cap%d", limit)
			}
			algos = append(algos, labAlgo{name, func(lt *labTable, perm []int32) labRun {
				c := c
				if c.limitPer == 0 {
					c.limitPer = defaultIterationsCutoff + lt.Plies*defaultPerPlyStopScaling
				}
				return lt.runBAI(perm, c)
			}})
		}
		for _, rs := range []int{128, 512} {
			if sweepOnly {
				continue
			}
			c := labBAIConfig{delta: 0.01, minimum: 128, utility: true, paired: true, similar: true, limitPer: limit, roundSize: rs}
			name := fmt.Sprintf("bai-99-min128-round%d", rs)
			if limit != 0 {
				name += fmt.Sprintf("-cap%d", limit)
			}
			algos = append(algos, labAlgo{name, func(lt *labTable, perm []int32) labRun {
				c := c
				if c.limitPer == 0 {
					c.limitPer = defaultIterationsCutoff + lt.Plies*defaultPerPlyStopScaling
				}
				return lt.runBAI(perm, c)
			}})
		}
	}
	add("95", 0.05, 128, false, true, 0)
	add("999", 0.001, 128, false, true, 0)
	return algos
}

func TestSimStoppingLab(t *testing.T) {
	dir := os.Getenv("STOPLAB_DIR")
	if dir == "" {
		t.Skip("STOPLAB_DIR not set")
	}
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	reps := simSpeedEnvInt("STOPLAB_REPS", 20)
	workers := simSpeedEnvInt("STOPLAB_WORKERS", 16)
	ks := []int{15, 50}
	if v := os.Getenv("STOPLAB_K"); v != "" {
		ks = nil
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			ks = append(ks, n)
		}
	}
	algos := labAlgos()
	if v := os.Getenv("STOPLAB_ALGOS"); v != "" {
		var keep []labAlgo
		for _, a := range algos {
			for _, prefix := range strings.Split(v, ",") {
				if strings.HasPrefix(a.name, prefix) {
					keep = append(keep, a)
					break
				}
			}
		}
		algos = keep
	}
	files, err := filepath.Glob(filepath.Join(dir, "pos_*.tab"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no tables in %s (%v)", dir, err)
	}
	sort.Strings(files)
	out, err := os.Create(os.Getenv("STOPLAB_OUT"))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	w := bufio.NewWriter(out)
	defer w.Flush()
	fmt.Fprintln(w, "pos,bag,k,algo,rep,samples,status,pick,best_win,regret_win,regret_eq,regret_util,gap_win")
	var mu sync.Mutex
	var wg sync.WaitGroup
	jobs := make(chan int)
	for wk := 0; wk < workers; wk++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for fi := range jobs {
				lt, err := loadLabTable(files[fi])
				if err != nil {
					panic(err)
				}
				var sb strings.Builder
				for _, k := range ks {
					lt.restrict(k)
					// The truth: the best play by Macondo's own ranking
					// (win probability, then equity), by equity and by
					// utility, and the win gap to the runner-up.
					bestWin, bestEq, bestUtil := 0, 0, 0
					for i := 1; i < lt.k; i++ {
						if lt.meanWin[i] > lt.meanWin[bestWin] {
							bestWin = i
						}
						if lt.meanEq[i] > lt.meanEq[bestEq] {
							bestEq = i
						}
						if lt.meanUtil[i] > lt.meanUtil[bestUtil] {
							bestUtil = i
						}
					}
					gap := math.Inf(1)
					for i := 0; i < lt.k; i++ {
						if i != bestWin {
							gap = math.Min(gap, lt.meanWin[bestWin]-lt.meanWin[i])
						}
					}
					for rep := 0; rep < reps; rep++ {
						rng := rand.New(rand.NewSource(int64(fi)*1000003 + int64(rep)))
						// The replay's iterations: draw number i is this
						// iteration of the table. Long enough for any arm
						// to take a whole capped sim's samples.
						perm := make([]int32, max(lt.N, (defaultIterationsCutoff+lt.Plies*defaultPerPlyStopScaling)*lt.k))
						for i := range perm {
							perm[i] = int32(rng.Intn(lt.N))
						}
						for _, a := range algos {
							r := a.run(lt, perm)
							if r.pick < 0 {
								fmt.Fprintf(&sb, "%d,%d,%d,%s,%d,%d,%s,-1,%d,,,,%.5f\n", fi, lt.Bag, lt.k, a.name, rep, r.samples, r.status, bestWin, gap)
								continue
							}
							fmt.Fprintf(&sb, "%d,%d,%d,%s,%d,%d,%s,%d,%d,%.5f,%.3f,%.5f,%.5f\n", fi, lt.Bag, lt.k, a.name, rep,
								r.samples, r.status, r.pick, bestWin,
								lt.meanWin[bestWin]-lt.meanWin[r.pick],
								lt.meanEq[bestEq]-lt.meanEq[r.pick],
								lt.meanUtil[bestUtil]-lt.meanUtil[r.pick], gap)
						}
					}
				}
				mu.Lock()
				w.WriteString(sb.String())
				mu.Unlock()
			}
		}()
	}
	for fi := range files {
		jobs <- fi
	}
	close(jobs)
	wg.Wait()
}

// TestSimStoppingReal runs real sims of the tables' positions, with the
// stopping rule the sim has now and with best-arm identification, and
// judges their picks against the tables as the lab does. It checks that
// the sim does what the lab's replays say it should.
//
//	STOPREAL_DIR=dir STOPREAL_OUT=file [STOPREAL_REPS=5 STOPREAL_K=50
//	  STOPREAL_THREADS=16 STOPREAL_WORKERS=4]
func TestSimStoppingReal(t *testing.T) {
	dir := os.Getenv("STOPREAL_DIR")
	if dir == "" {
		t.Skip("STOPREAL_DIR not set")
	}
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	reps := simSpeedEnvInt("STOPREAL_REPS", 5)
	k := simSpeedEnvInt("STOPREAL_K", 50)
	threads := simSpeedEnvInt("STOPREAL_THREADS", 16)
	workers := simSpeedEnvInt("STOPREAL_WORKERS", 4)
	files, err := filepath.Glob(filepath.Join(dir, "pos_*.tab"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no tables in %s (%v)", dir, err)
	}
	sort.Strings(files)
	out, err := os.Create(os.Getenv("STOPREAL_OUT"))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	w := bufio.NewWriter(out)
	defer w.Flush()
	fmt.Fprintln(w, "pos,bag,k,algo,rep,samples,status,pick,best_win,regret_win,regret_eq,regret_util,gap_win")
	type algo struct {
		name  string
		setup func(s *Simmer)
	}
	algos := []algo{
		{"real-macondo-99", func(s *Simmer) {}},
		{"real-bai-99", func(s *Simmer) { s.SetBAI(true) }},
		{"real-bai-99-cap1500", func(s *Simmer) { s.SetBAI(true); s.SetBAICapPerPlay(1500) }},
		{"real-bai-99-cap1000", func(s *Simmer) { s.SetBAI(true); s.SetBAICapPerPlay(1000) }},
	}
	var mu sync.Mutex
	var wg sync.WaitGroup
	jobs := make(chan int)
	for wk := 0; wk < workers; wk++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for fi := range jobs {
				lt, err := loadLabTable(files[fi])
				if err != nil {
					panic(err)
				}
				lt.restrict(k)
				bestWin, bestEq, bestUtil := 0, 0, 0
				for i := 1; i < lt.k; i++ {
					if lt.meanWin[i] > lt.meanWin[bestWin] {
						bestWin = i
					}
					if lt.meanEq[i] > lt.meanEq[bestEq] {
						bestEq = i
					}
					if lt.meanUtil[i] > lt.meanUtil[bestUtil] {
						bestUtil = i
					}
				}
				var sb strings.Builder
				for rep := 0; rep < reps; rep++ {
					for _, a := range algos {
						parsed, err := cgp.ParseCGP(DefaultConfig, lt.CGP)
						if err != nil {
							panic(err)
						}
						parsed.RecalculateBoard()
						g := parsed.Game
						var seed [32]byte
						seed[0], seed[1], seed[2], seed[3] = byte(fi), byte(fi>>8), byte(rep), 91
						g.SeedBag(seed)
						calcs, leaves := defaultSimCalculators(lt.Lexicon)
						s := &Simmer{}
						s.Init(g, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
						s.SetThreads(threads)
						s.TryLoadWMP(DefaultConfig.WGLConfig(), lt.Lexicon)
						s.SetStoppingCondition(Stop99)
						a.setup(s)
						if err := s.PrepareSim(lt.Plies, lt.moves[:lt.k]); err != nil {
							panic(err)
						}
						plays := append([]*SimmedPlay(nil), s.simmedPlays.plays...)
						if err := s.Simulate(context.Background()); err != nil {
							panic(err)
						}
						best := s.WinningPlay()
						pick, samples := -1, 0
						for i, p := range plays {
							samples += p.winPctStats.Iterations()
							if p == best {
								pick = i
							}
						}
						status := "stop"
						if s.usesBAI() {
							status = s.BAIStatus()
						} else if s.Iterations() > s.autostopper.iterationsCutoff+lt.Plies*s.autostopper.perPlyStopScaling {
							status = "limit"
						}
						fmt.Fprintf(&sb, "%d,%d,%d,%s,%d,%d,%s,%d,%d,%.5f,%.3f,%.5f,0\n", fi, lt.Bag, lt.k, a.name, rep,
							samples, status, pick, bestWin,
							lt.meanWin[bestWin]-lt.meanWin[pick],
							lt.meanEq[bestEq]-lt.meanEq[pick],
							lt.meanUtil[bestUtil]-lt.meanUtil[pick])
					}
				}
				mu.Lock()
				w.WriteString(sb.String())
				mu.Unlock()
			}
		}()
	}
	for fi := range files {
		jobs <- fi
	}
	close(jobs)
	wg.Wait()
}
