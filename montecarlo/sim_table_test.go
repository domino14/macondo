package montecarlo

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/rs/zerolog"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/move"
)

// A rollout table is every candidate play's result in each of N iterations
// of a sim that never stops: the raw material for comparing stopping rules,
// which can then be replayed over the same rollouts any number of times
// against a known best play (see sim_stopping_lab_test.go).
//
//	SIMTAB=gen SIMTAB_DIR=dir [SIMTAB_LEX=CSW21 SIMTAB_POSITIONS=200
//	  SIMTAB_K=50 SIMTAB_N=20000 SIMTAB_PLIES=5 SIMTAB_THREADS=16
//	  SIMTAB_FIRST=0]
//
// writes dir/pos_NNN.tab for each position: a JSON header line, then the
// win probabilities and the equities, K rows of N float32 each.

type simTableHeader struct {
	CGP           string    `json:"cgp"`
	Lexicon       string    `json:"lexicon"`
	Bag           int       `json:"bag"`
	Plies         int       `json:"plies"`
	K             int       `json:"k"`
	N             int       `json:"n"`
	InitialSpread int       `json:"initial_spread"`
	Plays         []string  `json:"plays"`
	StaticEquity  []float64 `json:"static_equity"`
}

type simTable struct {
	simTableHeader
	win, eq [][]float32
	moves   []*move.Move
}

func TestSimTableGen(t *testing.T) {
	if os.Getenv("SIMTAB") != "gen" {
		t.Skip("SIMTAB not set")
	}
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	dir := os.Getenv("SIMTAB_DIR")
	lexicon := os.Getenv("SIMTAB_LEX")
	if lexicon == "" {
		lexicon = "CSW21"
	}
	numPos := simSpeedEnvInt("SIMTAB_POSITIONS", 200)
	k := simSpeedEnvInt("SIMTAB_K", 50)
	n := simSpeedEnvInt("SIMTAB_N", 20000)
	plies := simSpeedEnvInt("SIMTAB_PLIES", 5)
	threads := simSpeedEnvInt("SIMTAB_THREADS", 16)
	first := simSpeedEnvInt("SIMTAB_FIRST", 0)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Bag sizes spread evenly from the opening to the last full bag.
	stops := make([]int, numPos)
	for i := range stops {
		stops[i] = 86 - (i*79)/max(numPos-1, 1)
	}
	positions := simSpeedPositions(t, lexicon, stops)
	for pi, pos := range positions {
		if pi < first {
			continue
		}
		parsed, err := cgp.ParseCGP(DefaultConfig, pos)
		if err != nil {
			t.Fatal(err)
		}
		parsed.RecalculateBoard()
		g := parsed.Game
		calcs, leaves := defaultSimCalculators(lexicon)
		p, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(g, DefaultConfig, calcs)
		if err != nil {
			t.Fatal(err)
		}
		plays := p.GenerateMoves(k)
		s := &Simmer{}
		s.Init(g, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
		s.SetThreads(threads)
		s.TryLoadWMP(DefaultConfig.WGLConfig(), lexicon)
		if err := s.PrepareSim(plies, plays); err != nil {
			t.Fatal(err)
		}
		tab := &simTable{simTableHeader: simTableHeader{
			CGP: pos, Lexicon: lexicon, Bag: g.Bag().TilesRemaining(), Plies: plies,
			K: len(plays), N: n, InitialSpread: s.initialSpread,
		}}
		for _, m := range plays {
			tab.Plays = append(tab.Plays, m.ShortDescription())
			tab.StaticEquity = append(tab.StaticEquity, m.Equity())
		}
		tab.win = make([][]float32, len(plays))
		tab.eq = make([][]float32, len(plays))
		for i := range plays {
			tab.win[i] = make([]float32, n)
			tab.eq[i] = make([]float32, n)
			for j := range tab.win[i] {
				tab.win[i][j] = float32(math.NaN())
			}
		}
		// Each iteration number is used once, so the cells are written by
		// one thread each.
		s.rolloutHook = func(iteration uint64, playIdx int, win, eq float64) {
			if iteration < uint64(n) {
				tab.win[playIdx][iteration] = float32(win)
				tab.eq[playIdx][iteration] = float32(eq)
			}
		}
		start := time.Now()
		if err := s.simulate(context.Background(), simRun{maxIters: n, quiet: true}); err != nil {
			t.Fatal(err)
		}
		for i := range tab.win {
			for j, v := range tab.win[i] {
				if math.IsNaN(float64(v)) {
					t.Fatalf("position %d: play %d iteration %d was never rolled out", pi, i, j)
				}
			}
		}
		if err := tab.write(filepath.Join(dir, fmt.Sprintf("pos_%03d.tab", pi))); err != nil {
			t.Fatal(err)
		}
		fmt.Printf("SIMTAB pos=%d bag=%d plays=%d iters=%d secs=%.1f\n", pi, tab.Bag, tab.K, n, time.Since(start).Seconds())
	}
}

func (tab *simTable) write(path string) error {
	f, err := os.Create(path + ".tmp")
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	hdr, err := json.Marshal(tab.simTableHeader)
	if err != nil {
		return err
	}
	w.Write(hdr)
	w.WriteByte('\n')
	for _, rows := range [][][]float32{tab.win, tab.eq} {
		for _, row := range rows {
			if err := binary.Write(w, binary.LittleEndian, row); err != nil {
				return err
			}
		}
	}
	if err := w.Flush(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

func readSimTable(path string) (*simTable, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	line, err := r.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	tab := &simTable{}
	if err := json.Unmarshal(line, &tab.simTableHeader); err != nil {
		return nil, err
	}
	read := func() ([][]float32, error) {
		rows := make([][]float32, tab.K)
		for i := range rows {
			rows[i] = make([]float32, tab.N)
			if err := binary.Read(r, binary.LittleEndian, rows[i]); err != nil {
				return nil, err
			}
		}
		return rows, nil
	}
	if tab.win, err = read(); err != nil {
		return nil, err
	}
	if tab.eq, err = read(); err != nil {
		return nil, err
	}
	return tab, nil
}
