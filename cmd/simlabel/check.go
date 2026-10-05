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
	"sort"
	"strconv"
	"strings"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/turnplayer"
)

// checkMain compares the net's ranking of each labelled position's
// candidates with the sim's. It replays the games from the turn log (the
// net needs the move history), scores every candidate with the served net
// (Triton, as FastMlBot does) and reports how often the net's pick is the
// sim's, and how much sim win probability and equity the net's pick gives
// up against the sim's best.
//
//	MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
//	MACONDO_TRITON_MODEL_NAME=macondo-nn-tf-streamopen MACONDO_TRITON_MODEL_VERSION=1 \
//	simlabel check -turns open2.txt.gz -tag open2 -positions positions-val.jsonl -labels labels-val.jsonl
func checkMain(args []string) {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	var turnsPath, tag, posPath, labPath, lexicon, outPath string
	fs.StringVar(&turnsPath, "turns", "", "the turn log the positions came from")
	fs.StringVar(&tag, "tag", "", "the tag used by select")
	fs.StringVar(&posPath, "positions", "", "positions file")
	fs.StringVar(&labPath, "labels", "", "labels file")
	fs.StringVar(&lexicon, "lexicon", "NWL23", "lexicon")
	fs.StringVar(&outPath, "out", "", "optional: per-position JSON lines with the net's values")
	fs.Parse(args)
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	cfg := config.DefaultConfig()

	labels := map[string]*Label{}
	readJSONL(labPath, func(b []byte) {
		var l Label
		if json.Unmarshal(b, &l) == nil && len(l.Cands) > 0 {
			labels[l.Key] = &l
		}
	})
	target := map[string]int{} // game -> turn
	readJSONL(posPath, func(b []byte) {
		var p Position
		if json.Unmarshal(b, &p) == nil && labels[p.Key] != nil {
			target[p.Game] = p.Turn
		}
	})
	fmt.Fprintf(os.Stderr, "%d labelled positions to check\n", len(target))
	leaves, err := equity.NewExhaustiveLeaveCalculator(lexicon, cfg, "")
	if err != nil {
		log.Fatal().Err(err).Msg("leaves")
	}
	var outW *bufio.Writer
	if outPath != "" {
		f, err := os.Create(outPath)
		if err != nil {
			log.Fatal().Err(err).Msg("out")
		}
		defer f.Close()
		outW = bufio.NewWriter(f)
		defer outW.Flush()
	}

	in, err := openMaybeGzip(turnsPath)
	if err != nil {
		log.Fatal().Err(err).Msg("turn log")
	}
	defer in.Close()
	cr := csv.NewReader(bufio.NewReaderSize(in, 1<<20))
	cr.FieldsPerRecord = -1
	cr.TrimLeadingSpace = true
	hdr, _ := cr.Read()
	col := map[string]int{}
	for i, h := range hdr {
		col[h] = i
	}
	type replay struct {
		tp   *turnplayer.BaseTurnPlayer
		hist []*move.Move
		done bool
	}
	live := map[string]*replay{}
	var n, same int
	var winRegret, eqRegret, rankCorr float64
	var simTopNetRank []int
	remaining := len(target)
	for remaining > 0 {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal().Err(err).Msg("turn row")
		}
		if rec[0] == "playerID" {
			continue
		}
		g := rec[col["gameID"]]
		tt, ok := target[g]
		if !ok {
			continue
		}
		rp := live[g]
		if rp == nil {
			other := "p2"
			if rec[0] == "p2" {
				other = "p1"
			}
			tp, err := newSelGame(cfg, lexicon, rec[0], other)
			if err != nil {
				log.Fatal().Err(err).Msg("game")
			}
			rp = &replay{tp: tp}
			live[g] = rp
		}
		if rp.done {
			continue
		}
		gm := rp.tp.Game
		onturn := gm.PlayerOnTurn()
		turn, _ := strconv.Atoi(rec[col["turn"]])
		if err := gm.SetRackFor(onturn, tilemapping.RackFromString(strings.TrimSpace(rec[col["rack"]]), gm.Alphabet())); err != nil {
			log.Fatal().Err(err).Msg("rack")
		}
		if turn == tt {
			l := labels[fmt.Sprintf("%s:%s:%d", tag, g, turn)]
			moves := make([]*move.Move, 0, len(l.Cands))
			idx := make([]int, 0, len(l.Cands))
			for i, c := range l.Cands {
				m, err := rp.tp.ParseMove(onturn, false, strings.Fields(stats.Normalize(c.Move)), false)
				if err != nil {
					log.Warn().Err(err).Str("move", c.Move).Msg("unparsable candidate; skipped")
					continue
				}
				moves = append(moves, m)
				idx = append(idx, i)
			}
			resp, err := gm.MLEvaluateMoves(moves, leaves, rp.hist)
			if err != nil {
				log.Fatal().Err(err).Msg("net (is Triton up and MACONDO_TRITON_* set?)")
			}
			// The net's pick: highest value (ties: as FastMlBot, more tiles).
			best := 0
			for i := range moves {
				if resp.Value[i] > resp.Value[best] || (resp.Value[i] == resp.Value[best] && moves[i].TilesPlayed() > moves[best].TilesPlayed()) {
					best = i
				}
			}
			simBest := l.Cands[0] // sorted by sim win probability
			pick := l.Cands[idx[best]]
			n++
			if idx[best] == 0 {
				same++
			}
			winRegret += simBest.Win - pick.Win
			bestEq := simBest.Eq
			for _, c := range l.Cands {
				if c.Win >= simBest.Win-1e-9 && c.Eq > bestEq {
					bestEq = c.Eq
				}
			}
			eqRegret += bestEq - pick.Eq
			// where the sim's best sits in the net's order
			if j := sort.SearchInts(idx, 0); j < len(idx) && idx[j] == 0 {
				r := 1
				for i := range moves {
					if resp.Value[i] > resp.Value[j] {
						r++
					}
				}
				simTopNetRank = append(simTopNetRank, r)
			}
			rankCorr += spearman(l, idx, resp.Value)
			if outW != nil {
				b, _ := json.Marshal(map[string]any{"key": l.Key, "net_value": resp.Value, "cand_index": idx})
				outW.Write(append(b, '\n'))
			}
			rp.done = true
			remaining--
		}
		m, err := rp.tp.ParseMove(onturn, false, strings.Fields(stats.Normalize(rec[col["play"]])), false)
		if err != nil {
			log.Fatal().Err(err).Msg("parse")
		}
		if err := gm.PlayMove(m, false, 0); err != nil {
			log.Fatal().Err(err).Msg("play")
		}
		rp.hist = append(rp.hist, m)
	}
	sort.Ints(simTopNetRank)
	fmt.Printf("positions %d\n", n)
	fmt.Printf("net's pick = sim's pick     %.1f%%\n", 100*float64(same)/float64(n))
	fmt.Printf("sim win prob given up       %.2f points per position (sim's best minus the net's pick)\n", 100*winRegret/float64(n))
	fmt.Printf("sim equity given up         %.2f points per position\n", eqRegret/float64(n))
	if k := len(simTopNetRank); k > 0 {
		fmt.Printf("rank of the sim's best in the net's order: median %d, 90th pct %d\n", simTopNetRank[k/2], simTopNetRank[k*9/10])
	}
	fmt.Printf("Spearman, net vs sim win    %.3f mean over positions\n", rankCorr/float64(n))
}

func readJSONL(path string, f func([]byte)) {
	in, err := openMaybeGzip(path)
	if err != nil {
		log.Fatal().Err(err).Str("path", path).Msg("open")
	}
	defer in.Close()
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		f(sc.Bytes())
	}
}

// spearman: rank correlation between the sim's win probabilities and the
// net's values over the candidates both scored.
func spearman(l *Label, idx []int, val []float32) float64 {
	n := len(idx)
	if n < 3 {
		return 0
	}
	a := make([]float64, n)
	b := make([]float64, n)
	for i := range idx {
		a[i] = l.Cands[idx[i]].Win
		b[i] = float64(val[i])
	}
	ra, rb := ranks(a), ranks(b)
	var ma, mb float64
	for i := 0; i < n; i++ {
		ma += ra[i]
		mb += rb[i]
	}
	ma /= float64(n)
	mb /= float64(n)
	var num, da, db float64
	for i := 0; i < n; i++ {
		num += (ra[i] - ma) * (rb[i] - mb)
		da += (ra[i] - ma) * (ra[i] - ma)
		db += (rb[i] - mb) * (rb[i] - mb)
	}
	if da == 0 || db == 0 {
		return 0
	}
	return num / math.Sqrt(da*db)
}

func ranks(x []float64) []float64 {
	idx := make([]int, len(x))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool { return x[idx[i]] < x[idx[j]] })
	r := make([]float64, len(x))
	for i := 0; i < len(idx); {
		j := i
		for j+1 < len(idx) && x[idx[j+1]] == x[idx[i]] {
			j++
		}
		for k := i; k <= j; k++ {
			r[idx[k]] = float64(i+j) / 2
		}
		i = j + 1
	}
	return r
}
