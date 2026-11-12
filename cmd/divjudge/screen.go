package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
)

// decision is one line of screendecide output.
type decision struct {
	Key    string  `json:"key"`
	Row    int     `json:"row"`
	Play   string  `json:"play"`
	Score  int     `json:"score"`
	Tiles  int     `json:"tiles"`
	Leave  string  `json:"leave"`
	Equity float64 `json:"equity"`
	Same   bool    `json:"same"`
	Err    string  `json:"err"`
}

func readDecisions(path string) (map[string]decision, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string]decision{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var d decision
		if err := json.Unmarshal(sc.Bytes(), &d); err != nil {
			return nil, err
		}
		if d.Err == "" {
			out[d.Key] = d
		}
	}
	return out, sc.Err()
}

// withMove returns a copy of a logged row with its move replaced.
func withMove(row []string, d decision) []string {
	r := slices.Clone(row)
	r[4] = d.Play
	r[5] = strconv.Itoa(d.Score)
	r[7] = strconv.Itoa(d.Tiles)
	r[8] = d.Leave
	r[9] = strconv.FormatFloat(d.Equity, 'f', 3, 64)
	return r
}

// screenRecord is a judged position: A is the variant, B the control.
type screenRecord struct {
	record
	Key string `json:"key"`
}

func screen(basePath, decidePath, controlPath, out string, iters, parallel, limit int) {
	base, err := readTurns(basePath)
	if err != nil {
		log.Fatal().Err(err).Msg("b")
	}
	vars, err := readDecisions(decidePath)
	if err != nil {
		log.Fatal().Err(err).Msg("decide")
	}
	var ctl map[string]decision
	if controlPath != "" {
		if ctl, err = readDecisions(controlPath); err != nil {
			log.Fatal().Err(err).Msg("control")
		}
	}
	type job struct {
		key    string
		ha, hb *half
		i      int
	}
	var jobs []job
	compared := 0
	for key, v := range vars {
		parts := strings.Split(key, "|")
		h, ok := base[parts[0]+"|"+parts[1]]
		if !ok || v.Row >= len(h.rows) {
			continue
		}
		logged := h.rows[v.Row]
		bRow := logged
		if ctl != nil {
			c, ok := ctl[key]
			if !ok {
				continue
			}
			bRow = withMove(logged, c)
		}
		compared++
		if strings.TrimSpace(v.Play) == strings.TrimSpace(bRow[4]) {
			continue
		}
		ha := &half{gid: h.gid, first: h.first, rows: slices.Clone(h.rows)}
		hb := &half{gid: h.gid, first: h.first, rows: slices.Clone(h.rows)}
		ha.rows[v.Row] = withMove(logged, v)
		hb.rows[v.Row] = bRow
		jobs = append(jobs, job{key, ha, hb, v.Row})
	}
	slices.SortFunc(jobs, func(a, b job) int { return strings.Compare(a.key, b.key) })
	fmt.Fprintf(os.Stderr, "%d positions compared, %d differ\n", compared, len(jobs))
	if limit > 0 && len(jobs) > limit {
		jobs = jobs[:limit]
	}
	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, "NWL23")
	cfg.Set(config.ConfigDefaultLetterDistribution, "english")
	rules, err := game.NewBasicGameRules(cfg, "NWL23", board.CrosswordGameLayout,
		"english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		log.Fatal().Err(err).Msg("rules")
	}
	f, err := os.Create(out)
	if err != nil {
		log.Fatal().Err(err).Msg("out")
	}
	defer f.Close()
	var mu sync.Mutex
	enc := json.NewEncoder(f)
	ch := make(chan job)
	var wg sync.WaitGroup
	var done atomic.Int64
	for w := 0; w < parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range ch {
				rec := screenRecord{judge(cfg, rules, j.ha, j.hb, j.i, iters), j.key}
				mu.Lock()
				enc.Encode(rec)
				mu.Unlock()
				if d := done.Add(1); d%50 == 0 {
					fmt.Fprintf(os.Stderr, "%d/%d\n", d, len(jobs))
				}
			}
		}()
	}
	for _, j := range jobs {
		ch <- j
	}
	close(ch)
	wg.Wait()
}
