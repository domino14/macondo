package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/playability"
)

// perGame turns per-turn sums into a per-game value and its standard error.
func perGame(sum, sum2 float64, n, tpg float64) (float64, float64) {
	if n == 0 {
		return 0, 0
	}
	m := sum / n
	v := math.Max(0, sum2/n-m*m)
	return m * tpg, math.Sqrt(v/n) * tpg
}

func runScore(args []string) error {
	fs := flag.NewFlagSet("score", flag.ExitOnError)
	genDir := fs.String("gen", "", "gen shard directory")
	synthDir := fs.String("synth", "", "synth chunk directory (optional)")
	out := fs.String("out", "", "output directory")
	lexicon := fs.String("lexicon", "NWL23", "lexicon")
	ld := fs.String("ld", "english", "letter distribution")
	fs.Parse(args)
	if *genDir == "" || *out == "" {
		return fmt.Errorf("-gen and -out are required")
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		return err
	}
	cfg := newConfig(*lexicon, *ld)

	files, _ := filepath.Glob(filepath.Join(*genDir, "shard-*.gob.gz"))
	sort.Strings(files)
	tot := playability.NewShard(-1)
	for _, f := range files {
		var sh playability.Shard
		if err := playability.ReadGob(f, &sh); err != nil {
			return fmt.Errorf("%s: %w", f, err)
		}
		sh.Positions = nil
		tot.Merge(&sh)
	}
	if tot.Games == 0 {
		return fmt.Errorf("no games in %s", *genDir)
	}
	G, T := float64(tot.Games), float64(tot.Turns)
	tpg := T / G
	log.Info().Int("shards", len(files)).Float64("games", G).Float64("turns/game", tpg).Msg("gen")

	synth := map[playability.Key]*SynthResult{}
	if *synthDir != "" {
		sf, _ := filepath.Glob(filepath.Join(*synthDir, "synth-*.gob.gz"))
		for _, f := range sf {
			var ch SynthChunk
			if err := playability.ReadGob(f, &ch); err != nil {
				return fmt.Errorf("%s: %w", f, err)
			}
			for a, r := range ch.Results {
				synth[a] = r
			}
		}
		log.Info().Int("alphagrams", len(synth)).Msg("synth")
	}

	classes, tm, err := lexiconClasses(cfg, *lexicon)
	if err != nil {
		return err
	}
	dist, err := tilemapping.GetDistribution(cfg.WGLConfig(), *ld)
	if err != nil {
		return err
	}
	d := dist.Distribution()
	nl := int(tm.NumLetters())
	isWord := map[playability.Key]bool{}
	for _, ws := range classes {
		for _, w := range ws {
			isWord[w] = true
		}
	}
	hooks := func(w playability.Key) (int, int) {
		f, b := 0, 0
		for ml := 1; ml < nl; ml++ {
			c := string([]byte{byte(ml)})
			if isWord[playability.Key(c)+w] {
				f++
			}
			if isWord[w+playability.Key(c)] {
				b++
			}
		}
		return f, b
	}
	vis := func(k playability.Key) string { return playability.Visible(k, tm) }

	var alphas []playability.Key
	for a := range classes {
		alphas = append(alphas, a)
	}
	sort.Slice(alphas, func(i, j int) bool { return alphas[i] < alphas[j] })

	wf, err := os.Create(filepath.Join(*out, "words.tsv"))
	if err != nil {
		return err
	}
	defer wf.Close()
	af, err := os.Create(filepath.Join(*out, "alphas.tsv"))
	if err != nil {
		return err
	}
	defer af.Close()
	ww, aw := bufio.NewWriter(wf), bufio.NewWriter(af)
	defer ww.Flush()
	defer aw.Flush()
	fmt.Fprintln(ww, "word\talphagram\tlen\tcombos\tanagrams\tfhooks\tbhooks\tn\tnmain\ta_play\ta_se\ta_main\ta_ph0\ta_ph1\ta_ph2\ta_ph3\tb_k\tb_play\tb_se\tb_hits")
	fmt.Fprintln(aw, "alphagram\tlen\tcombos\twords\tn\ta_play\ta_se\ta_ph0\ta_ph1\ta_ph2\ta_ph3\tb_k\tb_pmean\tb_play\tb_se\tb_hits")
	for _, a := range alphas {
		ws := classes[a]
		sort.Slice(ws, func(i, j int) bool { return ws[i] < ws[j] })
		combos := playability.Combos(d, []byte(a))
		sr := synth[a]
		for _, w := range ws {
			st := tot.Words[w]
			if st == nil {
				st = &playability.Stat{}
			}
			ap, ase := perGame(st.Sum, st.Sum2, T, tpg)
			fh, bh := hooks(w)
			bk, bp, bse, bh2 := "", "", "", ""
			if sr != nil {
				p, se, hits := sr.PerTurn(w)
				p, se = p*tpg, se*tpg
				bk, bp, bse, bh2 = fmt.Sprint(sr.K()), fmt.Sprintf("%.6g", p), fmt.Sprintf("%.3g", se), fmt.Sprint(hits)
			}
			fmt.Fprintf(ww, "%s\t%s\t%d\t%.6g\t%d\t%d\t%d\t%d\t%d\t%.6g\t%.3g\t%.6g\t%.4g\t%.4g\t%.4g\t%.4g\t%s\t%s\t%s\t%s\n",
				vis(w), vis(a), len(w), combos, len(ws), fh, bh, st.N, st.NMain, ap, ase, st.SumMain/G,
				st.PhaseSum[0]/G, st.PhaseSum[1]/G, st.PhaseSum[2]/G, st.PhaseSum[3]/G,
				bk, bp, bse, bh2)
		}
		st := tot.Alphas[a]
		if st == nil {
			st = &playability.Stat{}
		}
		ap, ase := perGame(st.Sum, st.Sum2, T, tpg)
		var names []string
		for _, w := range ws {
			names = append(names, vis(w))
		}
		bk, bpm, bp, bse, bh := "", "", "", "", ""
		if sr != nil {
			p, se, hits := sr.PerTurn("")
			p, se = p*tpg, se*tpg
			bk, bpm = fmt.Sprint(sr.K()), fmt.Sprintf("%.4g", sr.PMean())
			bp, bse, bh = fmt.Sprintf("%.6g", p), fmt.Sprintf("%.3g", se), fmt.Sprint(hits)
		}
		fmt.Fprintf(aw, "%s\t%d\t%.6g\t%s\t%d\t%.6g\t%.3g\t%.4g\t%.4g\t%.4g\t%.4g\t%s\t%s\t%s\t%s\t%s\n",
			vis(a), len(a), combos, strings.Join(names, ","), st.N, ap, ase,
			st.PhaseSum[0]/G, st.PhaseSum[1]/G, st.PhaseSum[2]/G, st.PhaseSum[3]/G,
			bk, bpm, bp, bse, bh)
	}
	meta := fmt.Sprintf("games\t%d\nturns\t%d\nturns_per_game\t%.4f\nphase_turns\t%v\n", tot.Games, tot.Turns, tpg, tot.PhaseN)
	return os.WriteFile(filepath.Join(*out, "meta.tsv"), []byte(meta), 0o644)
}
