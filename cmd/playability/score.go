package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
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
		tot.Merge(&sh)
	}
	if tot.Games == 0 {
		return fmt.Errorf("no games in %s", *genDir)
	}
	G, T := float64(tot.Games), float64(tot.Turns)
	tpg := T / G
	log.Info().Int("shards", len(files)).Float64("games", G).Float64("turns/game", tpg).Msg("gen")

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
	fmt.Fprintln(ww, "word\talphagram\tlen\tcombos\tanagrams\tfhooks\tbhooks\tn\tnmain\ta_play\ta_se\ta_main\ta_ph0\ta_ph1\ta_ph2\ta_ph3")
	fmt.Fprintln(aw, "alphagram\tlen\tcombos\twords\tn\ta_play\ta_se\ta_ph0\ta_ph1\ta_ph2\ta_ph3")
	for _, a := range alphas {
		ws := classes[a]
		sort.Slice(ws, func(i, j int) bool { return ws[i] < ws[j] })
		combos := playability.Combos(d, []byte(a))
		for _, w := range ws {
			st := tot.Words[w]
			if st == nil {
				st = &playability.Stat{}
			}
			ap, ase := perGame(st.Sum, st.Sum2, T, tpg)
			fh, bh := hooks(w)
			fmt.Fprintf(ww, "%s\t%s\t%d\t%.6g\t%d\t%d\t%d\t%d\t%d\t%.6g\t%.3g\t%.6g\t%.4g\t%.4g\t%.4g\t%.4g\n",
				vis(w), vis(a), len(w), combos, len(ws), fh, bh, st.N, st.NMain, ap, ase, st.SumMain/G,
				st.PhaseSum[0]/G, st.PhaseSum[1]/G, st.PhaseSum[2]/G, st.PhaseSum[3]/G)
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
		fmt.Fprintf(aw, "%s\t%d\t%.6g\t%s\t%d\t%.6g\t%.3g\t%.4g\t%.4g\t%.4g\t%.4g\n",
			vis(a), len(a), combos, strings.Join(names, ","), st.N, ap, ase,
			st.PhaseSum[0]/G, st.PhaseSum[1]/G, st.PhaseSum[2]/G, st.PhaseSum[3]/G)
	}
	meta := fmt.Sprintf("games\t%d\nturns\t%d\nturns_per_game\t%.4f\nphase_turns\t%v\n", tot.Games, tot.Turns, tpg, tot.PhaseN)
	return os.WriteFile(filepath.Join(*out, "meta.tsv"), []byte(meta), 0o644)
}

// lexiconClasses returns the lexicon's words grouped by alphagram.
func lexiconClasses(cfg *config.Config, lexicon string) (map[playability.Key][]playability.Key, *tilemapping.TileMapping, error) {
	k, err := kwg.GetKWG(cfg.WGLConfig(), lexicon)
	if err != nil {
		return nil, nil, err
	}
	classes := map[playability.Key][]playability.Key{}
	for _, mw := range lexiconWords(k, 15) {
		w := playability.KeyOf(mw)
		a := playability.AlphaKey(w)
		classes[a] = append(classes[a], w)
	}
	return classes, k.GetAlphabet(), nil
}

// lexiconWords lists every word in the KWG's DAWG of at most maxLength tiles.
func lexiconWords(k *kwg.KWG, maxLength int) []tilemapping.MachineWord {
	var words []tilemapping.MachineWord
	root := k.ArcIndex(0)
	if root == 0 {
		return words
	}
	var walk func(node uint32, prefix []tilemapping.MachineLetter)
	walk = func(node uint32, prefix []tilemapping.MachineLetter) {
		for i := node; ; i++ {
			next := append(prefix, tilemapping.MachineLetter(k.Tile(i)))
			if k.Accepts(i) {
				words = append(words, slices.Clone(tilemapping.MachineWord(next)))
			}
			if len(next) < maxLength {
				if arc := k.ArcIndex(i); arc != 0 {
					walk(arc, next)
				}
			}
			if k.IsEnd(i) {
				break
			}
		}
	}
	walk(root, make([]tilemapping.MachineLetter, 0, maxLength))
	return words
}
