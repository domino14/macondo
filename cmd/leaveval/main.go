// leaveval prints the engine's static leave value for each leave read from
// stdin, one per line -- a way to ask, over a corpus, whether the leaves the
// imputation misjudges have anything in common the leave table already knows.
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/rs/zerolog"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	lex := "NWL23"
	if len(os.Args) > 1 {
		lex = os.Args[1]
	}
	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigDefaultLexicon, lex)
	calc, err := equity.NewCombinedStaticCalculator(lex, cfg, "", equity.PEGAdjustmentFilename)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	ld, err := tilemapping.GetDistribution(cfg.WGLConfig(), "english")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		s := strings.TrimSpace(sc.Text())
		if s == "" {
			continue
		}
		mw, err := tilemapping.ToMachineWord(s, ld.TileMapping())
		if err != nil {
			fmt.Printf("%s\tNaN\n", s)
			continue
		}
		fmt.Printf("%s\t%.3f\n", s, calc.LeaveValue(mw))
	}
}
