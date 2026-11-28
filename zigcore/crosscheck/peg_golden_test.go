package crosscheck

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/preendgame"
)

var pegGoldens = flag.String("crosscheck.peggoldens", "",
	"write the pre-endgame golden files into this directory (zigcore/src/preendgame/testdata)")

// pegCase is a pre-endgame solved once by the Go engine to make a golden
// file the Zig engine's tests check.
type pegCase struct {
	name string
	cgp  string
	// The solver's options; the rest are its defaults. Every case runs on
	// one thread.
	plies        int
	id           bool
	maxTilesLeft int
	nested       int
	tiebreak     bool
	cutoff       bool
	only         []string // moves in UCGI notation
	oppRack      string
	// One ordering of the bag only (draw order), solved as Go's
	// eventuality mode solves it.
	eventuality string
}

const (
	pegStraightforward = "15/3Q7U3/3U2TAURINE2/1CHANSONS2W3/2AI6JO3/DIRL1PO3IN3/E1D2EF3V4/F1I2p1TRAIK3/O1L2T4E4/ABy1PIT2BRIG2/ME1MOZELLE5/1GRADE1O1NOH3/WE3R1V7/AT5E7/G6D7 ENOSTXY/ACEISUY 356/378 0 lex NWL20;"
	pegPassFRA20       = "11ONZE/10J2O1/8A1E1DO1/7QUETEE1H/10E1F1U/8ECUMERA/8C1R1TIR/7WOKS2ET/6DUR6/5G2N1M4/4HALLALiS3/1G1P1P1OM1XI3/VIVONS1BETEL3/IF1N3AS1RYAL1/ETUDIAIS7 AEINRST/ 301/300 0 lex FRA20; ld french;"
	pegTwoInBag        = "1T13/1W3Q9/VERB1U9/1E1OPIUM5C1/1LAWIN1I5O1/1Y3A1E5R1/7V4NO1/NOTArIZE1C2UN1/6ODAH2LA1/3TAHA2I2LED/2JUT4R2A1O/3G5P4D/3R3BrIEFING/3I5L4E/3K2DESYNES1M AEFGSTX/EEIOOST 370/341 0 lex CSW19;"
	// The manual's positions (docs/manual/preendgame.md): Josh Sokol and
	// Noah Walton, and after 2L P(O)ND drawing III, Noah to move; Matthew
	// Tunnicliffe and Jim Brennan.
	pegManual1     = "12D2/1U10O2/1p10L2/1R1C3KANJIS2/1I1O3A2U4/1G1T3I2I4/1H1E3Z2C1LOO/1TED3E1BYWORD/2Q4N3AXE1/1RuBIGOS3I3/F1A5WEAVE2/O1T8E3/V1E5LOURY2/ENSNARL2HM4/A6TEMP4 DEFNNPT/ 394/365 0 lex NWL20;"
	pegManualNoah  = "12D2/1U9POND/1p10L2/1R1C3KANJIS2/1I1O3A2U4/1G1T3I2I4/1H1E3Z2C1LOO/1TED3E1BYWORD/2Q4N3AXE1/1RuBIGOS3I3/F1A5WEAVE2/O1T8E3/V1E5LOURY2/ENSNARL2HM4/A6TEMP4 AEGRSTU/ 365/408 0 lex NWL20;"
	pegManual2     = "BEDEL10/R1R9U2/O1IT1Q5OM2/W1BIDI4YUM2/N2XI5AT3/E3G4T1R3/S1VOILE2OKA3/T3T1DISPACED1/9AWE1O1/9Z1s1FA/14R/13GO/13AH/3JUVIE4UTA/INRO3FLENCHES ?ANNOPY/AEGILNS 344/368 0 lex CSW21;"
	pegPolish          = "13P1/12SYĆ/11HET1/11IMA1/3C6T1EN1/2PiLŻE2WOŹNI1/1K1C5SI1Y2/DASHI2ZATNĘ3/1R1A4BAO4/1M1C1Z1ROŃ5/3ZŁU1A7/3Y1PÓKI1WERS1/GOJ2Y1IWIE4/1BOA3JĄŁ5/KANADZIE7 GLLMNOŚ/?DEFRUZ 307/288 0 lex OSPS49; ld polish;"
)

var pegCases = []pegCase{
	{name: "straightforward", cgp: pegStraightforward, plies: 2, maxTilesLeft: 1, nested: 1, tiebreak: true},
	{name: "straightforward-id3", cgp: pegStraightforward, plies: 3, id: true, maxTilesLeft: 1, nested: 1, tiebreak: true},
	{name: "known-opp-rack", cgp: pegStraightforward, plies: 2, maxTilesLeft: 1, nested: 1, tiebreak: true, oppRack: "AACEISU"},
	{name: "pass-fra20", cgp: pegPassFRA20, plies: 2, maxTilesLeft: 1, nested: 1, tiebreak: true},
	{name: "two-in-bag-x", cgp: pegTwoInBag, plies: 1, maxTilesLeft: 1, nested: 1, tiebreak: true, only: []string{"6F..X."}},
	{name: "two-in-bag-capped", cgp: pegTwoInBag, plies: 1, maxTilesLeft: 1, nested: 0, only: []string{"12C.A."}},
	{name: "polish-cutoff", cgp: pegPolish, plies: 1, maxTilesLeft: 1, nested: 1, tiebreak: true, cutoff: true},
	{name: "manual-noah", cgp: pegManualNoah, plies: 4, maxTilesLeft: 1, nested: 1, tiebreak: true, only: []string{"2A.A.GUSTER", "15H.....TER", "15H.....URA"}},
	{name: "manual-pond-iiia", cgp: pegManual1, plies: 4, maxTilesLeft: 1, nested: 1, only: []string{"2L.P.ND"}, eventuality: "IIIA"},
	{name: "manual-fet-iiia", cgp: pegManual1, plies: 4, maxTilesLeft: 1, nested: 1, only: []string{"E5.FET"}, eventuality: "IIIA"},
	{name: "manual-pant-aiii", cgp: pegManual1, plies: 4, maxTilesLeft: 1, nested: 1, only: []string{"I3.P.NT"}, eventuality: "AIII"},
	{name: "manual-pah-grl", cgp: pegManual2, plies: 4, maxTilesLeft: 1, nested: 1, only: []string{"13M.P.."}, eventuality: "GRL"},
}

// TestWritePEGGoldens solves each pegCase with the Go engine and writes its
// results, one file a case, for the Zig engine's tests
// (src/preendgame/testdata in zigcore). It runs only when given a directory:
//
//	go test ./zigcore/crosscheck/ -run TestWritePEGGoldens -crosscheck.peggoldens $PWD/zigcore/src/preendgame/testdata
//
// A file holds the position and options, then each play with its points
// (wins plus half the draws, each bag ordering weighted by its count),
// total spread when the tiebreak worked one out, and its outcome for each
// ordering of the tiles in the bag (in the order they are drawn), or `cut`
// for a play the early cutoff stopped. `best` is the first play of Go's
// result.
func TestWritePEGGoldens(t *testing.T) {
	if *pegGoldens == "" {
		t.Skip("no -crosscheck.peggoldens directory")
	}
	cfg := config.DefaultConfig()
	for _, pc := range pegCases {
		t.Run(pc.name, func(t *testing.T) {
			out := solvePEGCase(t, cfg, pc)
			path := filepath.Join(*pegGoldens, pc.name+".txt")
			if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func solvePEGCase(t *testing.T, cfg *config.Config, pc pegCase) string {
	t.Helper()
	g := parseCGP(t, cfg, pc.cgp)
	alph := g.Alphabet()
	gd, err := kwg.GetKWG(cfg.WGLConfig(), g.LexiconName())
	if err != nil {
		t.Fatal(err)
	}
	s := new(preendgame.Solver)
	if err := s.Init(g, gd); err != nil {
		t.Fatal(err)
	}
	s.SetThreads(1)
	s.SetEndgamePlies(pc.plies)
	s.SetIterativeDeepening(pc.id)
	s.SetMaxTilesLeft(pc.maxTilesLeft)
	s.SetNestedDepthLimit(pc.nested)
	s.SetSkipTiebreaker(!pc.tiebreak)
	s.SetEarlyCutoffOptim(pc.cutoff)
	if len(pc.only) > 0 {
		var only []*move.Move
		for _, m := range staticPlayer(t, cfg, g).GenerateMoves(1 << 30) {
			if slices.Contains(pc.only, ucgiMove(m, alph)) {
				only = append(only, m)
			}
		}
		if len(only) != len(pc.only) {
			t.Fatalf("found %d of the moves %v", len(only), pc.only)
		}
		s.SetSolveOnly(only)
	}
	if pc.oppRack != "" {
		mw, err := tilemapping.ToMachineWord(pc.oppRack, alph)
		if err != nil {
			t.Fatal(err)
		}
		s.SetKnownOppRack(mw)
	}
	if pc.eventuality != "" {
		mw, err := tilemapping.ToMachineLetters(pc.eventuality, alph)
		if err != nil {
			t.Fatal(err)
		}
		s.SetTraceWriter(io.Discard)
		s.SetTraceTargetBagTail(mw)
		s.SetTraceOnce(true)
		s.SetExplainMode(true)
	}
	plays, err := s.Solve(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "# %s: the Go engine's result, written by TestWritePEGGoldens (zigcore/crosscheck).\n", pc.name)
	fmt.Fprintf(&b, "cgp %s\n", pc.cgp)
	fmt.Fprintf(&b, "options plies %d id %d maxtilesleft %d nested %d tiebreak %d cutoff %d\n",
		pc.plies, b2i(pc.id), pc.maxTilesLeft, pc.nested, b2i(pc.tiebreak), b2i(pc.cutoff))
	if len(pc.only) > 0 {
		fmt.Fprintf(&b, "only %s\n", strings.Join(pc.only, " "))
	}
	if pc.oppRack != "" {
		fmt.Fprintf(&b, "opprack %s\n", pc.oppRack)
	}
	if pc.eventuality != "" {
		fmt.Fprintf(&b, "eventuality %s\n", pc.eventuality)
	}
	for _, p := range plays {
		mv := ucgiMove(p.Play, alph)
		if p.Ignore {
			fmt.Fprintf(&b, "play %s cut\n", mv)
			continue
		}
		fmt.Fprintf(&b, "play %s points %s", mv, fmtFloat(float64(p.Points)))
		if p.HasSpread() {
			fmt.Fprintf(&b, " spread %d", p.GetSpread())
		}
		b.WriteString("\n")
		for _, o := range p.OutcomesArray() {
			drawn := slices.Clone(o.Tiles())
			slices.Reverse(drawn)
			fmt.Fprintf(&b, "outcome %s %s %d\n", tilemapping.MachineWord(drawn).UserVisible(alph), o.OutcomeResult(), o.Count())
		}
	}
	fmt.Fprintf(&b, "best %s\n", ucgiMove(plays[0].Play, alph))
	return b.String()
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
