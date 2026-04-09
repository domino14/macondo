// montecarlo_wmp_bench_test.go: side-by-side benchmarks for sim performance
// with and without WMP, to measure the speedup the WMP integration provides
// for shadow play during simulation rollouts.
//
// Both benchmarks use the same CGP, lexicon (CSW24), thread count, ply
// depth, and rollout loop. The only difference is whether WMP is wired
// into the simmer's per-thread move generators after PrepareSim.
//
// Lookup order for required files:
//   - macondo data dir: $MACONDO_DATA_PATH (via DefaultConfig). Must contain
//     letterdistributions/english, strategy/CSW24/leaves.klv2, and
//     lexica/gaddag/CSW24.kwg.
//   - WMP file: $MACONDO_WMP_FILE if set, else
//     $MACONDO_DATA_PATH/lexica/CSW24.wmp.
//
// Benchmarks (and the equivalence test that shares this setup) skip
// automatically when any required file is missing.
package montecarlo

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/movegen"
	wmppkg "github.com/domino14/macondo/wmp"
)

const (
	// CGP from the existing BenchmarkSim, switched to CSW24 since
	// that's the lexicon we have local KWG + leaves + WMP for.
	benchSimCGP = "C14/O2TOY9/mIRADOR8/F4DAB2PUGH1/I5GOOEY3V/T4XI2MALTHA/14N/6GUM3OWN/7PEW2DOE/9EF1DOR/2KUNA1J1BEVELS/3TURRETs2S2/7A4T2/7N7/7S7 EEEIILZ/ 336/298 0 lex CSW24;"
)

// wmpDataPath returns the macondo data directory used by these tests.
// It defers to DefaultConfig (which respects $MACONDO_DATA_PATH).
func wmpDataPath() string {
	return DefaultConfig.GetString("data-path")
}

// wmpFilePath returns the path to the CSW24 WMP file. Override with
// $MACONDO_WMP_FILE; otherwise defaults to lexica/CSW24.wmp under the
// macondo data directory.
func wmpFilePath() string {
	if p := os.Getenv("MACONDO_WMP_FILE"); p != "" {
		return p
	}
	return filepath.Join(wmpDataPath(), "lexica", "CSW24.wmp")
}

// requireWMPSetup is a test helper (T or B) that skips the caller if any
// of the files needed by the WMP benchmarks / equivalence test are
// missing. Returns (dataPath, wmpFilePath) on success.
func requireWMPSetup(tb testing.TB) (string, string) {
	tb.Helper()
	dataPath := wmpDataPath()
	if dataPath == "" {
		tb.Skip("MACONDO_DATA_PATH is not set; skipping WMP benchmark/test")
	}
	if _, err := os.Stat(filepath.Join(dataPath, "lexica", "gaddag", "CSW24.kwg")); err != nil {
		tb.Skipf("CSW24.kwg not found under %s/lexica/gaddag/", dataPath)
	}
	if _, err := os.Stat(filepath.Join(dataPath, "strategy", "CSW24", "leaves.klv2")); err != nil {
		tb.Skipf("CSW24 leaves not found at %s/strategy/CSW24/leaves.klv2", dataPath)
	}
	wmpPath := wmpFilePath()
	if _, err := os.Stat(wmpPath); err != nil {
		tb.Skipf("WMP file not found at %s (set $MACONDO_WMP_FILE to override location)", wmpPath)
	}
	return dataPath, wmpPath
}

// setupSimBenchmark builds the Simmer + plays for the benchmark and
// returns it ready to call simSingleIteration. Skips the calling
// benchmark if required files are missing.
func setupSimBenchmark(b *testing.B) (*Simmer, int) {
	b.Helper()
	requireWMPSetup(b)

	game, err := cgp.ParseCGP(DefaultConfig, benchSimCGP)
	if err != nil {
		b.Fatalf("ParseCGP: %v", err)
	}
	game.RecalculateBoard()

	calc, err := equity.NewCombinedStaticCalculator(
		"CSW24", DefaultConfig, "", equity.PEGAdjustmentFilename)
	if err != nil {
		b.Fatalf("NewCombinedStaticCalculator: %v", err)
	}
	calcs := []equity.EquityCalculator{calc}
	leaves := calc

	gd, err := kwg.GetKWG(game.Config().WGLConfig(), game.LexiconName())
	if err != nil {
		b.Fatalf("GetKWG: %v", err)
	}

	// Initial play enumeration to pick the top 10 plays for the sim.
	gen := movegen.NewGordonGenerator(gd, game.Board(), game.Rules().LetterDistribution())
	gen.GenAll(game.RackFor(0), false)
	plays := gen.Plays()
	if len(plays) > 10 {
		plays = plays[:10]
	}

	zerolog.SetGlobalLevel(zerolog.Disabled)

	const plies = 2
	simmer := &Simmer{}
	simmer.Init(game.Game, calcs, leaves, DefaultConfig)
	simmer.SetThreads(1)
	simmer.PrepareSim(plies, plays)
	return simmer, plies
}

// BenchmarkSimCSW24NoWMP measures the baseline sim throughput without
// any WMP integration — i.e., the same path the main branch uses.
func BenchmarkSimCSW24NoWMP(b *testing.B) {
	simmer, plies := setupSimBenchmark(b)

	runtime.MemProfileRate = 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simmer.simSingleIteration(context.Background(), plies, 0, uint64(i+1), nil)
	}
}

// BenchmarkSimCSW24WMP measures the same sim but with the CSW24 WMP
// wired into every per-thread move generator before the rollout loop.
// All other state (CGP, plays list, plies, threads) is identical to
// the no-WMP variant.
func BenchmarkSimCSW24WMP(b *testing.B) {
	_, wmpPath := requireWMPSetup(b)
	w, err := wmppkg.LoadFromFile("CSW24", wmpPath)
	if err != nil {
		b.Fatalf("loading CSW24.wmp: %v", err)
	}

	simmer, plies := setupSimBenchmark(b)

	// Wire the WMP into every per-thread move generator that the
	// simmer's aiplayers were created with in PrepareSim.
	for i := range simmer.aiplayers {
		gen := simmer.aiplayers[i].MoveGenerator().(*movegen.GordonGenerator)
		gen.SetWMP(w)
	}

	runtime.MemProfileRate = 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simmer.simSingleIteration(context.Background(), plies, 0, uint64(i+1), nil)
	}
}

// setupGenAllBenchmark builds a single GordonGenerator + rack for the
// benchSimCGP position so the GenAll benchmarks below can isolate
// move-gen cost from the simmer's per-iteration setup.
func setupGenAllBenchmark(b *testing.B) (*movegen.GordonGenerator, *tilemapping.Rack) {
	b.Helper()
	requireWMPSetup(b)

	game, err := cgp.ParseCGP(DefaultConfig, benchSimCGP)
	if err != nil {
		b.Fatalf("ParseCGP: %v", err)
	}
	game.RecalculateBoard()

	calc, err := equity.NewCombinedStaticCalculator(
		"CSW24", DefaultConfig, "", equity.PEGAdjustmentFilename)
	if err != nil {
		b.Fatalf("NewCombinedStaticCalculator: %v", err)
	}

	gd, err := kwg.GetKWG(game.Config().WGLConfig(), game.LexiconName())
	if err != nil {
		b.Fatalf("GetKWG: %v", err)
	}

	gen := movegen.NewGordonGenerator(gd, game.Board(), game.Rules().LetterDistribution())
	gen.SetEquityCalculators([]equity.EquityCalculator{calc})
	gen.SetGame(game.Game)
	gen.SetPlayRecorderTopPlay()

	rack := game.RackFor(0)
	return gen, rack
}

// BenchmarkGenAllCSW24NoWMP measures GenAll (with shadow + best-first
// recursive_gen) on the benchmark position, no WMP.
func BenchmarkGenAllCSW24NoWMP(b *testing.B) {
	gen, rack := setupGenAllBenchmark(b)
	zerolog.SetGlobalLevel(zerolog.Disabled)
	runtime.MemProfileRate = 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.GenAll(rack, false)
	}
}

// BenchmarkGenAllCSW24WMP measures the same GenAll with WMP enabled.
func BenchmarkGenAllCSW24WMP(b *testing.B) {
	gen, rack := setupGenAllBenchmark(b)
	_, wmpPath := requireWMPSetup(b)
	w, err := wmppkg.LoadFromFile("CSW24", wmpPath)
	if err != nil {
		b.Fatalf("loading CSW24.wmp: %v", err)
	}
	gen.SetWMP(w)
	zerolog.SetGlobalLevel(zerolog.Disabled)
	runtime.MemProfileRate = 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.GenAll(rack, false)
	}
}
