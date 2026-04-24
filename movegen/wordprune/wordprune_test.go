package wordprune_test

import (
	"os"
	"sort"
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/movegen/wordprune"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

var defaultConfig = config.DefaultConfig()

func skipIfNoData(t testing.TB) {
	t.Helper()
	if os.Getenv("MACONDO_DATA_PATH") == "" {
		t.Skip("MACONDO_DATA_PATH not set")
	}
}

func loadNWL23(t testing.TB) (*kwg.KWG, *tilemapping.LetterDistribution) {
	t.Helper()
	gd, err := kwg.GetKWG(defaultConfig.WGLConfig(), "NWL23")
	if err != nil {
		t.Fatal(err)
	}
	dist, err := tilemapping.GetDistribution(defaultConfig.WGLConfig(), "English")
	if err != nil {
		t.Fatal(err)
	}
	return gd, dist
}

// moveSets returns a sorted slice of ShortDescription strings for a set of moves.
func moveStrings(plays []*move.Move) []string {
	strs := make([]string, len(plays))
	for i, m := range plays {
		strs[i] = m.ShortDescription()
	}
	sort.Strings(strs)
	return strs
}

// genMoves generates all plays for the given rack on the given board+dictionary.
func genMoves(b *board.GameBoard, gd *kwg.KWG, dist *tilemapping.LetterDistribution, rack *tilemapping.Rack) []*move.Move {
	mg := movegen.NewGordonGenerator(gd, b, dist)
	mg.SetPlayRecorder(movegen.AllPlaysRecorder)
	mg.GenAll(rack, false)
	return mg.Plays()
}

// TestPrunedKWGProducesExactSameMoves verifies that the pruned KWG generates
// exactly the same moves as the full KWG for both racks in a realistic endgame
// position. This is a stronger check than subset membership.
func TestPrunedKWGProducesExactSameMoves(t *testing.T) {
	skipIfNoData(t)

	gd, dist := loadNWL23(t)
	alph := gd.GetAlphabet()

	positions := []struct {
		name  string
		rows  map[int]string
		rack0 string
		rack1 string
	}{
		{
			name:  "TESTING-on-row7",
			rows:  map[int]string{7: "   TESTING   "},
			rack0: "AEINRST",
			rack1: "OUL",
		},
		{
			name: "mid-endgame-position",
			rows: map[int]string{
				7:  "  QUIXOTIC    ",
				8:  "  E           ",
				9:  "  A  JABS     ",
				10: "  R           ",
			},
			rack0: "AELR",
			rack1: "FNOW",
		},
	}

	for _, pos := range positions {
		t.Run(pos.name, func(t *testing.T) {
			b := board.MakeBoard(board.CrosswordGameBoard)
			for row, tiles := range pos.rows {
				b.SetRow(row, tiles, alph)
			}
			cross_set.GenAllCrossSets(b, gd, dist)

			rack0 := tilemapping.RackFromString(pos.rack0, alph)
			rack1 := tilemapping.RackFromString(pos.rack1, alph)

			prunedKWG, err := wordprune.GeneratePrunedKWG(b, rack0, rack1, gd)
			if err != nil {
				t.Fatalf("GeneratePrunedKWG: %v", err)
			}
			if prunedKWG == nil {
				t.Fatal("expected non-nil pruned KWG")
			}

			// Re-generate cross-sets with the pruned KWG to mirror what the
			// endgame solver does.
			cross_set.GenAllCrossSets(b, prunedKWG, dist)

			for _, rack := range []*tilemapping.Rack{rack0, rack1} {
				fullMoves := genMoves(b, gd, dist, rack)
				prunedMoves := genMoves(b, prunedKWG, dist, rack)

				fullStrs := moveStrings(fullMoves)
				prunedStrs := moveStrings(prunedMoves)

				if len(fullStrs) != len(prunedStrs) {
					t.Errorf("rack %v: full=%d moves, pruned=%d moves", rack, len(fullStrs), len(prunedStrs))
					// Report missing moves for easier debugging.
					fullSet := make(map[string]bool, len(fullStrs))
					for _, s := range fullStrs {
						fullSet[s] = true
					}
					for _, s := range prunedStrs {
						if !fullSet[s] {
							t.Errorf("  pruned has extra move: %q", s)
						}
					}
					prunedSet := make(map[string]bool, len(prunedStrs))
					for _, s := range prunedStrs {
						prunedSet[s] = true
					}
					for _, s := range fullStrs {
						if !prunedSet[s] {
							t.Errorf("  pruned missing move: %q", s)
						}
					}
					continue
				}

				for i := range fullStrs {
					if fullStrs[i] != prunedStrs[i] {
						t.Errorf("rack %v move[%d]: full=%q pruned=%q", rack, i, fullStrs[i], prunedStrs[i])
					}
				}

				t.Logf("rack %v: %d moves match exactly", rack, len(fullStrs))
			}
		})
	}
}

// TestPrunedKWGProducesConsistentMoves verifies pruned moves are a subset of
// full moves (lighter check, useful as a quick sanity gate).
func TestPrunedKWGProducesConsistentMoves(t *testing.T) {
	skipIfNoData(t)

	gd, dist := loadNWL23(t)
	alph := gd.GetAlphabet()

	b := board.MakeBoard(board.CrosswordGameBoard)
	b.SetRow(7, "   TESTING   ", alph)
	cross_set.GenAllCrossSets(b, gd, dist)

	rack0 := tilemapping.RackFromString("AEINRST", alph)
	rack1 := tilemapping.RackFromString("OUL", alph)

	// Generate moves using the full KWG.
	mg := movegen.NewGordonGenerator(gd, b, dist)
	mg.SetPlayRecorder(movegen.AllPlaysRecorder)
	mg.GenAll(rack0, false)
	fullMoves := mg.Plays()

	// Build the pruned KWG and generate moves with it.
	prunedKWG, err := wordprune.GeneratePrunedKWG(b, rack0, rack1, gd)
	if err != nil {
		t.Fatalf("GeneratePrunedKWG: %v", err)
	}
	if prunedKWG == nil {
		t.Fatal("expected non-nil pruned KWG")
	}

	mgPruned := movegen.NewGordonGenerator(prunedKWG, b, dist)
	mgPruned.SetPlayRecorder(movegen.AllPlaysRecorder)
	mgPruned.GenAll(rack0, false)
	prunedMoves := mgPruned.Plays()

	fullSet := make(map[string]bool, len(fullMoves))
	for _, m := range fullMoves {
		fullSet[m.ShortDescription()] = true
	}
	for _, m := range prunedMoves {
		if !fullSet[m.ShortDescription()] {
			t.Errorf("pruned move %q not in full move set", m.ShortDescription())
		}
	}

	t.Logf("full KWG: %d moves, pruned KWG: %d moves (reduction: %d%%)",
		len(fullMoves), len(prunedMoves),
		100*(len(fullMoves)-len(prunedMoves))/max(len(fullMoves), 1))
}

// BenchmarkGeneratePrunedKWG measures how long it takes to build a pruned KWG
// for a realistic endgame position.
func BenchmarkGeneratePrunedKWG(b *testing.B) {
	skipIfNoData(b)

	gd, dist := loadNWL23(b)
	alph := gd.GetAlphabet()

	brd := board.MakeBoard(board.CrosswordGameBoard)
	brd.SetRow(7, "   TESTING   ", alph)
	cross_set.GenAllCrossSets(brd, gd, dist)

	rack0 := tilemapping.RackFromString("AEINRST", alph)
	rack1 := tilemapping.RackFromString("OUL", alph)

	b.ResetTimer()
	for range b.N {
		_, err := wordprune.GeneratePrunedKWG(brd, rack0, rack1, gd)
		if err != nil {
			b.Fatal(err)
		}
	}
}
