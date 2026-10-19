package bot

import (
	"reflect"
	"testing"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
)

func TestExtremeIndices(t *testing.T) {
	vals := []float64{3, -1, 7, 0, 5, -4, 2}
	if got, want := extremeIndices(vals, 2), []int{5, 1, 4, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("extremeIndices = %v, want %v", got, want)
	}
	if got := extremeIndices(vals[:3], 2); len(got) != 3 {
		t.Fatalf("fewer values than 2k: got %v, want all 3", got)
	}
	if got := extremeIndices(nil, 3); len(got) != 0 {
		t.Fatalf("no values: %v", got)
	}
}

func TestMLEnergyExtraFromEnv(t *testing.T) {
	for env, want := range map[string]int{"": 0, "10": 10, "0": 0, "-2": 0, "x": 0} {
		t.Setenv("MACONDO_ML_ENERGY_EXTRA", env)
		if got := mlEnergyExtra(); got != want {
			t.Errorf("%q: %d, want %d", env, got, want)
		}
	}
}

func TestPlacedSquares(t *testing.T) {
	ld, err := tilemapping.EnglishLetterDistribution(config.DefaultConfig().WGLConfig())
	if err != nil {
		t.Fatal(err)
	}
	alph := ld.TileMapping()
	// 8D WORLD: row 8 (index 7), columns D..H (3..7).
	m := move.NewScoringMoveSimple(26, "8D", "WORLD", "", alph)
	if got, want := placedSquares(m), []int{108, 109, 110, 111, 112}; !reflect.DeepEqual(got, want) {
		t.Fatalf("8D WORLD: %v, want %v", got, want)
	}
	// D8 W.RLD: column D (3), rows 8..12 (7..11), the second square played through.
	m = move.NewScoringMoveSimple(10, "D8", "W.RLD", "", alph)
	if got, want := placedSquares(m), []int{7*15 + 3, 9*15 + 3, 10*15 + 3, 11*15 + 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("D8 W.RLD: %v, want %v", got, want)
	}
}
