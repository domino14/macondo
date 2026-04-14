//go:build wmp

package montecarlo

import (
	"runtime"
	"testing"

	"github.com/domino14/macondo/movegen"
	wmppkg "github.com/domino14/macondo/wmp"
	"github.com/rs/zerolog"
)

func loadWMPForBench(b *testing.B) *wmppkg.WMP {
	b.Helper()
	_, wmpPath := requireWMPSetup(b)
	w, err := wmppkg.LoadFromFile("CSW24", wmpPath)
	if err != nil {
		b.Fatal(err)
	}
	return w
}

func buildSimmersWMP(b *testing.B, spec positionSpec, w *wmppkg.WMP) []*Simmer {
	b.Helper()
	simmers := buildSimmers(b, spec)
	for _, s := range simmers {
		for i := range s.aiplayers {
			gen := s.aiplayers[i].MoveGenerator().(*movegen.GordonGenerator)
			gen.SetWMP(w)
		}
	}
	return simmers
}

func BenchmarkSimEarlyGameWMP(b *testing.B) {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	runtime.MemProfileRate = 0
	w := loadWMPForBench(b)
	simmers := buildSimmersWMP(b, earlyGame, w)
	runSimBench(b, simmers)
}

func BenchmarkSimMidGameWMP(b *testing.B) {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	runtime.MemProfileRate = 0
	w := loadWMPForBench(b)
	simmers := buildSimmersWMP(b, midGame, w)
	runSimBench(b, simmers)
}

func BenchmarkSimLateGameWMP(b *testing.B) {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	runtime.MemProfileRate = 0
	w := loadWMPForBench(b)
	simmers := buildSimmersWMP(b, lateGame, w)
	runSimBench(b, simmers)
}
