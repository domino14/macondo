package crosscheck

import (
	"context"
	"flag"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/domino14/macondo/config"
)

var exactRepeats = flag.Int("crosscheck.exactrepeats", 10,
	"solves of each endgame and setting in the exactness cross-check")

// The endgame of the Woogles blog's "Why more plies sometimes reveal the
// true best move" (NWL23; the scores are made up), and the position after
// its best move, 3D V(I)OL. The best line, 3D V(I)OL 15C AAS 10B YO
// 13A (D)UI, ends the game within 4 plies, but other lines take longer: the
// value is proven by the depth-13 search from the start (searched at odd
// depths), and by the depth-5 one after V(I)OL.
var exactEndgames = []struct {
	name, cgp string
	// The value at 4 plies, and the last depth searched before the one that
	// proves the value (one thread).
	value4, before int
}{
	{"start", "10ADDER/1BROWN3PROEM1/4I3MI2L2/3ANTIQUES1UN1/4T6AXe1/4L1G2J1NET1/3BEGET1E1T1S1/VICE2COLE1H1UP/6K4O1RE/5WOmANED1FA/HAFIZES4I2G/Y3I6A3/D3T10/R3I10/O14 LNOORVY/AAISU 400/310 0 lex NWL23;", 21, 11},
	{"after-viol", "10ADDER/1BROWN3PROEM1/3VIOL1MI2L2/3ANTIQUES1UN1/4T6AXe1/4L1G2J1NET1/3BEGET1E1T1S1/VICE2COLE1H1UP/6K4O1RE/5WOmANED1FA/HAFIZES4I2G/Y3I6A3/D3T10/R3I10/O14 AAISU/NORY 310/420 0 lex NWL23;", -1, 3},
}

// TestEndgameExact checks the engine's `exact` flag: a 4-ply search of
// these endgames is never exact, and a 25-ply one always ends exact, early,
// with the value Go's 25-ply solve gives; on one thread and on several,
// solve after solve.
func TestEndgameExact(t *testing.T) {
	e := startEngine(t)
	e.Timeout = 5 * time.Minute
	cfg := config.DefaultConfig()
	dataPath := cfg.GetString(config.ConfigDataPath)
	if dataPath == "" {
		t.Skip("no data path; set MACONDO_DATA_PATH")
	}
	if err := e.Send("setoption name data-path value " + dataPath); err != nil {
		t.Fatal(err)
	}
	for _, eg := range exactEndgames {
		t.Run(eg.name, func(t *testing.T) {
			s, _ := goEndgameSolver(t, cfg, eg.cgp)
			want, _, err := s.Solve(context.Background(), 25)
			if err != nil {
				t.Fatal(err)
			}
			if err := e.Send("position cgp " + eg.cgp); err != nil {
				t.Fatal(err)
			}
			for _, setting := range []struct {
				threads  int
				parallel string
			}{{1, ""}, {8, "pvsplit"}, {8, "abdada"}} {
				if err := e.Send(fmt.Sprintf("setoption name threads value %d", setting.threads)); err != nil {
					t.Fatal(err)
				}
				for i := 0; i < *exactRepeats; i++ {
					depths, value, exact := solveExact(t, e, 4, setting.parallel)
					if exact || value != eg.value4 {
						t.Fatalf("%d threads %s, 4 plies: value %d exact %v, want %d not exact (depths %v)",
							setting.threads, setting.parallel, value, exact, eg.value4, depths)
					}
					depths, value, exact = solveExact(t, e, 25, setting.parallel)
					if !exact || value != int(want) || depths[len(depths)-1] != 25 {
						t.Fatalf("%d threads %s, 25 plies: value %d exact %v, depths %v, want Go's %d, exact at 25",
							setting.threads, setting.parallel, value, exact, depths, want)
					}
					// On one thread the search stops at the first depth
					// that proves the value. (On several, a thread can prove
					// it sooner from the others' proven table entries.)
					if setting.threads == 1 && depths[len(depths)-2] != eg.before {
						t.Fatalf("1 thread, 25 plies: depths %v, want the proof right after depth %d", depths, eg.before)
					}
				}
			}
			if err := e.Send("setoption name threads value 1"); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// solveExact runs `go endgame` and returns the depths of its info lines, the
// last one's value and whether it is exact.
func solveExact(t *testing.T, e *Engine, plies int, parallel string) ([]int, int, bool) {
	t.Helper()
	lines, err := e.Command(strings.TrimSpace(fmt.Sprintf("go endgame plies %d %s", plies, parallel)), "bestmove")
	if err != nil {
		t.Fatal(err)
	}
	var depths []int
	value, exact := 0, false
	for _, l := range lines {
		f := strings.Fields(l)
		if len(f) < 10 || f[0] != "info" || f[1] != "endgame" || f[8] != "exact" {
			continue
		}
		d, _ := strconv.Atoi(f[3])
		depths = append(depths, d)
		value, _ = strconv.Atoi(f[5])
		exact = f[9] == "1"
	}
	if len(depths) == 0 {
		t.Fatalf("no info endgame lines: %q", lines)
	}
	return depths, value, exact
}
