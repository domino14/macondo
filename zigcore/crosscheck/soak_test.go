package crosscheck

import (
	"flag"
	"fmt"
	"testing"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
)

var soakSearches = flag.Int("crosscheck.soak", 0,
	"searches in the memory soak test (0 skips it): sims, plain and best-arm, on several threads, through one engine")

// TestSoak runs many searches through one engine process and watches its
// resident anonymous memory (mapped files such as rack info tables page in
// as they are read, and don't count): after the first rounds (data loaded,
// caches filled), it must stop growing. Leaks of what the engine frees are caught
// by its allocator on quit; this catches memory it keeps and never reuses.
func TestSoak(t *testing.T) {
	if *soakSearches == 0 {
		t.Skip("no -crosscheck.soak")
	}
	e := startEngine(t)
	cfg := config.DefaultConfig()
	dataPath := cfg.GetString(config.ConfigDataPath)
	if err := e.Send("setoption name data-path value " + dataPath); err != nil {
		t.Fatal(err)
	}
	if err := e.Send("setoption name threads value 4"); err != nil {
		t.Fatal(err)
	}
	bc := boardCase{lexicon: "NWL23", dist: "english", layout: board.CrosswordGameLayout}
	positions := simPositionsFrom(t, cfg, bc, automatic.DeriveSeed(7, 0), 24)
	var samples []int
	const rounds = 10
	per := max(*soakSearches/rounds, 1)
	for r := 0; r < rounds; r++ {
		for i := 0; i < per; i++ {
			p := positions[(r*per+i)%len(positions)]
			if err := e.Send("position cgp " + p); err != nil {
				t.Fatal(err)
			}
			cmd := fmt.Sprintf("go sim plies %d plays 10 iterations 40 seed %d", 2+i%3, i+1)
			if i%2 == 1 {
				cmd = fmt.Sprintf("go sim plies 2 plays 10 stop 99 cap 80 seed %d", i+1)
			}
			if _, err := e.Command(cmd, "bestmove"); err != nil {
				t.Fatal(err)
			}
		}
		kb, err := e.RSS()
		if err != nil {
			t.Skip(err)
		}
		samples = append(samples, kb)
		t.Logf("after %d searches: %d MB resident (anonymous)", (r+1)*per, kb/1024)
	}
	// From the second round on, growth stays small: allow 5% over the
	// second round's.
	if last, base := samples[len(samples)-1], samples[1]; float64(last) > 1.05*float64(base) {
		t.Fatalf("resident memory grew from %d to %d kB", base, last)
	}
}
