package winpct

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A max spread of 1 gives five bins: swings <= -2, -1, 0, 1, >= 2.
func TestParseSmallTable(t *testing.T) {
	// Bag 2 holds swings of -1, 0, 0 and +1 across the two samples; bag 1
	// has no data and borrows bag 2's; bag-empty states have one swing of 2.
	src := `winpct 2 7 2 1
0 2 7 7 -1 0 1 1 0 0
1 2 7 7 1 0 0 1 1 0
0 0 3 4 2 0 0 0 0 1
`
	tab, err := Parse(strings.NewReader(src))
	if err != nil {
		t.Fatal(err)
	}
	// Ahead by 0: the swing must exceed 0 to win (1 of 4), equal it to tie
	// (2 of 4): 0.25 + 0.5*0.5.
	check := func(got float32, want float64) {
		t.Helper()
		if math.Abs(float64(got)-want) > 1e-6 {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	check(tab.WinPct(0, 2, 7, 7), 0.5)
	// Ahead by 1: wins on swings above -1 (3 of 4), ties on -1.
	check(tab.WinPct(1, 2, 7, 7), 0.75+0.125)
	check(tab.WinPct(-1, 2, 7, 7), 0.125)
	// Spreads past the maximum clamp.
	check(tab.WinPct(50, 2, 7, 7), 0.875)
	check(tab.WinPct(0, 1, 7, 7), 0.5)
	// Bags past the maximum use the largest.
	check(tab.WinPct(0, 9, 7, 7), 0.5)
	// Every bag-empty state borrows the pooled bag-empty row.
	check(tab.WinPct(0, 0, 1, 7), 1)
	if got := tab.ExpectedSwing(2, 7, 7); got != 0 {
		t.Fatalf("expected swing %v", got)
	}
	if got := tab.ExpectedSwing(0, 5, 5); got != 2 {
		t.Fatalf("bag-empty expected swing %v", got)
	}
}

func TestParseRejects(t *testing.T) {
	for _, src := range []string{
		"",
		"winpct 1 7 2 1\n",
		"winpct 2 8 2 1\n",
		"winpct 2 7 2 1\n0 3 7 7 0 1 0 0 0 0\n",
		"winpct 2 7 2 1\n0 2 7 7 0 1 0 0 0\n",
		"winpct 2 7 2 1\n0 2 7 7 0 1 0 0 0 0\n", // no bag-empty data
	} {
		if _, err := Parse(strings.NewReader(src)); err == nil {
			t.Errorf("parsed %q", src)
		}
	}
}

// The shipped tables load, and win% rises with the spread and is near even at
// a tied full bag.
func TestShippedTables(t *testing.T) {
	files, _ := filepath.Glob("../data/strategy/default/winpct_*.csv")
	if len(files) == 0 {
		t.Skip("no tables")
	}
	for _, fn := range files {
		f, err := os.Open(fn)
		if err != nil {
			t.Fatal(err)
		}
		tab, err := Parse(f)
		f.Close()
		if err != nil {
			t.Fatalf("%s: %v", fn, err)
		}
		for bag := 0; bag <= tab.MaxBag(); bag++ {
			prev := float32(-1)
			for s := -tab.MaxSpread(); s <= tab.MaxSpread(); s++ {
				w := tab.WinPct(s, bag, 7, 7)
				if w < prev-1e-6 || w < 0 || w > 1 {
					t.Fatalf("%s: bag %d spread %d: %v after %v", fn, bag, s, w, prev)
				}
				prev = w
			}
		}
		if w := tab.WinPct(0, tab.MaxBag(), 7, 7); w < 0.4 || w > 0.6 {
			t.Errorf("%s: tied with a full bag: %v", fn, w)
		}
	}
}
