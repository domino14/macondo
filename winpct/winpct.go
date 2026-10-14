// Package winpct reads MAGPIE's win percentage tables: for a game state (the
// bag size, and once the bag is empty the two rack sizes) the chance that the
// player on turn, ahead by some spread, wins. MAGPIE records them from
// self-play, one per letter distribution, as histograms of the swing from a
// state to the end of the game.
//
// File format (MAGPIE's win_pct_counts.h, version 2):
//
//	winpct 2 <rack size> <max bag> <max spread>
//	<sample> <bag> <on-turn rack> <off-turn rack> <swing sum> <bins...>
//
// Bin b counts swings of b - (max spread + 1); the first and last bins also
// count everything beyond them. Missing rows are zero.
package winpct

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	wglconfig "github.com/domino14/word-golib/config"
	"github.com/domino14/word-golib/cache"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/dataloaders"
)

const (
	version       = 2
	rackSize      = 7
	numSamples    = 2
	bagEmptyCells = rackSize * rackSize
)

// Table is a win percentage table, read-only once loaded.
type Table struct {
	maxBag    int
	maxSpread int
	// winPcts[cell][maxSpread+spread]
	winPcts [][]float32
	// expectedSwings[cell] is the mean swing from the state to the game's end.
	expectedSwings []float64
}

// cellIndex is the row of a state, as MAGPIE's win_pct_get_cell_index: rack
// sizes only matter once the bag is empty, and are clamped to 1..7.
func cellIndex(bag, onTurnRack, offTurnRack int) int {
	if bag > 0 {
		return bagEmptyCells + bag - 1
	}
	onTurnRack = min(max(onTurnRack, 1), rackSize)
	offTurnRack = min(max(offTurnRack, 1), rackSize)
	return (onTurnRack-1)*rackSize + offTurnRack - 1
}

func cellBag(cell int) int {
	if cell < bagEmptyCells {
		return 0
	}
	return cell - bagEmptyCells + 1
}

// MaxBag is the largest bag size the table covers.
func (t *Table) MaxBag() int { return t.maxBag }

// MaxSpread is the largest spread the table tells apart.
func (t *Table) MaxSpread() int { return t.maxSpread }

// WinPct is the chance the player on turn wins, ahead by spread with bag
// tiles in the bag and the given rack sizes. Bags past the table's largest are
// looked up at its largest.
func (t *Table) WinPct(spread, bag, onTurnRack, offTurnRack int) float32 {
	spread = min(max(spread, -t.maxSpread), t.maxSpread)
	return t.winPcts[t.cell(bag, onTurnRack, offTurnRack)][t.maxSpread+spread]
}

// ExpectedSwing is the mean change in the on-turn player's spread from the
// state to the end of the game.
func (t *Table) ExpectedSwing(bag, onTurnRack, offTurnRack int) float64 {
	return t.expectedSwings[t.cell(bag, onTurnRack, offTurnRack)]
}

func (t *Table) cell(bag, onTurnRack, offTurnRack int) int {
	return cellIndex(min(max(bag, 0), t.maxBag), onTurnRack, offTurnRack)
}

// Parse reads a table. Like MAGPIE, it pools the samples, gives a state with
// no data the nearest bag size's row (or all bag-empty states pooled), and
// reads win percentages off the histograms.
func Parse(r io.Reader) (*Table, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	if !sc.Scan() {
		if err := sc.Err(); err != nil {
			return nil, err
		}
		return nil, errors.New("winpct: empty file")
	}
	header := strings.Fields(sc.Text())
	if len(header) != 5 || header[0] != "winpct" {
		return nil, fmt.Errorf("winpct: bad header %q", sc.Text())
	}
	var h [4]int
	for i := range h {
		v, err := strconv.Atoi(header[i+1])
		if err != nil {
			return nil, fmt.Errorf("winpct: bad header %q", sc.Text())
		}
		h[i] = v
	}
	if h[0] != version || h[1] != rackSize || h[2] < 1 || h[3] < 1 {
		return nil, fmt.Errorf("winpct: version %d, rack size %d, max bag %d, max spread %d; "+
			"need version %d, rack size %d and positive maximums", h[0], h[1], h[2], h[3], version, rackSize)
	}
	maxBag, maxSpread := h[2], h[3]
	numBins := 2*maxSpread + 3
	rowLen := numBins + 1 // the bins, then the swing sum
	numCells := bagEmptyCells + maxBag
	// Samples pooled as they are read.
	pooled := make([][]float64, numCells)
	for i := range pooled {
		pooled[i] = make([]float64, rowLen)
	}
	line := 1
	for sc.Scan() {
		line++
		fields := strings.Fields(sc.Text())
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 5+numBins {
			return nil, fmt.Errorf("winpct: line %d has %d fields, want %d", line, len(fields), 5+numBins)
		}
		var key [4]int
		for i := range key {
			v, err := strconv.Atoi(fields[i])
			if err != nil {
				return nil, fmt.Errorf("winpct: line %d: %w", line, err)
			}
			key[i] = v
		}
		sample, bag, onTurn, offTurn := key[0], key[1], key[2], key[3]
		if sample < 0 || sample >= numSamples || bag < 0 || bag > maxBag ||
			onTurn < 1 || onTurn > rackSize || offTurn < 1 || offTurn > rackSize {
			return nil, fmt.Errorf("winpct: line %d: bad state %v", line, key)
		}
		row := pooled[cellIndex(bag, onTurn, offTurn)]
		swingSum, err := strconv.ParseFloat(fields[4], 64)
		if err != nil {
			return nil, fmt.Errorf("winpct: line %d: %w", line, err)
		}
		row[numBins] += swingSum
		for b := 0; b < numBins; b++ {
			v, err := strconv.ParseFloat(fields[5+b], 64)
			if err != nil {
				return nil, fmt.Errorf("winpct: line %d: %w", line, err)
			}
			if v < 0 {
				return nil, fmt.Errorf("winpct: line %d: negative count", line)
			}
			row[b] += v
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}

	total := func(row []float64) float64 {
		s := 0.0
		for _, v := range row[:numBins] {
			s += v
		}
		return s
	}
	totals := make([]float64, numCells)
	bagEmpty := make([]float64, rowLen)
	for c := range pooled {
		totals[c] = total(pooled[c])
		if cellBag(c) == 0 {
			for i, v := range pooled[c] {
				bagEmpty[i] += v
			}
		}
	}
	fallback := func(c int) []float64 {
		bag := cellBag(c)
		if bag == 0 {
			return bagEmpty
		}
		for d := 1; d < maxBag; d++ {
			for _, b := range [2]int{bag - d, bag + d} {
				if b >= 1 && b <= maxBag && totals[cellIndex(b, 0, 0)] > 0 {
					return pooled[cellIndex(b, 0, 0)]
				}
			}
		}
		return pooled[c]
	}

	t := &Table{maxBag: maxBag, maxSpread: maxSpread,
		winPcts: make([][]float32, numCells), expectedSwings: make([]float64, numCells)}
	numSpreads := 2*maxSpread + 1
	for c := range pooled {
		row := pooled[c]
		if totals[c] <= 0 {
			row = fallback(c)
		}
		tot := total(row)
		if tot <= 0 {
			kind := "nonempty bag"
			if cellBag(c) == 0 {
				kind = "bag-empty"
			}
			return nil, fmt.Errorf("winpct: no data for %s states", kind)
		}
		// Ahead by spread, the on-turn player wins when the swing exceeds
		// -spread and ties when it equals it (bin maxSpread+1-spread). Walk
		// spreads upward so the tie bin walks downward.
		wp := make([]float32, numSpreads)
		above := row[numBins-1]
		for i := 0; i < numSpreads; i++ {
			tie := numBins - 2 - i
			wp[i] = float32((above + 0.5*row[tie]) / tot)
			above += row[tie]
		}
		t.winPcts[c] = wp
		t.expectedSwings[c] = row[numBins] / tot
	}
	return t, nil
}

var (
	loadedMu sync.Mutex
	loaded   = map[string]*Table{}
)

// ForDistribution is the table for a letter distribution: MAGPIE's
// strategy/default/winpct_<distribution>.csv, else the English one (the super
// English one for a super distribution). Tables are loaded once.
func ForDistribution(cfg *wglconfig.Config, distName string) (*Table, error) {
	distName = strings.ToLower(distName)
	loadedMu.Lock()
	defer loadedMu.Unlock()
	if t, ok := loaded[distName]; ok {
		return t, nil
	}
	names := []string{distName}
	if strings.HasSuffix(distName, "_super") {
		names = append(names, "english_super")
	} else {
		names = append(names, "english")
	}
	dir := filepath.Join(dataloaders.StrategyParamsPath(cfg), "default")
	var firstErr error
	for _, name := range names {
		f, _, err := cache.Open(filepath.Join(dir, "winpct_"+name+".csv"))
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		t, err := Parse(f)
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("winpct_%s.csv: %w", name, err)
		}
		if name != distName {
			log.Warn().Str("distribution", distName).Str("using", name).
				Msg("no win percentage table for this distribution")
		}
		loaded[distName] = t
		return t, nil
	}
	return nil, fmt.Errorf("no win percentage table for %s: %w", distName, firstErr)
}
