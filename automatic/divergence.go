package automatic

import (
	"encoding/csv"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/domino14/word-golib/cache"

	"github.com/domino14/macondo/stats"
)

// Both halves of a game pair are dealt the same tiles, so the only thing that
// can make them differ is a bot choosing differently. This finds the turn where
// that first happened and shows what each bot did with the identical rack.
//
// It also catches the case that would mean the pairing itself is broken: if the
// two halves reach a turn with *different racks*, the tiles diverged rather than
// the strategies, and any comparison built on that run is measuring luck again.

// DivergenceKind says what parted the two halves of a pair.
type DivergenceKind int

const (
	// DivergenceNone means the halves played the same game throughout.
	DivergenceNone DivergenceKind = iota
	// DivergenceChoice is the normal case: same rack, different play.
	DivergenceChoice
	// DivergenceTiles means the halves were dealt different racks. The pairing
	// is broken; this should never happen.
	DivergenceTiles
	// DivergenceLength means the halves agreed on every shared turn but one ran
	// longer. Also a bug, since identical play cannot produce different games.
	DivergenceLength
)

func (k DivergenceKind) String() string {
	switch k {
	case DivergenceChoice:
		return "different choice"
	case DivergenceTiles:
		return "DIFFERENT TILES"
	case DivergenceLength:
		return "DIFFERENT LENGTH"
	}
	return "none"
}

// PairDivergence is where and how one pair's halves parted ways.
type PairDivergence struct {
	Pair   int
	GameID string
	Kind   DivergenceKind

	// Turn is the first turn the halves disagreed on, 1-based.
	Turn int
	// Rack is what both halves held on that turn. When Kind is
	// DivergenceTiles the two halves held RackA and RackB instead.
	Rack         string
	RackA, RackB string

	// BotA played PlayA in the first half; BotB played PlayB in the second,
	// from the same seat and the same tiles.
	BotA, BotB       string
	PlayA, PlayB     string
	ScoreA, ScoreB   int
	EquityA, EquityB float64

	TurnsA, TurnsB int
	// ScoreDiff is bot 1's score minus bot 2's, summed over both halves. A pair
	// that never diverges scores exactly 0 here.
	ScoreDiff int
}

// DivergenceReport summarizes a whole paired run.
type DivergenceReport struct {
	Bot1, Bot2 string
	Pairs      int
	Divergent  int
	ExactTies  int
	// Broken counts pairs whose halves saw different tiles or ran different
	// lengths -- bugs rather than strategy differences.
	Broken int
	// NoTurnData counts divergent pairs the per-turn log had nothing for.
	NoTurnData int
	// TurnStats is the first-divergence turn across every divergent pair.
	TurnStats *stats.Statistic
	// EarliestTurn and LatestTurn bound the same distribution.
	EarliestTurn, LatestTurn int
	// Details covers the first maxDetails divergent pairs, in pair order.
	Details []PairDivergence
	// Truncated is how many divergent pairs are not in Details.
	Truncated int
}

// turnRow is one line of the per-turn log.
type turnRow struct {
	nick   string
	turn   int
	rack   string
	play   string
	score  int
	equity float64
}

// pairInfo is what the per-game log says about one pair.
type pairInfo struct {
	pair      int
	divergent bool
	scoreDiff int
	rows      int
}

// TurnLogFor guesses the per-turn log that goes with a per-game log: autoplay
// writes games-{id}.txt beside {id}.txt.
func TurnLogFor(gamesFile string) string {
	dir, base := filepath.Split(gamesFile)
	if !strings.HasPrefix(base, "games-") {
		return ""
	}
	return filepath.Join(dir, strings.TrimPrefix(base, "games-"))
}

// AnalyzeDivergence reads a paired run's per-game log and the per-turn log
// beside it, and reports where each divergent pair's halves parted ways.
// maxDetails caps how many pairs are described individually; the counts and the
// turn statistics cover them all.
func AnalyzeDivergence(gamesFile, turnFile string, maxDetails int) (*DivergenceReport, error) {
	report := &DivergenceReport{TurnStats: &stats.Statistic{}}
	pairs, err := readPairInfo(gamesFile, report)
	if err != nil {
		return nil, err
	}
	if report.Pairs == 0 {
		return nil, fmt.Errorf("no games found in %s", gamesFile)
	}

	// Only the divergent pairs need their turns read back.
	wanted := map[string]bool{}
	for gid, p := range pairs {
		if p.divergent {
			wanted[gid] = true
		}
	}
	turns, err := readTurnLog(turnFile, wanted)
	if err != nil {
		return nil, err
	}

	ordered := make([]*pairInfo, 0, len(wanted))
	gids := make(map[*pairInfo]string, len(wanted))
	for gid, p := range pairs {
		if p.divergent {
			ordered = append(ordered, p)
			gids[p] = gid
		}
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].pair < ordered[j].pair })

	if maxDetails < 0 {
		maxDetails = 0
	}
	for _, p := range ordered {
		gid := gids[p]
		rows := turns[gid]
		first, second, ok := splitHalves(rows)
		if !ok {
			report.NoTurnData++
			continue
		}
		d := comparePair(p, gid, first, second, report.Bot1, report.Bot2)
		if d.Kind == DivergenceTiles || d.Kind == DivergenceLength {
			report.Broken++
		}
		if d.Kind != DivergenceNone {
			report.TurnStats.Push(float64(d.Turn))
			if report.EarliestTurn == 0 || d.Turn < report.EarliestTurn {
				report.EarliestTurn = d.Turn
			}
			if d.Turn > report.LatestTurn {
				report.LatestTurn = d.Turn
			}
		}
		if len(report.Details) < maxDetails {
			report.Details = append(report.Details, d)
		}
	}
	report.Truncated = report.Divergent - report.NoTurnData - len(report.Details)
	if report.Truncated < 0 {
		report.Truncated = 0
	}
	return report, nil
}

// readPairInfo reads the per-game log, filling in the report's counts and
// returning one entry per pair, keyed by game ID.
func readPairInfo(gamesFile string, report *DivergenceReport) (map[string]*pairInfo, error) {
	file, _, err := cache.Open(gamesFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	r := csv.NewReader(file)
	r.FieldsPerRecord = -1
	pairs := map[string]*pairInfo{}
	sawHeader := false

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if record[0] == "playerID" {
			return nil, fmt.Errorf(
				"this is a per-turn log; pass the per-game log (games-*.txt)")
		}
		if record[0] == "gameID" {
			sawHeader = true
			if len(record) < 10 || record[8] != "pair" || record[9] != "divergent" {
				return nil, fmt.Errorf(
					"%s has no pair/divergent columns; it was not run with -gamepairs",
					filepath.Base(gamesFile))
			}
			report.Bot1 = strings.TrimSuffix(record[1], "_score")
			report.Bot2 = strings.TrimSuffix(record[2], "_score")
			continue
		}
		if !sawHeader || len(record) < 10 {
			continue
		}
		p1, err1 := strconv.Atoi(record[1])
		p2, err2 := strconv.Atoi(record[2])
		pairIdx, err3 := strconv.Atoi(record[8])
		if err1 != nil || err2 != nil || err3 != nil {
			return nil, fmt.Errorf("could not parse game row %q", strings.Join(record, ","))
		}
		gid := record[0]
		p, ok := pairs[gid]
		if !ok {
			p = &pairInfo{pair: pairIdx, divergent: record[9] == "true"}
			pairs[gid] = p
		}
		p.scoreDiff += p1 - p2
		p.rows++
	}

	for _, p := range pairs {
		report.Pairs++
		if p.divergent {
			report.Divergent++
		}
		if p.scoreDiff == 0 {
			report.ExactTies++
		}
	}
	return pairs, nil
}

// readTurnLog reads the per-turn log, keeping only the games asked for.
func readTurnLog(turnFile string, wanted map[string]bool) (map[string][]turnRow, error) {
	out := map[string][]turnRow{}
	if len(wanted) == 0 {
		return out, nil
	}
	file, _, err := cache.Open(turnFile)
	if err != nil {
		return nil, fmt.Errorf("opening per-turn log %s: %w", turnFile, err)
	}
	defer file.Close()

	r := csv.NewReader(file)
	// The last column is only present for inferring bots, so rows are ragged.
	r.FieldsPerRecord = -1
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(record) < 10 || record[0] == "playerID" {
			continue
		}
		if !wanted[record[1]] {
			continue
		}
		turn, err := strconv.Atoi(record[2])
		if err != nil {
			continue
		}
		score, _ := strconv.Atoi(record[5])
		equity, _ := strconv.ParseFloat(record[9], 64)
		out[record[1]] = append(out[record[1]], turnRow{
			nick:   record[0],
			turn:   turn,
			rack:   record[3],
			play:   strings.TrimSpace(record[4]),
			score:  score,
			equity: equity,
		})
	}
	return out, nil
}

// splitHalves cuts a pair's turns in two. Both halves of a pair carry the same
// game ID -- it comes from the seed they share -- so they are told apart by the
// turn counter starting over.
func splitHalves(rows []turnRow) ([]turnRow, []turnRow, bool) {
	if len(rows) < 2 {
		return nil, nil, false
	}
	split := -1
	for i := 1; i < len(rows); i++ {
		if rows[i].turn <= rows[i-1].turn {
			split = i
			break
		}
	}
	if split <= 0 {
		return nil, nil, false
	}
	return rows[:split], rows[split:], true
}

// comparePair walks both halves together and stops at the first disagreement.
func comparePair(p *pairInfo, gid string, first, second []turnRow, bot1, bot2 string) PairDivergence {
	d := PairDivergence{
		Pair:      p.pair,
		GameID:    gid,
		ScoreDiff: p.scoreDiff,
		TurnsA:    len(first),
		TurnsB:    len(second),
	}
	// Seats swap between halves, so the same seat is played by the other bot.
	nameFor := func(nick string) string {
		if nick == "p1" {
			return bot1
		}
		return bot2
	}

	n := min(len(first), len(second))
	for i := 0; i < n; i++ {
		a, b := first[i], second[i]
		if a.rack != b.rack {
			// Identical tiles are the whole premise of a pair. Different racks
			// mean the bag went out of step, which is a bug in the pairing.
			d.Kind = DivergenceTiles
			d.Turn = a.turn
			d.RackA, d.RackB = a.rack, b.rack
			d.BotA, d.BotB = nameFor(a.nick), nameFor(b.nick)
			d.PlayA, d.PlayB = a.play, b.play
			d.ScoreA, d.ScoreB = a.score, b.score
			d.EquityA, d.EquityB = a.equity, b.equity
			return d
		}
		if a.play != b.play {
			d.Kind = DivergenceChoice
			d.Turn = a.turn
			d.Rack = a.rack
			d.BotA, d.BotB = nameFor(a.nick), nameFor(b.nick)
			d.PlayA, d.PlayB = a.play, b.play
			d.ScoreA, d.ScoreB = a.score, b.score
			d.EquityA, d.EquityB = a.equity, b.equity
			return d
		}
	}
	if len(first) != len(second) {
		// Same plays throughout but a different number of them, which identical
		// play cannot produce.
		d.Kind = DivergenceLength
		d.Turn = n
	}
	return d
}

// FormatDivergence renders a divergence report for the shell.
func FormatDivergence(r *DivergenceReport) string {
	var b strings.Builder

	pct := 0.0
	if r.Pairs > 0 {
		pct = 100.0 * float64(r.Divergent) / float64(r.Pairs)
	}
	fmt.Fprintf(&b, "Pairs: %d    divergent: %d (%.1f%%)    exact ties: %d\n",
		r.Pairs, r.Divergent, pct, r.ExactTies)

	if r.Divergent == 0 {
		b.WriteString(
			"\nNo pair diverged: both halves of every pair played the same game.\n" +
				"That is what a bot playing itself should do. Between two different\n" +
				"bots it means they never once disagreed, and the run is measuring\n" +
				"nothing.\n")
		return b.String()
	}

	if r.TurnStats.Iterations() > 0 {
		fmt.Fprintf(&b, "First divergence on turn: mean %.1f, earliest %d, latest %d\n",
			r.TurnStats.Mean(), r.EarliestTurn, r.LatestTurn)
	}
	if r.Broken > 0 {
		fmt.Fprintf(&b,
			"\n  ** %d pair(s) diverged in TILES, not choices. The two halves are\n"+
				"     supposed to be dealt identical racks, so this is a bug in the\n"+
				"     pairing and the run is back to comparing luck. **\n", r.Broken)
	}
	if r.NoTurnData > 0 {
		fmt.Fprintf(&b, "\n  (%d divergent pair(s) had no per-turn data to explain)\n",
			r.NoTurnData)
	}

	for _, d := range r.Details {
		fmt.Fprintf(&b, "\npair %-5d %s\n", d.Pair, d.GameID)
		lead := r.Bot1
		diff := d.ScoreDiff
		if diff < 0 {
			lead, diff = r.Bot2, -diff
		}
		fmt.Fprintf(&b, "  pair result: %s +%d    turns: %d vs %d\n",
			lead, diff, d.TurnsA, d.TurnsB)

		switch d.Kind {
		case DivergenceNone:
			b.WriteString("  no difference found in the per-turn log\n")
			continue
		case DivergenceLength:
			fmt.Fprintf(&b,
				"  ** same plays for all %d shared turns but different lengths **\n", d.Turn)
			continue
		case DivergenceTiles:
			fmt.Fprintf(&b, "  ** turn %d: the halves held DIFFERENT racks: %s vs %s **\n",
				d.Turn, d.RackA, d.RackB)
		case DivergenceChoice:
			fmt.Fprintf(&b, "  turn %d, both holding %s:\n", d.Turn, d.Rack)
		}
		fmt.Fprintf(&b, "    %-20s %-14s %4d pts   eq %7.2f\n",
			d.BotA, d.PlayA, d.ScoreA, d.EquityA)
		fmt.Fprintf(&b, "    %-20s %-14s %4d pts   eq %7.2f\n",
			d.BotB, d.PlayB, d.ScoreB, d.EquityB)
	}

	if r.Truncated > 0 {
		fmt.Fprintf(&b, "\n(%d more divergent pair(s) not shown; raise -limit to see them)\n",
			r.Truncated)
	}
	return b.String()
}
