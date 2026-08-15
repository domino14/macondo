package stats

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/aybabtme/uniplot/histogram"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/word-golib/tilemapping"
)

type Heat struct {
	numHits       int
	fractionOfMax float64
}

func (h Heat) FractionOfMax() float64 {
	return h.fractionOfMax
}

type HeatMap struct {
	board    *board.GameBoard
	squares  [][]Heat
	alphabet *tilemapping.TileMapping
}

func (h HeatMap) Squares() [][]Heat {
	return h.squares
}

var exchRe = regexp.MustCompile(`\((exch|exchange) ([^)]+)\)`)
var throughPlayRe = regexp.MustCompile(`\(([^)]+)\)`)

type SimStats struct {
	board  *board.GameBoard
	game   *bot.BotTurnPlayer
	simmer *montecarlo.Simmer

	oppHist histogram.Histogram
	ourHist histogram.Histogram
}

func NewSimStats(simmer *montecarlo.Simmer, g *bot.BotTurnPlayer) *SimStats {
	return &SimStats{simmer: simmer, game: g, board: g.Board()}
}

func (ss *SimStats) CalculateHeatmap(play string, ply int) (*HeatMap, error) {
	iters, err := ss.simmer.ReadHeatmap()
	if err != nil {
		return nil, err
	}
	log.Debug().Msgf("Read %d log lines", len(iters))
	h := &HeatMap{
		squares:  make([][]Heat, ss.board.Dim()),
		board:    ss.board,
		alphabet: ss.game.Alphabet(),
	}
	for idx := range h.squares {
		h.squares[idx] = make([]Heat, h.board.Dim())
	}

	maxNumHits := 0
	log.Debug().Msg("parsing-iterations")
	normalizedPlay := Normalize(play)

	for i := range iters {
		for j := range iters[i].Plays {
			if normalizedPlay != Normalize(iters[i].Plays[j].Play) {
				continue
			}
			if len(iters[i].Plays[j].Plies) <= ply {
				continue
			}
			analyzedPlay := Normalize(iters[i].Plays[j].Plies[ply].Play)

			if strings.HasPrefix(analyzedPlay, "exchange ") ||
				analyzedPlay == "pass" || analyzedPlay == "UNHANDLED" {
				continue
			}

			// this is a tile-play move.
			playFields := strings.Fields(analyzedPlay)
			if len(playFields) != 2 {
				return nil, errors.New("unexpected play " + analyzedPlay)
			}
			coords := strings.ToUpper(playFields[0])
			row, col, vertical := move.FromBoardGameCoords(coords, false)
			mw, err := tilemapping.ToMachineWord(playFields[1], ss.game.Alphabet())
			if err != nil {
				return nil, err
			}
			ri, ci := 1, 0
			if !vertical {
				ri, ci = 0, 1
			}

			for idx := range mw {
				if mw[idx] == 0 {
					continue // playthrough doesn't create heat map
				}
				newRow := row + (ri * idx)
				newCol := col + (ci * idx)
				h.squares[newRow][newCol].numHits++
				if h.squares[newRow][newCol].numHits > maxNumHits {
					maxNumHits = h.squares[newRow][newCol].numHits
				}
			}

		}
	}

	for ri := range h.squares {
		for ci := range h.squares[ri] {
			h.squares[ri][ci].fractionOfMax = float64(h.squares[ri][ci].numHits) / float64(maxNumHits)
		}
	}

	return h, nil
}

func Normalize(p string) string {
	// Trim leading and trailing whitespace
	trimmed := strings.TrimSpace(p)

	if trimmed == "(Pass)" {
		return "pass"
	}

	// Check for "(exch FOO)" or "(exchange FOO)" and extract the content
	if strings.HasPrefix(trimmed, "(exch ") || strings.HasPrefix(trimmed, "(exchange ") {
		// Define a regex to extract "exchange FOO" from "(exch FOO)" or "(exchange FOO)"
		matches := exchRe.FindStringSubmatch(trimmed)
		if len(matches) == 3 {
			return "exchange " + matches[2]
		}
	}

	// Define a regular expression to match groups in parentheses

	// Replace each match with as many dots as there are characters inside the parentheses
	normalized := throughPlayRe.ReplaceAllStringFunc(trimmed, func(match string) string {
		// Extract the content inside parentheses using capturing group
		content := throughPlayRe.FindStringSubmatch(match)[1]
		return strings.Repeat(".", len(content))
	})

	return normalized
}

// getHeatColor returns an ANSI escape sequence for a given heat level.
func getHeatColor(fraction float64) string {
	// Map the fraction (0 to 1) to grayscale colors (232 to 255 in ANSI 256-color palette)
	// 232 is darkest (black), 255 is lightest (white)
	start := 232
	end := 255
	colorCode := int(float64(start) + fraction*float64(end-start))
	return fmt.Sprintf("\033[48;5;%dm", colorCode) // Background color
}

// display renders the heatmap to the terminal.
func (h HeatMap) Display() {
	fmt.Println()
	reset := "\033[0m" // Reset color
	for ri, row := range h.squares {
		for ci, heat := range row {
			color := getHeatColor(heat.fractionOfMax)
			letter := h.board.SQDisplayStr(ri, ci, h.alphabet, false)
			fmt.Printf("%s%s%s", color, letter, reset) // Colored block
		}
		fmt.Println() // Newline after each row
	}
}

type nextPlay struct {
	play          string
	ifdraw        string
	score         int
	bingo         bool
	count         int
	fractionOfMax float64
}

func addNextPlay(play string, score int, bingo bool, plays map[string]*nextPlay) {
	if _, ok := plays[play]; !ok {
		plays[play] = &nextPlay{play: play, score: score, bingo: bingo}
	}
	plays[play].count = plays[play].count + 1
}

// playFamily groups the plays that make the same word in the same spot and
// differ only in which tile is played as the blank. (Z)WIEBAcK and (Z)WIEbACK
// are genuinely different plays - they score differently and need a different
// tile drawn - but a player thinks of them as one opportunity, so we show them
// as a single row carrying the combined chance of making the play by any
// route, with the individual routes listed underneath.
type playFamily struct {
	label    string // the play, upper-cased: the key we group on
	variants []*nextPlay
	count    int
	minScore int
	maxScore int
}

// name labels the family in the table. A family with a single play keeps that
// play's exact notation; a grouped one is upper-cased, since no single
// notation is correct for all of its variants.
func (f *playFamily) name() string {
	if len(f.variants) == 1 {
		return f.variants[0].play
	}
	return f.label
}

func (f *playFamily) scoreStr() string {
	if f.minScore == f.maxScore {
		return strconv.Itoa(f.maxScore)
	}
	return fmt.Sprintf("%d-%d", f.minScore, f.maxScore)
}

// drawStr renders the draws that unlock this family. Each alternative is a
// complete draw, and they are joined with a pipe meaning "or": {EI|E?|I?}
// means drawing both E and I, or E and the second blank, or I and the second
// blank. A "-" alternative is a route that needs no draw at all. A trailing
// "..." means the list was too wide for the column; every alternative is
// spelled out on the family's variant lines anyway.
func (f *playFamily) drawStr() string {
	if len(f.variants) == 1 {
		return f.variants[0].ifdraw
	}
	seen := map[string]bool{}
	opts := []string{}
	for _, np := range f.variants {
		opt := strings.Trim(np.ifdraw, "{}")
		if opt == "" {
			opt = "-"
		}
		if !seen[opt] {
			seen[opt] = true
			opts = append(opts, opt)
		}
	}
	if len(opts) == 1 && opts[0] == "-" {
		return ""
	}
	cell := "{" + strings.Join(opts, "|") + "}"
	for len(cell) > drawColWidth && len(opts) > 1 {
		opts = opts[:len(opts)-1]
		cell = "{" + strings.Join(opts, "|") + "|...}"
	}
	return cell
}

// isTilePlay reports whether the family is a tile placement, as opposed to an
// exchange or a pass.
func (f *playFamily) isTilePlay() bool {
	// Normalize is case-sensitive, so ask a variant rather than the
	// upper-cased label.
	p := Normalize(f.variants[0].play)
	return !strings.HasPrefix(p, "exchange ") && p != "pass" && p != "UNHANDLED"
}

func groupPlays(npmap map[string]*nextPlay) []*playFamily {
	byLabel := map[string]*playFamily{}
	for _, np := range npmap {
		label := strings.ToUpper(np.play)
		f, ok := byLabel[label]
		if !ok {
			f = &playFamily{label: label, minScore: np.score, maxScore: np.score}
			byLabel[label] = f
		}
		f.variants = append(f.variants, np)
		f.count += np.count
		f.minScore = min(f.minScore, np.score)
		f.maxScore = max(f.maxScore, np.score)
	}
	l := make([]*playFamily, 0, len(byLabel))
	for _, f := range byLabel {
		sort.Slice(f.variants, func(i, j int) bool {
			if f.variants[i].count != f.variants[j].count {
				return f.variants[i].count > f.variants[j].count
			}
			if f.variants[i].score != f.variants[j].score {
				return f.variants[i].score > f.variants[j].score
			}
			return f.variants[i].play < f.variants[j].play
		})
		l = append(l, f)
	}
	return l
}

func sortedFamilies(npmap map[string]*nextPlay) []*playFamily {
	l := groupPlays(npmap)
	sort.Slice(l, func(i, j int) bool {
		if l[i].count != l[j].count {
			return l[i].count > l[j].count
		}
		if l[i].maxScore != l[j].maxScore {
			return l[i].maxScore > l[j].maxScore
		}
		return l[i].label < l[j].label
	})
	return l
}

func sortedFamiliesByScore(npmap map[string]*nextPlay) []*playFamily {
	l := groupPlays(npmap)
	sort.Slice(l, func(i, j int) bool {
		if l[i].maxScore != l[j].maxScore {
			return l[i].maxScore > l[j].maxScore
		}
		if l[i].count != l[j].count {
			return l[i].count > l[j].count
		}
		return l[i].label < l[j].label
	})
	return l
}

// neededDraw returns the tiles we'd have to draw, on top of the leave from the
// play being analyzed, to be able to make this play.
func neededDraw(st *SimStats, ourPlayLeave, play string) string {
	analyzedPlay := Normalize(play)
	playFields := strings.Fields(analyzedPlay)
	if len(playFields) != 2 {
		panic("unexpected play " + analyzedPlay)
	}
	mw, err := tilemapping.ToMachineWord(playFields[1], st.game.Alphabet())
	if err != nil {
		panic("error converting to machine word " + err.Error())
	}
	leaveMW, err := tilemapping.ToMachineWord(ourPlayLeave, st.game.Alphabet())
	if err != nil {
		panic("error converting to machine word " + err.Error())
	}
	// check what letters in leaveMW are in mw, and calculate the
	// letters we'd need to draw to make the play
	draw := []tilemapping.MachineLetter{}

	avail := make(map[tilemapping.MachineLetter]int)
	for _, l := range leaveMW {
		avail[l]++
	}
	for _, l := range mw {
		if l == 0 {
			continue // playthrough doesn't create heat map
		}
		normalizedLetter := l.IntrinsicTileIdx()

		if avail[normalizedLetter] > 0 {
			avail[normalizedLetter]--
		} else {
			draw = append(draw, normalizedLetter)
		}
	}

	if len(draw) == 0 {
		return ""
	}
	return "{" + tilemapping.MachineWord(draw).UserVisible(st.game.Alphabet()) + "}"
}

// Column formats. Every column pads to one less than its width and then emits
// a literal space, so a value that overflows its column still can't run into
// the next one - long plays and long lists of alternative draws stay readable
// and stay parseable.
const (
	drawColWidth = 13

	playCol  = "%-19s "
	drawCol  = "%-13s "
	scoreCol = "%-8s "
	countCol = "%-8s "
	pctCol   = "%-16.2f"
	// variants are indented under their family; the marker plus the narrower
	// play column line up with the family's play column.
	variantMarker  = "  - "
	variantPlayCol = "%-15s "
)

func playStatsStr(st *SimStats, ourPlayLeave string, families []*playFamily, desc string, maxToDisplay int, totalPlayCount int, showBingos bool,
	showNeededDraw bool) string {
	var ss strings.Builder
	if len(families) == 0 {
		return ""
	}
	pct := func(count int) float64 {
		return float64(count*100) / float64(totalPlayCount)
	}

	ss.WriteString(desc + "\n")
	if showNeededDraw {
		fmt.Fprintf(&ss, playCol+drawCol+scoreCol+countCol+"%-16s\n", "Play", "Needed Draw", "Score", "Count", "% of time")
	} else {
		fmt.Fprintf(&ss, playCol+scoreCol+countCol+"%-16s\n", "Play", "Score", "Count", "% of time")
	}

	bingos := 0
	for i, fam := range families {
		for _, np := range fam.variants {
			if np.bingo {
				bingos += np.count
			}
		}
		if i >= maxToDisplay {
			continue
		}

		if !showNeededDraw {
			fmt.Fprintf(&ss, playCol+scoreCol+countCol+pctCol+"\n", fam.name(), fam.scoreStr(),
				strconv.Itoa(fam.count), pct(fam.count))
			if len(fam.variants) > 1 {
				for _, np := range fam.variants {
					fmt.Fprintf(&ss, variantMarker+variantPlayCol+scoreCol+countCol+pctCol+"\n", np.play,
						strconv.Itoa(np.score), strconv.Itoa(np.count), pct(np.count))
				}
			}
			continue
		}

		// The needed-draw table only covers tile plays; exchanges and passes
		// are left out of it, although they still count towards the totals.
		if !fam.isTilePlay() {
			continue
		}
		for _, np := range fam.variants {
			np.ifdraw = neededDraw(st, ourPlayLeave, np.play)
		}
		fmt.Fprintf(&ss, playCol+drawCol+scoreCol+countCol+pctCol+"\n", fam.name(), fam.drawStr(),
			fam.scoreStr(), strconv.Itoa(fam.count), pct(fam.count))
		if len(fam.variants) > 1 {
			for _, np := range fam.variants {
				fmt.Fprintf(&ss, variantMarker+variantPlayCol+drawCol+scoreCol+countCol+pctCol+"\n", np.play,
					np.ifdraw, strconv.Itoa(np.score), strconv.Itoa(np.count), pct(np.count))
			}
		}
	}
	if showBingos {
		fmt.Fprintf(&ss, "Bingo probability: %.2f%%\n", float64(bingos*100)/float64(totalPlayCount))
	}
	return ss.String()
}

func (st *SimStats) CalculatePlayStats(play string) (string, error) {
	var ss strings.Builder

	iters, err := st.simmer.ReadHeatmap()
	if err != nil {
		return "", err
	}
	log.Debug().Msgf("Read %d log lines", len(iters))
	normalizedPlay := Normalize(play)

	oppResponses := map[string]*nextPlay{}
	ourNextPlays := map[string]*nextPlay{}
	totalOppResponses := 0
	totalOurNextPlays := 0
	oppScores := []float64{}
	ourScores := []float64{}
	leave := ""
	for i := range iters {
		for j := range iters[i].Plays {
			if normalizedPlay != Normalize(iters[i].Plays[j].Play) {
				continue
			}
			leave = iters[i].Plays[j].Leave
			if len(iters[i].Plays[j].Plies) > 0 {
				nextPlay := iters[i].Plays[j].Plies[0]
				addNextPlay(nextPlay.Play, nextPlay.Pts, nextPlay.Bingo, oppResponses)
				oppScores = append(oppScores, float64(nextPlay.Pts))
				totalOppResponses++
			}
			if len(iters[i].Plays[j].Plies) > 1 {
				nextPlay := iters[i].Plays[j].Plies[1]
				addNextPlay(nextPlay.Play, nextPlay.Pts, nextPlay.Bingo, ourNextPlays)
				ourScores = append(ourScores, float64(nextPlay.Pts))
				totalOurNextPlays++
			}
		}
	}

	oppResponsesList := sortedFamiliesByScore(oppResponses)

	ss.WriteString(playStatsStr(st, leave, oppResponsesList, "### Opponent's highest scoring plays", 10, totalOppResponses, false, false))
	ss.WriteString("\n\n")

	maxToDisplay := 15

	oppResponsesList = sortedFamilies(oppResponses)
	ourNextPlaysList := sortedFamilies(ourNextPlays)

	ss.WriteString(playStatsStr(st, leave, oppResponsesList, "### Opponent's next play", maxToDisplay, totalOppResponses, true, false))
	ss.WriteString("\n")

	ss.WriteString(playStatsStr(st, leave, ourNextPlaysList, "### Our follow-up play", maxToDisplay, totalOurNextPlays, true, true))
	ss.WriteString("\n")

	st.oppHist = histogram.Hist(15, oppScores)
	st.ourHist = histogram.Hist(15, ourScores)

	return ss.String(), nil
}

func (st *SimStats) LastHistogram() (histogram.Histogram, histogram.Histogram) {
	return st.oppHist, st.ourHist
}

func (st *SimStats) CalculateTileStats() (string, error) {
	var ss strings.Builder

	iters, err := st.simmer.ReadHeatmap()
	if err != nil {
		return "", err
	}

	oppTileRawCount := map[tilemapping.MachineLetter]int{}
	oppTileAtLeastCount := map[tilemapping.MachineLetter]int{}

	numRacks := 0

	for i := range iters {
		for j := range iters[i].Plays {
			if len(iters[i].Plays[j].Plies) > 0 {
				nextPlay := iters[i].Plays[j].Plies[0]
				rack := nextPlay.Rack
				numRacks++

				mls, err := tilemapping.ToMachineLetters(rack, st.game.Alphabet())
				if err != nil {
					return "", err
				}
				atLeast := map[tilemapping.MachineLetter]bool{}
				for _, ml := range mls {
					oppTileRawCount[ml]++
					if _, ok := atLeast[ml]; !ok {
						oppTileAtLeastCount[ml]++
						atLeast[ml] = true
					}
				}
			}
		}
	}
	fmt.Fprintf(&ss, "Note: these tile counts are obtained by simulation and thus are not mathematically exact.\n")
	fmt.Fprintf(&ss, "  Tile         %% chance in opp rack   Expected # in opp rack\n")

	for ml := range tilemapping.MaxAlphabetSize {
		if ct, ok := oppTileAtLeastCount[tilemapping.MachineLetter(ml)]; ok {
			fmt.Fprintf(&ss, "  %s            %.2f%%                 %.2f\n",
				tilemapping.MachineLetter(ml).UserVisible(st.game.Alphabet(), false),
				float64(ct*100)/float64(numRacks),
				float64(oppTileRawCount[tilemapping.MachineLetter(ml)])/float64(numRacks),
			)
		}
	}

	return ss.String(), nil
}
