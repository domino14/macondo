package automatic

import (
	"compress/gzip"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/stats"
	"github.com/domino14/macondo/turnplayer"
)

func confidenceInterval(wins, total int, confidenceZ float64) (float64, float64) {
	if total == 0 {
		return 0, 0
	}
	p := float64(wins) / float64(total)
	se := math.Sqrt(p * (1 - p) / float64(total))
	margin := confidenceZ * se
	return p, margin
}

// PlayerStats holds per-player statistics accumulators.
type PlayerStats struct {
	Name   string
	Score  *stats.Statistic
	Bingos *stats.Statistic
	PPT    *stats.Statistic
}

// AnalysisResult holds the full results of analyzing a log file.
type AnalysisResult struct {
	GamesPlayed   int
	Player1       PlayerStats
	Player2       PlayerStats
	P1Wins        float64
	P1First       float64
	WentFirstWins float64
	WinPValue     float64 // two-sided binomial z-test, H0: win rate = 0.5
	ScorePValue   float64 // two-sided paired z-test on per-game score diff
	ScoreDiff     *stats.Statistic

	// Paired is filled in for a -gamepairs run, where the two games of a pair
	// share a bag and cannot be treated as independent.
	Paired *PairedResult
}

// PairedResult analyzes a game-pair run with the pair, rather than the game, as
// the unit of observation.
//
// The two games of a pair are played from one bag with the seats swapped, so
// good tiles for one bot in the first game become good tiles for the other bot
// in the second. The two results are strongly anti-correlated by construction,
// and treating them as independent games throws away the variance reduction the
// pairing was run to get. Adding a pair's two margins together cancels the
// tiles instead, and cancels the first-move advantage with them, since each bot
// moves first exactly once per pair.
type PairedResult struct {
	Pairs int
	// Incomplete counts pairs that did not have both halves in the log -- a run
	// stopped midway, say. They are left out of everything below.
	Incomplete int
	// Divergent counts pairs whose halves ever chose differently, straight from
	// the log column. Pairs that never diverge contribute an exact zero, so this
	// is the count that carries any information.
	Divergent int
	// ExactTies counts pairs whose two margins cancel to zero.
	ExactTies int

	// Margin is bot 1's score margin summed over a pair's two games, one
	// observation per pair. Halve it for a per-game figure.
	Margin *stats.Statistic
	// WinShare is bot 1's share of a pair's two games: 1 for winning both, 0.5
	// for a split, 0 for losing both. The null value is 0.5.
	WinShare *stats.Statistic

	// Correlation is between the two halves' margins. The pairing is working
	// when this is well below zero.
	Correlation float64

	// MarginPValue and WinPValue come from a sign-flip randomization test over
	// the pairs, which assumes only the symmetry the design guarantees.
	MarginPValue float64
	WinPValue    float64
	// ExactP is true when every sign assignment was enumerated rather than
	// sampled.
	ExactP bool
	// MarginTPValue is the same test done by t-test, for comparison.
	MarginTPValue float64

	// NaiveScoreSE is the standard error the per-game analysis reports for the
	// same quantity, kept to show what the pairing bought.
	NaiveScoreSE float64
}

// MarginPerGame is the average score margin per game, with its 95% confidence
// half-width.
func (p *PairedResult) MarginPerGame() (float64, float64) {
	if p.Pairs < 2 {
		return p.Margin.Mean() / 2, math.NaN()
	}
	t := stats.TCriticalValue(0.95, p.Pairs)
	return p.Margin.Mean() / 2, t * p.Margin.StandardError() / 2
}

// PairedSE is the standard error of the per-game margin.
func (p *PairedResult) PairedSE() float64 {
	return p.Margin.StandardError() / 2
}

// pairHalf is one game of a pair, as the per-game log records it.
type pairHalf struct {
	margin    float64
	winShare  float64
	divergent bool
}

// buildPairedResult turns the two halves of each pair into one observation and
// tests those, which is the analysis a paired design calls for.
func buildPairedResult(halves map[int][]pairHalf, order []int, naiveSE float64) *PairedResult {
	p := &PairedResult{
		Margin:       &stats.Statistic{},
		WinShare:     &stats.Statistic{},
		NaiveScoreSE: naiveSE,
		Correlation:  math.NaN(),
		MarginPValue: 1.0,
		WinPValue:    1.0,
		ExactP:       true,
	}

	margins := make([]float64, 0, len(order))
	winDiffs := make([]float64, 0, len(order))
	// The two halves separately, to measure how hard the pairing is working.
	var firstHalves, secondHalves []float64

	for _, idx := range order {
		h := halves[idx]
		if len(h) != 2 {
			// Half a pair says nothing on its own, and counting it as a whole
			// one would read a single game's luck as a result.
			p.Incomplete++
			continue
		}
		p.Pairs++
		if h[0].divergent || h[1].divergent {
			p.Divergent++
		}
		margin := h[0].margin + h[1].margin
		if margin == 0 {
			p.ExactTies++
		}
		// Every pair counts, including the ones that cancel to zero: those are
		// observations of no difference, and leaving them out would inflate the
		// average.
		p.Margin.Push(margin)
		margins = append(margins, margin)

		winShare := (h[0].winShare + h[1].winShare) / 2
		p.WinShare.Push(winShare)
		winDiffs = append(winDiffs, winShare-0.5)

		firstHalves = append(firstHalves, h[0].margin)
		secondHalves = append(secondHalves, h[1].margin)
	}

	if p.Pairs == 0 {
		return p
	}

	p.Correlation = correlation(firstHalves, secondHalves)
	p.MarginPValue, p.ExactP = stats.SignFlipPValue(margins, 0)
	p.WinPValue, _ = stats.SignFlipPValue(winDiffs, 0)
	p.MarginTPValue = stats.PairedTTestPValue(
		p.Margin.Mean(), p.Margin.Stdev(), p.Pairs)
	return p
}

// correlation returns Pearson's r, or NaN when either side never varies.
func correlation(xs, ys []float64) float64 {
	if len(xs) != len(ys) || len(xs) < 2 {
		return math.NaN()
	}
	n := float64(len(xs))
	var sx, sy float64
	for i := range xs {
		sx += xs[i]
		sy += ys[i]
	}
	mx, my := sx/n, sy/n
	var sxy, sxx, syy float64
	for i := range xs {
		dx, dy := xs[i]-mx, ys[i]-my
		sxy += dx * dy
		sxx += dx * dx
		syy += dy * dy
	}
	if sxx == 0 || syy == 0 {
		return math.NaN()
	}
	return sxy / math.Sqrt(sxx*syy)
}

// AnalyzeLogFileData analyzes the given game CSV file and returns structured results.
func AnalyzeLogFileData(filepath string) (*AnalysisResult, error) {
	file, _, err := cache.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	r := csv.NewReader(file)

	result := &AnalysisResult{
		Player1: PlayerStats{
			Score:  &stats.Statistic{},
			Bingos: &stats.Statistic{},
			PPT:    &stats.Statistic{},
		},
		Player2: PlayerStats{
			Score:  &stats.Statistic{},
			Bingos: &stats.Statistic{},
			PPT:    &stats.Statistic{},
		},
		ScoreDiff: &stats.Statistic{},
	}

	// Per-pair halves, gathered only if the log turns out to be a paired run.
	pairHalves := map[int][]pairHalf{}
	pairOrder := []int{}
	paired := false

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if record[0] == "gameID" {
			result.Player1.Name = strings.Split(record[1], "_")[0]
			result.Player2.Name = strings.Split(record[2], "_")[0]
			paired = len(record) >= 10 && record[8] == "pair" && record[9] == "divergent"
			continue
		}
		if record[0] == "playerID" {
			return nil, fmt.Errorf("this looks like a per-turn log; autoanalyze expects the game summary log (e.g. games-*.txt)")
		}
		p1score, err := strconv.Atoi(record[1])
		if err != nil {
			return nil, err
		}
		p2score, err := strconv.Atoi(record[2])
		if err != nil {
			return nil, err
		}
		p1bingos, err := strconv.Atoi(record[3])
		if err != nil {
			return nil, err
		}
		p2bingos, err := strconv.Atoi(record[4])
		if err != nil {
			return nil, err
		}
		p1turns, err := strconv.Atoi(record[5])
		if err != nil {
			return nil, err
		}
		p2turns, err := strconv.Atoi(record[6])
		if err != nil {
			return nil, err
		}

		result.Player1.Score.Push(float64(p1score))
		result.Player2.Score.Push(float64(p2score))
		result.Player1.Bingos.Push(float64(p1bingos))
		result.Player2.Bingos.Push(float64(p2bingos))
		result.Player1.PPT.Push(float64(p1score) / float64(p1turns))
		result.Player2.PPT.Push(float64(p2score) / float64(p2turns))
		result.ScoreDiff.Push(float64(p1score - p2score))

		if p1score > p2score {
			result.P1Wins += 1.0
			if record[7] == result.Player1.Name {
				result.WentFirstWins += 1.0
			}
		} else if p1score == p2score {
			result.P1Wins += 0.5
			result.WentFirstWins += 0.5
		} else {
			if record[7] == result.Player2.Name {
				result.WentFirstWins += 1.0
			}
		}
		if record[7] == result.Player1.Name {
			result.P1First++
		}

		if paired && len(record) >= 10 {
			pairIdx, err := strconv.Atoi(record[8])
			if err != nil {
				return nil, fmt.Errorf("bad pair index %q: %w", record[8], err)
			}
			win := 0.0
			switch {
			case p1score > p2score:
				win = 1.0
			case p1score == p2score:
				win = 0.5
			}
			if _, seen := pairHalves[pairIdx]; !seen {
				pairOrder = append(pairOrder, pairIdx)
			}
			pairHalves[pairIdx] = append(pairHalves[pairIdx], pairHalf{
				margin:    float64(p1score - p2score),
				winShare:  win,
				divergent: record[9] == "true",
			})
		}

		result.GamesPlayed++
	}

	result.WinPValue = stats.BinomialZTestPValue(result.P1Wins, float64(result.GamesPlayed))
	result.ScorePValue = stats.PairedZTestPValue(
		result.ScoreDiff.Mean(), result.ScoreDiff.Stdev(), result.GamesPlayed)

	if paired {
		result.Paired = buildPairedResult(pairHalves, pairOrder,
			result.ScoreDiff.StandardError())
	}

	return result, nil
}

// FormatTable formats analysis results as a side-by-side comparison table.
func FormatTable(r *AnalysisResult) string {
	n := float64(r.GamesPlayed)
	_, cimargin := confidenceInterval(int(r.P1Wins), r.GamesPlayed, stats.Z95)

	p1name := r.Player1.Name
	p2name := r.Player2.Name

	// Truncate long names to fit in table columns
	const colWidth = 18
	p1display := truncate(p1name, colWidth)
	p2display := truncate(p2name, colWidth)

	p2wins := n - r.P1Wins

	// Header line
	s := fmt.Sprintf("Games: %d    %s wins: %.3f%% ± %.3f%%    p = %.2e\n",
		r.GamesPlayed,
		p1name,
		100.0*r.P1Wins/n,
		cimargin*100.0,
		r.WinPValue,
	)
	s += "\n"

	// Column headers
	s += fmt.Sprintf("  %-18s  %-22s  %-22s\n", "", p1display, p2display)

	// Win counts row
	s += fmt.Sprintf("  %-18s  %-22s  %-22s\n",
		"Wins",
		fmt.Sprintf("%.1f (%.3f%%)", r.P1Wins, 100.0*r.P1Wins/n),
		fmt.Sprintf("%.1f (%.3f%%)", p2wins, 100.0*p2wins/n),
	)

	ci95 := stats.Z95
	// Score row (with p-value)
	s += fmt.Sprintf("  %-18s  %7.3f ± %-12.3f  %7.3f ± %-12.3f  (p = %.2e)\n",
		"Mean Score",
		r.Player1.Score.Mean(), ci95*r.Player1.Score.StandardError(),
		r.Player2.Score.Mean(), ci95*r.Player2.Score.StandardError(),
		r.ScorePValue,
	)

	// Bingos row
	s += fmt.Sprintf("  %-18s  %7.3f ± %-12.3f  %7.3f ± %-12.3f\n",
		"Mean Bingos",
		r.Player1.Bingos.Mean(), ci95*r.Player1.Bingos.StandardError(),
		r.Player2.Bingos.Mean(), ci95*r.Player2.Bingos.StandardError(),
	)

	// PPT row
	s += fmt.Sprintf("  %-18s  %7.3f ± %-12.3f  %7.3f ± %-12.3f\n",
		"Mean PPT",
		r.Player1.PPT.Mean(), ci95*r.Player1.PPT.StandardError(),
		r.Player2.PPT.Mean(), ci95*r.Player2.PPT.StandardError(),
	)

	s += "\n"

	// Went first
	s += fmt.Sprintf("  %-18s  %-22s  %-22s\n",
		"Went first",
		fmt.Sprintf("%.3f%%", 100.0*r.P1First/n),
		fmt.Sprintf("%.3f%%", 100.0*(n-r.P1First)/n),
	)

	s += fmt.Sprintf("  First player wins: %.3f%%\n",
		100.0*r.WentFirstWins/n,
	)

	s += "\n  (± values are 95% confidence intervals on the mean)\n"

	if r.Paired != nil {
		s += formatPaired(r.Paired, p1name, p2name)
	}

	return s
}

// formatPaired renders the pair-level analysis, which is the one to read for a
// game-pair run.
func formatPaired(p *PairedResult, p1name, p2name string) string {
	if p.Pairs == 0 {
		return "\nGame pairs: no complete pair in this log.\n"
	}

	var b strings.Builder
	b.WriteString("\n  The figures above count each game on its own. For a paired run that\n" +
		"  overstates the uncertainty, because the two games of a pair share a bag.\n")
	b.WriteString("\nGame pairs\n")
	fmt.Fprintf(&b, "  %d pairs (%d games), %d divergent, %d exact ties\n",
		p.Pairs, 2*p.Pairs, p.Divergent, p.ExactTies)
	if p.Incomplete > 0 {
		fmt.Fprintf(&b, "  %d incomplete pair(s) left out\n", p.Incomplete)
	}

	margin, ci := p.MarginPerGame()
	leader, trailer := p1name, p2name
	if margin < 0 {
		leader, trailer = p2name, p1name
	}
	fmt.Fprintf(&b, "\n  %-22s %+.2f ± %.2f points per game\n",
		leader+" over "+trailer, math.Abs(margin), ci)
	fmt.Fprintf(&b, "  %-22s %.3f%% (0.5 = even)\n",
		p1name+" win share", 100*p.WinShare.Mean())

	kind := "exact"
	if !p.ExactP {
		kind = "sampled"
	}
	fmt.Fprintf(&b, "\n  score p = %.2e, win p = %.2e   (%s sign-flip test over %d pairs)\n",
		p.MarginPValue, p.WinPValue, kind, p.Pairs)
	fmt.Fprintf(&b, "  score p = %.2e by t-test, for comparison\n", p.MarginTPValue)

	if !math.IsNaN(p.Correlation) {
		fmt.Fprintf(&b, "\n  The two halves of a pair correlate at %+.3f: ", p.Correlation)
		switch {
		case p.Correlation < -0.99:
			b.WriteString("they mirror each other\n  exactly, which is what a bot playing itself does.\n")
		case p.Correlation < -0.05:
			b.WriteString("tile luck largely cancels\n  inside a pair, which is what the pairing is for.\n")
		default:
			b.WriteString("barely negative, so the\n  pairing is not cancelling much luck here.\n")
		}
	}
	if p.NaiveScoreSE > 0 {
		pairedSE := p.PairedSE()
		if pairedSE == 0 {
			fmt.Fprintf(&b,
				"  Every pair cancelled exactly, so the margin is known to be %+.2f with\n"+
					"  no uncertainty at all. Read per game, the same data would have\n"+
					"  claimed a standard error of %.2f.\n", margin, p.NaiveScoreSE)
		} else {
			fmt.Fprintf(&b,
				"  Standard error %.2f per game, against %.2f if the games were counted\n"+
					"  separately: pairing is worth about %.1fx the games here.\n",
				pairedSE, p.NaiveScoreSE, math.Pow(p.NaiveScoreSE/pairedSE, 2))
		}
	}
	if p.Divergent == 0 {
		b.WriteString("\n  No pair diverged, so the bots never once chose differently and this\n" +
			"  run cannot separate them. Expected when a bot plays itself.\n")
	} else if p.Divergent < p.Pairs/4 {
		fmt.Fprintf(&b, "\n  Only %d of %d pairs diverged, so the comparison rests on those.\n",
			p.Divergent, p.Pairs)
	}
	return b.String()
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-1] + "…"
}

// AnalyzeLogFile analyzes the given game CSV file and returns formatted stats.
// Kept for backward compatibility.
func AnalyzeLogFile(filepath string) (string, error) {
	result, err := AnalyzeLogFileData(filepath)
	if err != nil {
		return "", err
	}
	return FormatTable(result), nil
}

func ExportGCG(cfg *config.Config, filename, letterdist, lexicon, boardlayout, gid string,
	out io.Writer) error {
	if letterdist == "" {
		letterdist = "english"
	}
	if boardlayout == "" {
		boardlayout = board.CrosswordGameLayout
	}
	if lexicon == "" {
		lexicon = "CSW21"
	}
	useGzip := strings.HasSuffix(filename, ".gz")
	var file io.ReadCloser
	if useGzip {
		f, _, err := cache.Open(filename)
		if err != nil {
			return err
		}
		gz, err := gzip.NewReader(f)
		if err != nil {
			f.Close()
			return err
		}
		file = struct {
			io.Reader
			io.Closer
		}{gz, f}
	} else {
		var err error
		file, _, err = cache.Open(filename)
		if err != nil {
			return err
		}
	}

	defer file.Close()
	r := csv.NewReader(file)

	gameLines := [][]string{}
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if record[1] == "gameID" {
			// this is the header line
			continue
		}
		if record[1] != gid {
			continue
		}
		gameLines = append(gameLines, record)
	}
	if len(gameLines) == 0 {
		return errors.New("gameID not found in log file")
	}

	rules, err := game.NewBasicGameRules(cfg, lexicon, boardlayout,
		letterdist, game.CrossScoreOnly, game.VarClassic)
	if err != nil {
		return err
	}
	players := []*pb.PlayerInfo{
		{Nickname: gameLines[0][0], RealName: gameLines[0][0]},
		{Nickname: gameLines[1][0], RealName: gameLines[1][0]},
	}

	g, err := turnplayer.BaseTurnPlayerFromRules(&turnplayer.GameOptions{
		BoardLayoutName: boardlayout,
		Variant:         game.VarClassic,
	}, players, rules)
	if err != nil {
		return err
	}
	g.StartGame()

	for _, row := range gameLines {
		pidx := 0
		if g.History().Players[1].Nickname == row[0] {
			pidx = 1
		}
		err = g.SetRackFor(pidx, tilemapping.RackFromString(row[3], g.Alphabet()))
		if err != nil {
			return err
		}
		if strings.HasPrefix(row[4], "(exch") {
			cmd := strings.Split(row[4], " ")
			exchanged := strings.TrimSuffix(cmd[1], ")")
			m, err := g.NewExchangeMove(pidx, exchanged)
			if err != nil {
				return err
			}
			err = g.PlayMove(m, true, 0)
			if err != nil {
				return err
			}
		} else if row[4] == "(Pass)" {
			m, err := g.NewPassMove(pidx)
			if err != nil {
				return err
			}
			err = g.PlayMove(m, true, 0)
			if err != nil {
				return err
			}
		} else {
			play := strings.Split(strings.TrimSpace(row[4]), " ")
			m, err := g.NewPlacementMove(pidx, play[0], play[1], false)
			if err != nil {
				return err
			}
			err = g.PlayMove(m, true, 0)
			if err != nil {
				return err
			}
		}
	}
	contents, err := gcgio.GameHistoryToGCG(g.History(), true)
	if err != nil {
		return err
	}
	_, err = out.Write([]byte(contents))
	return err
}
