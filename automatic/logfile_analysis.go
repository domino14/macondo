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

		result.GamesPlayed++
	}

	result.WinPValue = stats.BinomialZTestPValue(result.P1Wins, float64(result.GamesPlayed))
	result.ScorePValue = stats.PairedZTestPValue(
		result.ScoreDiff.Mean(), result.ScoreDiff.Stdev(), result.GamesPlayed)

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

	// Header line
	s := fmt.Sprintf("Games: %d    %s wins: %.2f%% ± %.2f%%    p = %.4f\n",
		r.GamesPlayed,
		p1name,
		100.0*r.P1Wins/n,
		cimargin*100.0,
		r.WinPValue,
	)
	s += "\n"

	// Column headers
	s += fmt.Sprintf("  %-18s  %-20s  %-20s\n", "", p1display, p2display)

	// Score row (with p-value)
	s += fmt.Sprintf("  %-18s  %6.2f ± %-11.2f  %6.2f ± %-11.2f  (p = %.4f)\n",
		"Mean Score",
		r.Player1.Score.Mean(), r.Player1.Score.Stdev(),
		r.Player2.Score.Mean(), r.Player2.Score.Stdev(),
		r.ScorePValue,
	)

	// Bingos row
	s += fmt.Sprintf("  %-18s  %6.2f ± %-11.2f  %6.2f ± %-11.2f\n",
		"Mean Bingos",
		r.Player1.Bingos.Mean(), r.Player1.Bingos.Stdev(),
		r.Player2.Bingos.Mean(), r.Player2.Bingos.Stdev(),
	)

	// PPT row
	s += fmt.Sprintf("  %-18s  %6.2f ± %-11.2f  %6.2f ± %-11.2f\n",
		"Mean PPT",
		r.Player1.PPT.Mean(), r.Player1.PPT.Stdev(),
		r.Player2.PPT.Mean(), r.Player2.PPT.Stdev(),
	)

	s += "\n"

	// Went first
	s += fmt.Sprintf("  %-18s  %-20s  %-20s\n",
		"Went first",
		fmt.Sprintf("%.2f%%", 100.0*r.P1First/n),
		fmt.Sprintf("%.2f%%", 100.0*(n-r.P1First)/n),
	)

	s += fmt.Sprintf("  First player wins: %.2f%%\n",
		100.0*r.WentFirstWins/n,
	)

	return s
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
