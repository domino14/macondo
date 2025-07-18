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

// AnalyzeLogFile analyzes the given game CSV file and spits out a bunch of
// statistics.
func AnalyzeLogFile(filepath string) (string, error) {
	file, _, err := cache.Open(filepath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	r := csv.NewReader(file)

	// Record looks like:
	// gameID,p1score,p2score

	player1scores := &stats.Statistic{}
	player2scores := &stats.Statistic{}
	player1bingos := &stats.Statistic{}
	player2bingos := &stats.Statistic{}
	player1ppt := &stats.Statistic{}
	player2ppt := &stats.Statistic{}

	p1wl := float64(0)
	p1first := float64(0)
	wentFirstWL := float64(0)
	gamesPlayed := 0
	var p1Name, p2Name string
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		if record[0] == "gameID" {
			// this is the header line
			p1Name = strings.Split(record[1], "_")[0]
			p2Name = strings.Split(record[2], "_")[0]
			continue
		}
		p1score, err := strconv.Atoi(record[1])
		if err != nil {
			return "", err
		}
		p2score, err := strconv.Atoi(record[2])
		if err != nil {
			return "", err
		}

		p1bingos, err := strconv.Atoi(record[3])
		if err != nil {
			return "", err
		}
		p2bingos, err := strconv.Atoi(record[4])
		if err != nil {
			return "", err
		}

		p1turns, err := strconv.Atoi(record[5])
		if err != nil {
			return "", err
		}

		p2turns, err := strconv.Atoi(record[6])
		if err != nil {
			return "", err
		}

		player1scores.Push(float64(p1score))
		player2scores.Push(float64(p2score))
		player1bingos.Push(float64(p1bingos))
		player2bingos.Push(float64(p2bingos))
		player1ppt.Push(float64(p1score) / float64(p1turns))
		player2ppt.Push(float64(p2score) / float64(p2turns))

		if p1score > p2score {
			p1wl += 1.0
			if record[7] == p1Name {
				wentFirstWL += 1.0
			}
		} else if p1score == p2score {
			p1wl += 0.5
			wentFirstWL += 0.5
		} else if p1score < p2score {
			if record[7] == p2Name {
				wentFirstWL += 1.0
			}
		}
		if record[7] == p1Name {
			p1first++
		}

		gamesPlayed++
	}

	_, cimargin := confidenceInterval(int(p1wl), gamesPlayed, stats.Z95)

	// build stats string
	stats := fmt.Sprintf("Games played: %d\n", gamesPlayed)
	stats += fmt.Sprintf("%v wins: %.1f (%.3f%% +/- %.3f)\n", p1Name, p1wl, 100.0*p1wl/float64(gamesPlayed), cimargin*100.0)
	stats += fmt.Sprintf("%v Mean Score: %.4f  Stdev: %.4f\n",
		p1Name, player1scores.Mean(), player1scores.Stdev())
	stats += fmt.Sprintf("%v Mean Score: %.4f  Stdev: %.4f\n",
		p2Name, player2scores.Mean(), player2scores.Stdev())
	stats += fmt.Sprintf("%v Mean Bingos: %.4f  Stdev: %.4f\n",
		p1Name, player1bingos.Mean(), player1bingos.Stdev())
	stats += fmt.Sprintf("%v Mean Bingos: %.4f  Stdev: %.4f\n",
		p2Name, player2bingos.Mean(), player2bingos.Stdev())
	stats += fmt.Sprintf("%v Mean Points Per Turn: %.4f  Stdev: %.4f\n",
		p1Name, player1ppt.Mean(), player1ppt.Stdev())
	stats += fmt.Sprintf("%v Mean Points Per Turn: %.4f  Stdev: %.4f\n",
		p2Name, player2ppt.Mean(), player2ppt.Stdev())

	stats += fmt.Sprintf("%v went first: %.1f (%.3f%%)\n", p1Name, p1first, 100.0*p1first/float64(gamesPlayed))
	stats += fmt.Sprintf("Player who went first wins: %.1f (%.3f%%)\n",
		wentFirstWL, 100.0*wentFirstWL/float64(gamesPlayed))
	return stats, nil
}

func gameLinesFromLogFile(cfg *config.Config, filename, gid string) ([][]string, error) {
	useGzip := strings.HasSuffix(filename, ".gz")
	var file io.ReadCloser
	if useGzip {
		f, _, err := cache.Open(filename)
		if err != nil {
			return nil, err
		}
		gz, err := gzip.NewReader(f)
		if err != nil {
			f.Close()
			return nil, err
		}
		file = struct {
			io.Reader
			io.Closer
		}{gz, f}
	} else {
		var err error
		file, _, err = cache.Open(filename)
		if err != nil {
			return nil, err
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
			return nil, err
		}
		if record[0] == "gameID" {
			// this is the header line
			continue
		}
		if record[0] != gid {
			continue
		}
		gameLines = append(gameLines, record)
	}
	if len(gameLines) == 0 {
		return nil, errors.New("gameID not found in log file")
	}
	return gameLines, nil
}

func gameHistoryFromGameLines(cfg *config.Config, gameLines [][]string, letterdist,
	lexicon, boardlayout string) (*pb.GameHistory, error) {

	if letterdist == "" {
		letterdist = "english"
	}
	if boardlayout == "" {
		boardlayout = board.CrosswordGameLayout
	}
	if lexicon == "" {
		lexicon = "CSW21"
	}

	rules, err := game.NewBasicGameRules(cfg, lexicon, boardlayout,
		letterdist, game.CrossScoreOnly, game.VarClassic)
	if err != nil {
		return nil, err
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
		return nil, err
	}
	g.StartGame()
	for _, row := range gameLines {
		pidx := 0
		if g.History().Players[1].Nickname == row[0] {
			pidx = 1
		}
		err = g.SetRackFor(pidx, tilemapping.RackFromString(row[3], g.Alphabet()))
		if err != nil {
			return nil, err
		}
		if strings.HasPrefix(row[4], "(exch") {
			cmd := strings.Split(row[4], " ")
			exchanged := strings.TrimSuffix(cmd[1], ")")
			m, err := g.NewExchangeMove(pidx, exchanged)
			if err != nil {
				return nil, err
			}
			err = g.PlayMove(m, true, 0)
			if err != nil {
				return nil, err
			}
		} else if row[4] == "(Pass)" {
			m, err := g.NewPassMove(pidx)
			if err != nil {
				return nil, err
			}
			err = g.PlayMove(m, true, 0)
			if err != nil {
				return nil, err
			}
		} else {
			play := strings.Split(strings.TrimSpace(row[4]), " ")
			m, err := g.NewPlacementMove(pidx, play[0], play[1], false)
			if err != nil {
				return nil, err
			}
			err = g.PlayMove(m, true, 0)
			if err != nil {
				return nil, err
			}
		}
	}
	return g.History(), nil
}

func ExportGCG(cfg *config.Config, filename, letterdist, lexicon, boardlayout, gid string,
	out io.Writer) error {

	gameLines, err := gameLinesFromLogFile(cfg, filename, gid)
	if err != nil {
		return err
	}

	gh, err := gameHistoryFromGameLines(cfg, gameLines, letterdist, lexicon, boardlayout)
	if err != nil {
		return err
	}

	contents, err := gcgio.GameHistoryToGCG(gh, true)
	if err != nil {
		return err
	}
	_, err = out.Write([]byte(contents))
	return err
}
