package automatic

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/domino14/macondo/cache"
	"github.com/domino14/macondo/montecarlo"
)

// AnalyzeLogFile analyzes the given game CSV file and spits out a bunch of
// statistics.
func AnalyzeLogFile(filepath string) (string, error) {
	file, err := cache.Open(filepath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	r := csv.NewReader(file)

	// Record looks like:
	// gameID,p1score,p2score

	player1scores := &montecarlo.Statistic{}
	player2scores := &montecarlo.Statistic{}
	player1bingos := &montecarlo.Statistic{}
	player2bingos := &montecarlo.Statistic{}
	player1ppt := &montecarlo.Statistic{}
	player2ppt := &montecarlo.Statistic{}

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

	// build stats string
	stats := fmt.Sprintf("Games played: %d\n", gamesPlayed)
	stats += fmt.Sprintf("%v wins: %.1f (%.3f%%)\n", p1Name, p1wl, 100.0*p1wl/float64(gamesPlayed))
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
		p1Name, player2ppt.Mean(), player2ppt.Stdev())

	stats += fmt.Sprintf("%v went first: %.1f (%.3f%%)\n", p1Name, p1first, 100.0*p1first/float64(gamesPlayed))
	stats += fmt.Sprintf("Player who went first wins: %.1f (%.3f%%)\n",
		wentFirstWL, 100.0*wentFirstWL/float64(gamesPlayed))
	return stats, nil
}

func AnalyzeMoveLogFile(filepath string) (string, error) {
	file, err := cache.Open(filepath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	r := csv.NewReader(file)

	averageByScore := map[int]*montecarlo.Statistic{}
	averageByScoreAndTilesRemaining := map[int]*montecarlo.Statistic{}
	movesPlayed := 0
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		if record[0] == "playerID" {
			// this is the header line
			continue
		}
		score, err := strconv.Atoi(record[5])
		if err != nil {
			panic(err)
		}
		tilesPlayed, err := strconv.Atoi(record[7])
		if err != nil {
			panic(err)
		}
		equity, err := strconv.ParseFloat(record[9], 64)
		if err != nil {
			panic(err)
		}
		leaveEquity := equity - float64(score)
		averageByScoreStat := averageByScore[score]
		if averageByScoreStat == nil {
			averageByScoreStat = &montecarlo.Statistic{}
			averageByScore[score] = averageByScoreStat
		}
		idx := score*8 + (7 - tilesPlayed)
		averageByScoreAndTilesRemainingStat := averageByScoreAndTilesRemaining[idx]
		if averageByScoreAndTilesRemainingStat == nil {
			averageByScoreAndTilesRemainingStat = &montecarlo.Statistic{}
			averageByScoreAndTilesRemaining[idx] = averageByScoreAndTilesRemainingStat
		}
		averageByScoreStat.Push(leaveEquity)
		averageByScoreAndTilesRemainingStat.Push(leaveEquity)
		movesPlayed++
	}

	// build stats string
	var statsString strings.Builder
	fmt.Fprintf(&statsString, "Moves analyzed: %d\nAverage Leave Equity by Score (score, n, average leave equity, stdev):\n", movesPlayed)
	for i := 0; i < 3000; i++ {
		stat := averageByScore[i]
		if stat != nil {
			fmt.Fprintf(&statsString, "%d, %d, %.2f, %.2f\n", i, stat.Total(), stat.Mean(), stat.Stdev())
		}
	}
	fmt.Fprint(&statsString, "\nAverage Leave Equity by Score and Tiles Remaining (score, tiles remaining, n, average leave equity, stdev):\n")
	for i := 0; i < 24000; i++ {
		stat := averageByScoreAndTilesRemaining[i]
		if stat != nil {
			fmt.Fprintf(&statsString, "%d, %d, %d, %.2f, %.2f\n", i/8, i%8, stat.Total(), stat.Mean(), stat.Stdev())
		}
	}
	return statsString.String(), nil
}
