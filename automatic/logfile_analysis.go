package automatic

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/domino14/macondo/montecarlo"
)

// AnalyzeLogFile analyzes the given game CSV file and spits out a bunch of
// statistics.
func AnalyzeLogFile(filepath string) (string, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	r := csv.NewReader(file)

	// Record looks like:
	// gameID,p1score,p2score

	player1stats := &montecarlo.Statistic{}
	player2stats := &montecarlo.Statistic{}

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
		player1stats.Push(float64(p1score))
		player2stats.Push(float64(p2score))
		if p1score > p2score {
			p1wl += 1.0
			if record[3] == p1Name {
				wentFirstWL += 1.0
			}
		} else if p1score == p2score {
			p1wl += 0.5
			wentFirstWL += 0.5
		} else if p1score < p2score {
			if record[3] == p2Name {
				wentFirstWL += 1.0
			}
		}
		if record[3] == p1Name {
			p1first++
		}

		gamesPlayed++
	}

	// build stats string
	stats := fmt.Sprintf("Games played: %d\n", gamesPlayed)
	stats += fmt.Sprintf("%v wins: %.1f (%.3f%%)\n", p1Name, p1wl, 100.0*p1wl/float64(gamesPlayed))
	stats += fmt.Sprintf("%v went first: %.1f (%.3f%%)\n", p1Name, p1first, 100.0*p1first/float64(gamesPlayed))
	stats += fmt.Sprintf("Player who went first wins: %.1f (%.3f%%)\n",
		wentFirstWL, 100.0*wentFirstWL/float64(gamesPlayed))
	stats += fmt.Sprintf("%v Mean Score: %.6f  Stdev: %.6f\n",
		p1Name, player1stats.Mean(), player1stats.Stdev())
	stats += fmt.Sprintf("%v Mean Score: %.6f  Stdev: %.6f\n",
		p2Name, player2stats.Mean(), player2stats.Stdev())

	return stats, nil
}
