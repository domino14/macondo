package automatic

import (
	"encoding/csv"
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/domino14/macondo/cache"
	"github.com/domino14/macondo/montecarlo"
)

type Key struct {
	array   []int
	atEnd   bool
	maxVals []int
}

var DefaultMinVal = -1

var MoveAnalysisDimensionIndexes = map[string]int{
	"turn":         2,
	"score":        5,
	"tiles_played": 7,
}

var MoveAnalysisDimensionOrder = map[string]int{
	"turn":         0,
	"score":        1,
	"tiles_played": 2,
}

var DefaultOutputPath = "analysis_data"

// AnalyzeLogFile analyzes the given game CSV file and spits out a bunch of
// statistics.
func AnalyzeLogFile(filename string) (string, error) {
	file, err := cache.Open(filename)
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

func AnalyzeMoveLogFile(filename string) (string, error) {
	file, err := cache.Open(filename)
	if err != nil {
		return "", err
	}
	defer file.Close()
	r := csv.NewReader(file)

	scores := map[string]*montecarlo.Statistic{}
	leaveEquities := map[string]*montecarlo.Statistic{}
	vals := make([]int, len(MoveAnalysisDimensionOrder))
	movesAnalyzed := 0
	maxScore := -1
	maxTurn := -1
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

		for dim, idx := range MoveAnalysisDimensionIndexes {
			colVal, err := strconv.Atoi(record[idx])
			if err != nil {
				return "", err
			}
			vals[MoveAnalysisDimensionOrder[dim]] = colVal
		}
		keys := CreateDimensionKeys(vals)

		score := vals[1]
		turn := vals[0]
		tilesPlayed := vals[2]
		if score == 0 && turn == 1 && tilesPlayed == 0 {
			fmt.Printf("moves: %d, line: %s\n", movesAnalyzed, strings.Join(record, ", "))
		}
		if score > maxScore {
			maxScore = score
		}
		if turn > maxTurn {
			maxTurn = turn
		}
		equity, err := strconv.ParseFloat(record[9], 64)
		if err != nil {
			return "", err
		}

		for _, key := range keys {
			scoreMC := scores[key]
			if scoreMC == nil {
				scoreMC = &montecarlo.Statistic{}
				scores[key] = scoreMC
			}
			scoreMC.Push(float64(score))

			leaveEquityMC := leaveEquities[key]
			if leaveEquityMC == nil {
				leaveEquityMC = &montecarlo.Statistic{}
				leaveEquities[key] = leaveEquityMC
			}
			leaveEquities[key].Push(equity - float64(score))
		}
		movesAnalyzed++
	}

	fileMask := newKey(len(MoveAnalysisDimensionOrder), make([]int, len(MoveAnalysisDimensionOrder)))
	dataKey := newKey(len(MoveAnalysisDimensionOrder), []int{maxTurn, maxScore, 7})
	files := []string{}
	for !fileMask.isAtEnd() {
		mask := fileMask.array
		var csvBuilder strings.Builder
		columns := []string{}
		for dim, ord := range MoveAnalysisDimensionOrder {
			if mask[ord] != DefaultMinVal {
				columns = append(columns, dim)
			}
		}
		if len(columns) != 0 {
			sort.Slice(columns, func(i, j int) bool {
				return MoveAnalysisDimensionOrder[columns[i]] < MoveAnalysisDimensionOrder[columns[j]]
			})
		}
		headerPart := strings.Join(columns, ",")
		csvFilename := strings.Join(columns, "_") + ".csv"
		if len(columns) == 0 {
			csvFilename = "all.csv"
		}

		if headerPart != "" {
			headerPart += ", "
		}

		_, err = fmt.Fprintf(&csvBuilder, "%s\n", headerPart+"total, score, score stdev, leave equity, leave equity stdev")
		if err != nil {
			return "", err
		}
		numMatches := 0
		numExists := 0
		dataKey.reset()
		for !dataKey.isAtEnd() {
			if dataKey.matchesMask(mask) {
				numMatches++
				rowKey := dataKey.toStringWithMask(mask)
				_, exists := scores[rowKey]
				if exists {
					numExists++
					total := scores[rowKey].Total()
					score := scores[rowKey].Mean()
					scoreStdev := scores[rowKey].Stdev()
					leaveEquity := leaveEquities[rowKey].Mean()
					leaveEquityStdev := leaveEquities[rowKey].Stdev()
					dims := dataKey.toStringOmitNull()
					if dims != "" {
						dims += ", "
					}
					rowString := fmt.Sprintf("%s%d,%.2f,%.2f,%.2f,%.2f\n", dims, total, score, scoreStdev, leaveEquity, leaveEquityStdev)
					//fmt.Print(rowString)
					_, err = fmt.Fprint(&csvBuilder, rowString)
					if err != nil {
						return "", err
					}
				}
			}
			dataKey.increment()
		}
		csvFilepath := filepath.Join(DefaultOutputPath, csvFilename)
		err = ioutil.WriteFile(csvFilepath, []byte(csvBuilder.String()), 0644)
		if err != nil {
			return "", err
		}
		files = append(files, fmt.Sprintf("%s | %d, %d, ", fileMask.toString(), numMatches, numExists)+csvFilepath)
		fileMask.increment()
	}
	return fmt.Sprintf("%d moves analyzed\n%d files written:\n%s", movesAnalyzed, len(files), strings.Join(files, "\n")), nil
}

func CreateDimensionKeys(dims []int) []string {
	// We don't care about the max values, we
	// just want the string
	key := newKeyFromArray(dims, []int{0, 0, 0})
	mask := newKey(len(dims), []int{0, 0, 0})
	keyStrings := []string{}
	for !mask.isAtEnd() {
		keyStrings = append(keyStrings, key.toStringWithMask(mask.array))
		mask.increment()
	}
	return keyStrings
}

func newKey(arrLength int, maxVals []int) *Key {
	key := &Key{array: make([]int, arrLength), atEnd: false, maxVals: maxVals}
	key.reset()
	return key
}

func newKeyFromArray(arr []int, maxVals []int) *Key {
	dst := make([]int, len(arr))
	copy(dst, arr)
	return &Key{array: dst, atEnd: false, maxVals: maxVals}
}

func (key *Key) matchesMask(mask []int) bool {
	if len(key.array) != len(mask) {
		panic("key array length does not match mask")
	}
	for i := 0; i < len(mask); i++ {
		if (key.array[i] == DefaultMinVal && mask[i] != DefaultMinVal) ||
			(key.array[i] != DefaultMinVal && mask[i] == DefaultMinVal) {
			return false
		}
	}
	return true
}

func (key *Key) reset() {
	for i := 0; i < len(key.array); i++ {
		key.array[i] = DefaultMinVal
	}
	key.atEnd = false
}

func (key *Key) increment() {
	if key.atEnd {
		return
	}
	minimumPointerToMove := len(key.array) - 1
	for key.array[minimumPointerToMove] == key.maxVals[minimumPointerToMove] {
		minimumPointerToMove--
		if minimumPointerToMove < 0 {
			key.atEnd = true
			return
		}
	}
	key.array[minimumPointerToMove]++
	for i := minimumPointerToMove + 1; i < len(key.array); i++ {
		key.array[i] = DefaultMinVal
	}
}

func (key *Key) isAtEnd() bool {
	return key.atEnd
}

func (key *Key) toString() string {
	return strings.Trim(strings.Join(strings.Fields(fmt.Sprint(key.array)), ","), "[]")
}

func (key *Key) toStringOmitNull() string {
	nullRemoved := []int{}
	for i := 0; i < len(key.array); i++ {
		if key.array[i] != DefaultMinVal {
			nullRemoved = append(nullRemoved, key.array[i])
		}
	}
	return strings.Trim(strings.Join(strings.Fields(fmt.Sprint(nullRemoved)), ","), "[]")
}

func (key *Key) toStringWithMask(mask []int) string {
	if len(key.array) != len(mask) {
		panic("mask and key array are not the same length")
	}
	maskedArray := make([]int, len(key.array))
	copy(maskedArray, key.array)
	for i := 0; i < len(mask); i++ {
		if mask[i] == DefaultMinVal {
			maskedArray[i] = DefaultMinVal
		}
	}
	maskedKey := newKeyFromArray(maskedArray, key.maxVals)
	return maskedKey.toString()
}

// def increment_pointers(pointers, number_of_tiles):
//     minimum_pointer_to_move = len(pointers) - 1
//     endpoint_for_pointer = number_of_tiles - 1
//     while pointers[minimum_pointer_to_move] == endpoint_for_pointer:
//         minimum_pointer_to_move -= 1
//         endpoint_for_pointer -= 1
//         if minimum_pointer_to_move < 0:
//             return None

//     old_minimum_pointer_to_move_value = pointers[minimum_pointer_to_move]
//     for i in range (minimum_pointer_to_move, len(pointers)):
//         # Move pointer to break point instead
//         old_minimum_pointer_to_move_value += 1
//         pointers[i] = old_minimum_pointer_to_move_value

//     return pointers
