// Package strategy encapsulates all strategic concerns for crossword game.
// This includes for example equity calculations, first-turn placement
// heuristics, and more. It shouldn't be placed in the movegen package
// as that package is solely for generating moves.
package strategy

import (
	"encoding/csv"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

var DataDir = os.Getenv("DATA_DIR")

// Strategizer is an interface used by the movegen (and possibly other places)
// to apply strategical calculations.
type Strategizer interface {
	// Init initializes the strategizer, by doing things such as loading parameters
	// from disk.
	Init(lexiconName string, alph *alphabet.Alphabet, leavefile string) error
	// Equity is a catch-all term for most post-score adjustments that
	// need to be made. It includes first-turn placement heuristic,
	// leave calculations, any pre-endgame timing heuristics, and more.
	Equity(play *move.Move, board *board.GameBoard) float64
}

// NoLeaveStrategy does not take leave into account at all.
type NoLeaveStrategy struct{}

// ExhaustiveLeaveStrategy should apply an equity calculation for all leaves
// exhaustively.
type ExhaustiveLeaveStrategy struct{}

// SimpleSynergyStrategy uses a strategy borne of a simple synergy calculation,
// based on a few million computer vs computer games.
// The details of this calculation are in the /notebooks directory in this
// repo.
type SimpleSynergyStrategy struct {
	leaveMap SynergyLeaveMap
}

func (sss *SimpleSynergyStrategy) Init(lexiconName string, alph *alphabet.Alphabet,
	leavefile string) error {
	leaveMap := map[string]SynergyAndEV{}
	file, err := os.Open(filepath.Join(DataDir, lexiconName, leavefile))
	if err != nil {
		return err
	}
	defer file.Close()

	r := csv.NewReader(file)

	records, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	for idx, record := range records {
		if idx == 0 {
			continue // skip; header row
		}
		letters := record[0]
		str, err := alphabet.ToMachineOnlyString(letters, alph)
		if err != nil {
			log.Fatal(err)
		}
		ev, err := strconv.ParseFloat(record[4], 64)
		if err != nil {
			log.Fatal(err)
		}
		synergy, err := strconv.ParseFloat(record[5], 64)
		if err != nil {
			// Single tiles don't have synergy.
			synergy = 0
		}
		leaveMap[str] = SynergyAndEV{
			synergy: synergy,
			ev:      ev,
		}
	}
	sss.leaveMap = SynergyLeaveMap(leaveMap)
	return nil
}

func (sss SimpleSynergyStrategy) Equity(play *move.Move, board *board.GameBoard) float64 {
	leave := play.Leave()
	score := play.Score()

	adjustment := 0.0
	if board.IsEmpty() {
		adjustment += placementAdjustment(play)
	}

	leaveLookup := sss.lookup(leave)

	// also need a pre-endgame adjustment that biases towards leaving
	// one in the bag, etc.
	return leaveLookup + float64(score) + adjustment
}

func (sss SimpleSynergyStrategy) lookup(leave alphabet.MachineWord) float64 {
	if len(leave) > 1 {
		sort.Slice(leave, func(i, j int) bool {
			return leave[i] < leave[j]
		})
	}
	if len(leave) <= 3 {
		return sss.leaveMap[string(leave)].ev
	}
	// Otherwise, do a rough calculation using pairwise synergies.
	leaveval := 0.0
	for _, ml := range leave {
		leaveval += sss.leaveMap[string([]alphabet.MachineLetter{ml})].ev
	}
	for i := 0; i < len(leave); i++ {
		for j := i + 1; j < len(leave); j++ {
			tolookup := []byte{byte(leave[i]), byte(leave[j])}
			leaveval += sss.leaveMap[string(tolookup)].synergy
		}
	}
	return leaveval
}

func placementAdjustment(play *move.Move) float64 {
	// Very simply just checks how many vowels are overlapping bonus squares.
	// This only gets considered when the board is empty.
	if play.Action() != move.MoveTypePlay {
		return 0
	}
	row, col, vertical := play.CoordsAndVertical()
	var start, end int
	start = col
	if vertical {
		start = row
	}
	end = start + play.TilesPlayed()

	j := start
	penalty := 0.0
	vPenalty := -0.7 // VERY ROUGH approximation from Maven paper.
	for j < end {
		if play.Tiles()[j-start].IsVowel(play.Alphabet()) {
			switch j {
			case 2, 6, 8, 12:
				// row/col below/above have a 2LS. note this only works
				// for specific board configurations
				penalty += vPenalty
			default:

			}
		}
		j++
	}
	return penalty
}

func (nls *NoLeaveStrategy) Init(string, *alphabet.Alphabet, string) error {
	return nil
}

func (nls *NoLeaveStrategy) Equity(play *move.Move, board *board.GameBoard) float64 {
	score := play.Score()
	adjustment := 0.0
	if board.IsEmpty() {
		adjustment += placementAdjustment(play)
	}
	return float64(score) + adjustment
}
