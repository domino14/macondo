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

// Init initializes the strategizer, by doing things such as loading parameters
// from disk.
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
	log.Printf("Loaded strategy from %v", file.Name())
	return nil
}

func NewSimpleSynergyStrategy(bag *alphabet.Bag, lexiconName string,
	alph *alphabet.Alphabet, leaveFilename string) *SimpleSynergyStrategy {
	strategy := &SimpleSynergyStrategy{
		bag: bag,
	}
	err := strategy.Init(lexiconName, alph, leaveFilename)
	if err != nil {
		log.Printf("[ERROR] Initializing strategy: %v", err)
		return nil
	}
	return strategy
}

func (sss SimpleSynergyStrategy) Equity(play *move.Move, board *board.GameBoard,
	bag *alphabet.Bag, oppRack *alphabet.Rack) float64 {
	leave := play.Leave()
	score := play.Score()

	leaveAdjustment := 0.0
	otherAdjustments := 0.0
	if board.IsEmpty() {
		otherAdjustments += placementAdjustment(play)
	}
	if bag.TilesRemaining() == 0 {
		otherAdjustments += endgameAdjustment(play, oppRack, sss.bag)
	} else {
		// the leave doesn't matter if the bag is empty
		leaveAdjustment = float64(sss.LeaveValue(leave))
	}

	// also need a pre-endgame adjustment that biases towards leaving
	// one in the bag, etc.
	return float64(score) + leaveAdjustment + otherAdjustments
}

func (sss SimpleSynergyStrategy) LeaveValue(leave alphabet.MachineWord) float32 {
	if len(leave) > 1 {
		sort.Slice(leave, func(i, j int) bool {
			return leave[i] < leave[j]
		})
	}
	if len(leave) <= 3 {
		return float32(sss.leaveMap[string(leave)].ev)
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
	return float32(leaveval)
}

func (nls *NoLeaveStrategy) Equity(play *move.Move, board *board.GameBoard,
	bag *alphabet.Bag, oppRack *alphabet.Rack) float64 {
	score := play.Score()
	adjustment := 0.0
	if board.IsEmpty() {
		adjustment += placementAdjustment(play)
	}
	return float64(score) + adjustment
}

func (nls *NoLeaveStrategy) LeaveValue(leave alphabet.MachineWord) float32 {
	return float32(0.0)
}
