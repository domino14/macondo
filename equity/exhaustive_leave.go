package equity

import (
	"strings"

	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
)

type BlankLeaves struct{}

func (b *BlankLeaves) LeaveValue(leave tilemapping.MachineWord) float64 {
	return float64(0.0)
}

// ExhaustiveLeaveCalculator should apply an equity calculation for all leaves
// exhaustively.
type ExhaustiveLeaveCalculator struct {
	leaveValues Leaves
}

func defaultForLexicon(lexiconName string) string {
	// If there doesn't exist a specific folder with the name of the
	// lexicon, we'll call this function.
	if strings.HasPrefix(lexiconName, "CSW") {
		return "CSW21"
	} else if strings.HasPrefix(lexiconName, "TWL") ||
		strings.HasPrefix(lexiconName, "NWL") ||
		strings.HasPrefix(lexiconName, "NSWL") {

		return "NWL23"
	} else if strings.HasPrefix(lexiconName, "ECWL") || // obsolete name for CEL
		strings.HasPrefix(lexiconName, "CEL") { // common english words
		return "ECWL"
	} else if strings.HasPrefix(lexiconName, "RD") {
		return "RD28"
	} else if strings.HasPrefix(lexiconName, "NSF") {
		return "NSF23"
	} else if strings.HasPrefix(lexiconName, "FRA") {
		return "FRA24"
	} else if strings.HasPrefix(lexiconName, "DISC") {
		return "DISC2"
	} else if strings.HasPrefix(lexiconName, "OSPS") {
		return "OSPS49"
	}
	return ""
}

func NewExhaustiveLeaveCalculator(lexiconName string,
	cfg *config.Config, leaveFilename string) (
	*ExhaustiveLeaveCalculator, error) {

	calc := &ExhaustiveLeaveCalculator{}
	if leaveFilename == "" {
		leaveFilename = LeavesFilename
	}

	leaves, err := cache.Load(cfg.AllSettings(), "leavefile:"+lexiconName+":"+leaveFilename, LeaveCacheLoadFunc)
	if err != nil {
		log.Err(err).Msg("loading-leaves")
	}

	var ok bool
	calc.leaveValues, ok = leaves.(*KLV)
	if !ok {
		log.Info().Msg("no leaves found, will use greedy strategy")
		calc.leaveValues = &BlankLeaves{}
	}

	return calc, nil
}

func (els ExhaustiveLeaveCalculator) Equity(play *move.Move, board *board.GameBoard,
	bag *tilemapping.Bag, oppRack *tilemapping.Rack) float64 {

	if bag.TilesRemaining() > 0 {
		return float64(play.Score()) + els.LeaveValue(play.Leave())
	}
	return float64(play.Score())
}

func (els ExhaustiveLeaveCalculator) LeaveValue(leave tilemapping.MachineWord) float64 {
	return els.leaveValues.LeaveValue(leave)
}
