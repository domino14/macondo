package equity

import (
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cache"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
)

type BlankLeaves struct{}

func (b *BlankLeaves) LeaveValue(leave alphabet.MachineWord) float64 {
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
	if strings.HasPrefix(lexiconName, "CSW") ||
		strings.HasPrefix(lexiconName, "TWL") ||
		strings.HasPrefix(lexiconName, "NWL") ||
		strings.HasPrefix(lexiconName, "ECWL") || // obsolete name for CEL
		strings.HasPrefix(lexiconName, "CEL") || // common english words
		strings.HasPrefix(lexiconName, "NSWL") {

		return "default_english"
	} else if strings.HasPrefix(lexiconName, "RD") {
		return "german"
	} else if strings.HasPrefix(lexiconName, "NSF") {
		return "norwegian"
	} else if strings.HasPrefix(lexiconName, "FRA") {
		return "french"
	}
	return ""
}

func NewExhaustiveLeaveCalculator(lexiconName string,
	cfg *config.Config, leaveFilename string) (
	*ExhaustiveLeaveCalculator, error) {

	calc := &ExhaustiveLeaveCalculator{}
	if leaveFilename == "" {
		leaveFilename = LeaveFilename
	}

	leaves, err := cache.Load(cfg, "leavefile:"+lexiconName+":"+leaveFilename, LeaveCacheLoadFunc)
	if err != nil {
		log.Err(err).Msg("loading-leaves")
	}

	var ok bool
	calc.leaveValues, ok = leaves.(*OldLeaves)
	if !ok {
		log.Info().Msg("no leaves found, will use greedy strategy")
		calc.leaveValues = &BlankLeaves{}
	}

	return calc, nil
}

func (els ExhaustiveLeaveCalculator) Equity(play *move.Move, board *board.GameBoard,
	bag *alphabet.Bag, oppRack *alphabet.Rack) float64 {

	if bag.TilesRemaining() > 0 {
		return float64(play.Score()) + els.LeaveValue(play.Leave())
	}
	return float64(play.Score())
}

func (els ExhaustiveLeaveCalculator) LeaveValue(leave alphabet.MachineWord) float64 {
	return els.leaveValues.LeaveValue(leave)
}
