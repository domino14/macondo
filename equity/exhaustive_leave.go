package equity

import (
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

func NewExhaustiveLeaveCalculator(lexiconName string,
	cfg *config.Config, leaveFilename string) (
	*ExhaustiveLeaveCalculator, error) {

	calc := &ExhaustiveLeaveCalculator{}
	if leaveFilename == "" {
		leaveFilename = LeavesFilename
	}

	leaves, err := cache.Load(cfg.WGLConfig(), "leavefile:"+lexiconName+":"+leaveFilename, LeaveCacheLoadFunc)
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

func (els ExhaustiveLeaveCalculator) EquityWithLeave(play *move.Move, board *board.GameBoard,
	bag *tilemapping.Bag, oppRack *tilemapping.Rack, leaveValue float64) float64 {

	if bag.TilesRemaining() > 0 {
		return float64(play.Score()) + leaveValue
	}
	return float64(play.Score())
}

// Type returns the type of the equity calculator.
func (els ExhaustiveLeaveCalculator) Type() string {
	return "ExhaustiveLeaveCalculator"
}

func (els ExhaustiveLeaveCalculator) LeaveValue(leave tilemapping.MachineWord) float64 {
	return els.leaveValues.LeaveValue(leave)
}
