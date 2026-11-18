package bot

import (
	"errors"

	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/move"
)

// MLScores returns the net's value head for each move, as the ML bot would
// score them (its own model, its last-move history).
func (p *BotTurnPlayer) MLScores(moves []*move.Move) ([]float32, error) {
	var lc *equity.ExhaustiveLeaveCalculator
	for _, c := range p.Calculators() {
		if e, ok := c.(*equity.ExhaustiveLeaveCalculator); ok {
			lc = e
			break
		}
	}
	if lc == nil {
		return nil, errors.New("no ExhaustiveLeaveCalculator")
	}
	resp, err := p.MLEvaluateMovesWith(p.tritonClient, moves, lc, p.lastMoves)
	if err != nil {
		return nil, err
	}
	return resp.Value, nil
}
