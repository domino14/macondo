package turnplayer

import (
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/move"
)

// maxAllMoves caps how many moves a request for "all of them" keeps: MAGPIE
// keeps the best n in a heap sized up front, and callers asking for
// everything pass numbers like 1e6.
const maxAllMoves = 300000

// staticEval is how a static player ranks moves in MAGPIE: by MAGPIE's static
// equity with the player's leaves, or (with no leave calculator) by score.
type staticEval struct {
	byScore    bool
	leavesFile string
}

// leaveFiler is a calculator with leaves; its leave file picks MAGPIE's.
type leaveFiler interface {
	LeavesFile() string
}

func staticEvalFor(calcs []equity.EquityCalculator) staticEval {
	for _, c := range calcs {
		if lf, ok := c.(leaveFiler); ok {
			return staticEval{leavesFile: lf.LeavesFile()}
		}
	}
	return staticEval{byScore: true}
}

// magpieGenerate generates the best numPlays moves for the player on turn in
// g, best first, ranked as eval says.
func magpieGenerate(g *game.Game, eval staticEval, numPlays int) ([]*move.Move, error) {
	pos, err := g.MagpiePosition()
	if err != nil {
		return nil, err
	}
	sort := magpie.SortEquity
	if eval.byScore {
		sort = magpie.SortScore
	} else {
		d, err := g.Rules().MagpieDataWithLeaves(eval.leavesFile)
		if err != nil {
			return nil, err
		}
		pos.SetData(d)
	}
	if game.IsWordSmog(g.Rules().Variant()) {
		// Every permutation of a WordSmog multiset is its own play, so there
		// is no "all": keep a pool of the best (see GenerateWordSmogTopPlays).
		numPlays = min(max(numPlays, wordSmogMinPool), wordSmogMaxPool)
	}
	var moves []magpie.Move
	if numPlays == 1 {
		moves = pos.Generate(magpie.RecordBest, sort, 0)
	} else {
		moves = pos.GenerateN(magpie.RecordAll, sort, 0, min(numPlays, maxAllMoves))
	}
	plays := make([]*move.Move, len(moves))
	for i := range moves {
		plays[i] = g.MoveFromMagpie(&moves[i])
		if eval.byScore {
			if plays[i].Action() == move.MoveTypePass {
				plays[i].SetEquity(StaticPassPenalty)
			} else {
				plays[i].SetEquity(float64(plays[i].Score()))
			}
		}
	}
	return plays, nil
}
