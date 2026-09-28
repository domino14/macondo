package bot

import (
	"math"
	"os"
	"sort"
	"strconv"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/stats"
)

// The value head is P(win) - P(loss) for the position after a play. Once a
// game is decided it pins near +/-1 for every candidate and its ranking among
// them is noise: the 57% model, down 250 with 8 in the bag, walked past a
// 176-point bingo for a 7-point dump because both scored -0.999. Its spread
// head still resolves those positions, so once the best candidate's value is
// beyond mlDecidedValue, the candidates within mlDecidedTieWindow of it are
// ranked by their expected final spread instead: the spread after the play
// plus the head's predicted change from there to the end of the game.
//
// MACONDO_ML_DECIDED_THRESHOLD overrides the threshold ("0" turns the rule
// off), so a match can A/B it without a rebuild.
const (
	mlDecidedValue     = 0.97
	mlDecidedTieWindow = 0.01
)

func mlDecidedThreshold() float32 {
	if v := os.Getenv("MACONDO_ML_DECIDED_THRESHOLD"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return float32(f)
		}
	}
	return mlDecidedValue
}

// expectedFinalSpread is the mover's spread after playing m plus the spread
// head's predicted change to the end of the game, in points.
func expectedFinalSpread(m *move.Move, spreadNow int, spreadOut float32) float64 {
	after := float64(spreadNow)
	if m.Action() == move.MoveTypePlay {
		after += float64(m.Score())
	}
	x := math.Max(-0.999999, math.Min(0.999999, float64(spreadOut)))
	return after + float64(game.MLSpreadScale)*math.Atanh(x)
}

// rankMLMoves orders the candidates best first. `values` and `spreads` are
// the model's outputs per candidate; spreadNow is the mover's spread before
// the play. A zero or negative threshold disables the decided-game rule.
func rankMLMoves(moves []*move.Move, values, spreads []float32, spreadNow int, threshold float32) []*move.Move {
	n := len(moves)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	byValue := func(a, b int) bool {
		// Equal values: prefer the play that uses more tiles (helps the endgame).
		if stats.FuzzyEqual(float64(values[a]), float64(values[b])) {
			return moves[a].TilesPlayed() > moves[b].TilesPlayed()
		}
		return values[a] > values[b]
	}
	sort.Slice(idx, func(i, j int) bool { return byValue(idx[i], idx[j]) })

	best := values[idx[0]]
	if threshold > 0 && len(spreads) == n && (best >= threshold || best <= -threshold) {
		// Decided: among the candidates the value head cannot tell apart,
		// take the one with the best expected final spread.
		k := 0
		for k < n && values[idx[k]] >= best-mlDecidedTieWindow {
			k++
		}
		tied := idx[:k]
		sort.SliceStable(tied, func(i, j int) bool {
			return expectedFinalSpread(moves[tied[i]], spreadNow, spreads[tied[i]]) >
				expectedFinalSpread(moves[tied[j]], spreadNow, spreads[tied[j]])
		})
	}
	out := make([]*move.Move, n)
	for i, j := range idx {
		out[i] = moves[j]
	}
	return out
}
