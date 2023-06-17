package movegen

import (
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
	"github.com/samber/lo"
)

type PlayRecorderFunc func(*GordonGenerator, *tilemapping.Rack, int, int, move.MoveType)

func NullPlayRecorder(gen *GordonGenerator, a *tilemapping.Rack, leftstrip, rightstrip int, t move.MoveType) {
}

func AllPlaysRecorder(gen *GordonGenerator, rack *tilemapping.Rack, leftstrip, rightstrip int, t move.MoveType) {

	switch t {
	case move.MoveTypePlay:
		startRow := gen.curRowIdx
		tilesPlayed := gen.tilesPlayed

		startCol := leftstrip
		row := startRow
		col := startCol
		if gen.vertical {
			// We flip it here because we only generate vertical moves when we transpose
			// the board, so the row and col are actually transposed.
			row, col = col, row
		}

		length := rightstrip - leftstrip + 1
		if length < 2 {
			return
		}
		word := make([]tilemapping.MachineLetter, length)
		copy(word, gen.strip[startCol:startCol+length])

		alph := gen.letterDistribution.TileMapping()
		play := move.NewScoringMove(gen.scoreMove(word, startRow, startCol, tilesPlayed),
			word, rack.TilesOn(), gen.vertical,
			tilesPlayed, alph, row, col)
		gen.plays = append(gen.plays, play)

	case move.MoveTypeExchange:
		// ignore the empty exchange case
		if rightstrip == 0 {
			return
		}
		alph := gen.letterDistribution.TileMapping()
		exchanged := make([]tilemapping.MachineLetter, rightstrip)
		copy(exchanged, gen.exchangestrip[:rightstrip])
		play := move.NewExchangeMove(exchanged, rack.TilesOn(), alph)
		gen.plays = append(gen.plays, play)
	case move.MoveTypePass:
		alph := gen.letterDistribution.TileMapping()
		gen.plays = append(gen.plays, move.NewPassMove(rack.TilesOn(), alph))

	default:

	}

}

// TopPlayOnlyRecorder is a heavily optimized, ugly function to avoid allocating
// a lot of moves just to throw them out. It only records the very top move.
func TopPlayOnlyRecorder(gen *GordonGenerator, rack *tilemapping.Rack, leftstrip, rightstrip int, t move.MoveType) {

	var eq float64
	var tilesLength int
	var leaveLength int

	switch t {
	case move.MoveTypePlay:
		startRow := gen.curRowIdx
		tilesPlayed := gen.tilesPlayed

		startCol := leftstrip
		row := startRow
		col := startCol
		if gen.vertical {
			// We flip it here because we only generate vertical moves when we transpose
			// the board, so the row and col are actually transposed.
			row, col = col, row
		}
		tilesLength = rightstrip - leftstrip + 1
		// word is in gen.strip[startCol:startCol+length]
		if tilesLength < 2 {
			return
		}
		// note that this is a pointer right now:
		word := gen.strip[startCol : startCol+tilesLength]

		score := gen.scoreMove(word, startRow, startCol, tilesPlayed)
		leaveLength = rack.NoAllocTilesOn(gen.leavestrip)

		gen.placeholder.Set(word, gen.leavestrip[:leaveLength], score,
			row, col, tilesPlayed, gen.vertical, move.MoveTypePlay,
			gen.letterDistribution.TileMapping())

		eq = lo.SumBy(gen.equityCalculators, func(c equity.EquityCalculator) float64 {
			return c.Equity(gen.placeholder, gen.board, gen.game.Bag(), gen.game.RackFor(gen.game.NextPlayer()))
		})

	case move.MoveTypeExchange:
		// ignore the empty exchange case
		if rightstrip == 0 {
			return
		}
		tilesLength = rightstrip
		exchanged := gen.exchangestrip[:rightstrip]
		leaveLength = rack.NoAllocTilesOn(gen.leavestrip)

		gen.placeholder.Set(exchanged, gen.leavestrip[:leaveLength], 0,
			0, 0, tilesLength, gen.vertical, move.MoveTypeExchange,
			gen.letterDistribution.TileMapping())

		eq = lo.SumBy(gen.equityCalculators, func(c equity.EquityCalculator) float64 {
			return c.Equity(gen.placeholder, gen.board, gen.game.Bag(), gen.game.RackFor(gen.game.NextPlayer()))
		})
	case move.MoveTypePass:
		leaveLength = rack.NoAllocTilesOn(gen.leavestrip)
		alph := gen.letterDistribution.TileMapping()
		gen.placeholder.Set(nil, gen.leavestrip[:leaveLength],
			0, 0, 0, 0, false, move.MoveTypePass, alph)
		eq = lo.SumBy(gen.equityCalculators, func(c equity.EquityCalculator) float64 {
			return c.Equity(gen.placeholder, gen.board, gen.game.Bag(), gen.game.RackFor(gen.game.NextPlayer()))
		})
	default:

	}
	if gen.winner.Action() == move.MoveTypeUnset || eq > gen.winner.Equity() {
		gen.winner.CopyFrom(gen.placeholder)
		gen.winner.SetEquity(eq)
		if len(gen.plays) == 0 {
			gen.plays = append(gen.plays, gen.winner)
		} else {
			gen.plays[0] = gen.winner
		}
	}

}
