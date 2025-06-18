package movegen

import (
	"log"
	"sync"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/samber/lo"

	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tinymove"
)

var SmallPlaySlicePool = sync.Pool{
	New: func() interface{} {
		s := make([]tinymove.SmallMove, 0)
		return &s
	},
}

type PlayRecorderFunc func(MoveGenerator, *tilemapping.Rack, int, int, move.MoveType, int)

func NullPlayRecorder(gen MoveGenerator, a *tilemapping.Rack, leftstrip, rightstrip int, t move.MoveType, score int) {
}

func AllPlaysRecorder(gen MoveGenerator, rack *tilemapping.Rack, leftstrip, rightstrip int, t move.MoveType, score int) {
	gordonGen, ok := gen.(*GordonGenerator)
	if !ok {
		// For now, only GordonGenerator is supported.
		return
	}
	switch t {
	case move.MoveTypePlay:
		startRow := gordonGen.curRowIdx
		tilesPlayed := gordonGen.tilesPlayed

		startCol := leftstrip
		row := startRow
		col := startCol
		if gordonGen.vertical {
			// We flip it here because we only generate vertical moves when we transpose
			// the board, so the row and col are actually transposed.
			row, col = col, row
		}

		length := rightstrip - leftstrip + 1
		if length < 2 {
			return
		}
		word := make([]tilemapping.MachineLetter, length)
		copy(word, gordonGen.strip[startCol:startCol+length])

		alph := gordonGen.letterDistribution.TileMapping()
		play := move.NewScoringMove(score, word, rack.TilesOn(), gordonGen.vertical,
			tilesPlayed, alph, row, col)
		gordonGen.plays = append(gordonGen.plays, play)

	case move.MoveTypeExchange:
		// ignore the empty exchange case
		if rightstrip == 0 {
			return
		}
		if rightstrip > gordonGen.maxCanExchange {
			return
		}
		alph := gordonGen.letterDistribution.TileMapping()
		exchanged := make([]tilemapping.MachineLetter, rightstrip)
		copy(exchanged, gordonGen.exchangestrip[:rightstrip])
		play := move.NewExchangeMove(exchanged, rack.TilesOn(), alph)
		gordonGen.plays = append(gordonGen.plays, play)
	case move.MoveTypePass:
		alph := gordonGen.letterDistribution.TileMapping()
		gordonGen.plays = append(gordonGen.plays, move.NewPassMove(rack.TilesOn(), alph))

	default:

	}

}

// AllPlaysSmallRecorder is a recorder that records all plays, but as "SmallMove"s,
// which allocate much less and are smaller overall than a regular move.Move
func AllPlaysSmallRecorder(gen MoveGenerator, rack *tilemapping.Rack, leftstrip, rightstrip int, t move.MoveType, score int) {
	gordonGen, ok := gen.(*GordonGenerator)
	if !ok {
		// For now, only GordonGenerator is supported.
		return
	}
	switch t {

	case move.MoveTypePlay:
		startRow := gordonGen.curRowIdx
		startCol := leftstrip
		row := startRow
		col := startCol
		if gordonGen.vertical {
			// We flip it here because we only generate vertical moves when we transpose
			// the board, so the row and col are actually transposed.
			row, col = col, row
		}

		length := rightstrip - leftstrip + 1
		if length < 2 {
			return
		}
		var moveCode uint64
		tidx := 0
		bts := 20 // start at a bitshift of 20 for the first tile
		var blanksMask int
		for i := startCol; i < startCol+length; i++ {
			ml := gordonGen.strip[i]
			if ml == 0 {
				// play-through tile
				continue
			}
			it := ml.IntrinsicTileIdx()
			val := ml
			if it == 0 {
				blanksMask |= (1 << tidx)
				// this would be a designated blank
				val = ml.Unblank()
			}

			moveCode |= (uint64(val) << bts)

			tidx++
			bts += 6
		}
		if gordonGen.vertical {
			moveCode |= 1
		}
		moveCode |= (uint64(col) << 1)
		moveCode |= (uint64(row) << 6)
		moveCode |= (uint64(blanksMask) << 12)
		gordonGen.smallPlays = append(gordonGen.smallPlays, tinymove.TilePlayMove(
			tinymove.TinyMove(moveCode), int16(score), uint8(gordonGen.tilesPlayed),
			uint8(length)))

	case move.MoveTypeExchange:
		// Not meant for this, yet.
		log.Fatal("move type exchange is not compatible with SmallMove")
	case move.MoveTypePass:
		gordonGen.smallPlays = append(gordonGen.smallPlays, tinymove.PassMove())
	default:

	}

}

// TopPlayOnlyRecorder is a heavily optimized, ugly function to avoid allocating
// a lot of moves just to throw them out. It only records the very top move.
func TopPlayOnlyRecorder(gen MoveGenerator, rack *tilemapping.Rack, leftstrip, rightstrip int, t move.MoveType, score int) {
	gordonGen, ok := gen.(*GordonGenerator)
	if !ok {
		// For now, only GordonGenerator is supported.
		return
	}
	var eq float64
	var tilesLength int
	var leaveLength int

	switch t {
	case move.MoveTypePlay:
		startRow := gordonGen.curRowIdx
		tilesPlayed := gordonGen.tilesPlayed

		startCol := leftstrip
		row := startRow
		col := startCol
		if gordonGen.vertical {
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
		word := gordonGen.strip[startCol : startCol+tilesLength]
		leaveLength = rack.NoAllocTilesOn(gordonGen.leavestrip)

		gordonGen.placeholder.Set(word, gordonGen.leavestrip[:leaveLength], score,
			row, col, tilesPlayed, gordonGen.vertical, move.MoveTypePlay,
			gordonGen.letterDistribution.TileMapping())
		if len(gordonGen.equityCalculators) > 0 {
			eq = lo.SumBy(gordonGen.equityCalculators, func(c equity.EquityCalculator) float64 {
				return c.Equity(gordonGen.placeholder, gordonGen.board, gordonGen.game.Bag(), gordonGen.game.RackFor(gordonGen.game.NextPlayer()))
			})
		} else {
			eq = float64(score)
		}

	case move.MoveTypeExchange:
		// ignore the empty exchange case
		if rightstrip == 0 {
			return
		}
		if rightstrip > gordonGen.maxCanExchange {
			return
		}
		tilesLength = rightstrip
		exchanged := gordonGen.exchangestrip[:rightstrip]
		leaveLength = rack.NoAllocTilesOn(gordonGen.leavestrip)

		gordonGen.placeholder.Set(exchanged, gordonGen.leavestrip[:leaveLength], 0,
			0, 0, tilesLength, gordonGen.vertical, move.MoveTypeExchange,
			gordonGen.letterDistribution.TileMapping())

		eq = lo.SumBy(gordonGen.equityCalculators, func(c equity.EquityCalculator) float64 {
			return c.Equity(gordonGen.placeholder, gordonGen.board, gordonGen.game.Bag(), gordonGen.game.RackFor(gordonGen.game.NextPlayer()))
		})
	case move.MoveTypePass:
		leaveLength = rack.NoAllocTilesOn(gordonGen.leavestrip)
		alph := gordonGen.letterDistribution.TileMapping()
		gordonGen.placeholder.Set(nil, gordonGen.leavestrip[:leaveLength],
			0, 0, 0, 0, false, move.MoveTypePass, alph)
		eq = lo.SumBy(gordonGen.equityCalculators, func(c equity.EquityCalculator) float64 {
			return c.Equity(gordonGen.placeholder, gordonGen.board, gordonGen.game.Bag(), gordonGen.game.RackFor(gordonGen.game.NextPlayer()))
		})
	default:

	}
	if gordonGen.winner.IsEmpty() || eq > gordonGen.winner.Equity() {
		gordonGen.winner.CopyFrom(gordonGen.placeholder)
		gordonGen.winner.SetEquity(eq)
		if len(gordonGen.plays) == 0 {
			gordonGen.plays = append(gordonGen.plays, gordonGen.winner)
		} else {
			gordonGen.plays[0] = gordonGen.winner
		}
	}

}

func TopNPlayRecorder(gen MoveGenerator, rack *tilemapping.Rack, leftstrip, rightstrip int, t move.MoveType, score int) {
	gordonGen, ok := gen.(*GordonGenerator)
	if !ok {
		// For now, only GordonGenerator is supported.
		return
	}
	var eq float64
	var tilesLength int
	var leaveLength int

	switch t {
	case move.MoveTypePlay:
		startRow := gordonGen.curRowIdx
		tilesPlayed := gordonGen.tilesPlayed

		startCol := leftstrip
		row := startRow
		col := startCol
		if gordonGen.vertical {
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
		word := gordonGen.strip[startCol : startCol+tilesLength]
		leaveLength = rack.NoAllocTilesOn(gordonGen.leavestrip)

		gordonGen.placeholder.Set(word, gordonGen.leavestrip[:leaveLength], score,
			row, col, tilesPlayed, gordonGen.vertical, move.MoveTypePlay,
			gordonGen.letterDistribution.TileMapping())
		if len(gordonGen.equityCalculators) > 0 {
			eq = lo.SumBy(gordonGen.equityCalculators, func(c equity.EquityCalculator) float64 {
				return c.Equity(gordonGen.placeholder, gordonGen.board, gordonGen.game.Bag(), gordonGen.game.RackFor(gordonGen.game.NextPlayer()))
			})
		} else {
			eq = float64(score)
		}

	case move.MoveTypeExchange:
		// ignore the empty exchange case
		if rightstrip == 0 {
			return
		}
		if rightstrip > gordonGen.maxCanExchange {
			return
		}
		tilesLength = rightstrip
		exchanged := gordonGen.exchangestrip[:rightstrip]
		leaveLength = rack.NoAllocTilesOn(gordonGen.leavestrip)

		gordonGen.placeholder.Set(exchanged, gordonGen.leavestrip[:leaveLength], 0,
			0, 0, tilesLength, gordonGen.vertical, move.MoveTypeExchange,
			gordonGen.letterDistribution.TileMapping())

		eq = lo.SumBy(gordonGen.equityCalculators, func(c equity.EquityCalculator) float64 {
			return c.Equity(gordonGen.placeholder, gordonGen.board, gordonGen.game.Bag(), gordonGen.game.RackFor(gordonGen.game.NextPlayer()))
		})
	case move.MoveTypePass:
		leaveLength = rack.NoAllocTilesOn(gordonGen.leavestrip)
		alph := gordonGen.letterDistribution.TileMapping()
		gordonGen.placeholder.Set(nil, gordonGen.leavestrip[:leaveLength],
			0, 0, 0, 0, false, move.MoveTypePass, alph)
		eq = lo.SumBy(gordonGen.equityCalculators, func(c equity.EquityCalculator) float64 {
			return c.Equity(gordonGen.placeholder, gordonGen.board, gordonGen.game.Bag(), gordonGen.game.RackFor(gordonGen.game.NextPlayer()))
		})
	default:

	}

	newIdx := -1
	// fmt.Printf("finding a spot for eq %v (play %s)\n", eq, gen.placeholder.ShortDescription())
	for i := 0; i < gordonGen.maxTopMovesSize; i++ {
		if eq > gordonGen.topNPlays[i].Equity() {
			newIdx = i
			// fmt.Printf("inserting eq %v at idx %v\n", eq, newIdx)
			break
		}
	}
	if newIdx != -1 {
		// fmt.Println("before", gen.topNPlays)
		// Shift right the moves to make room for the new move
		for j := gordonGen.maxTopMovesSize - 1; j > newIdx; j-- {
			gordonGen.topNPlays[j].CopyFrom(gordonGen.topNPlays[j-1])
		}
		// Insert the new move at the correct position
		gordonGen.topNPlays[newIdx].CopyFrom(gordonGen.placeholder)
		gordonGen.topNPlays[newIdx].SetEquity(eq)
		// fmt.Println(gen.topNPlays)
	}
}
