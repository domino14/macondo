package puzzles

import (
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// computeStats computes PuzzleStats for the answer move in ctx.
// Returns nil and logs if ValidateMove fails (the puzzle still gets emitted).
func computeStats(ctx *tagCtx) *pb.PuzzleStats {
	m := ctx.moves[0]
	g := ctx.g
	brd := g.Board()
	ld := g.Bag().LetterDistribution()

	words, err := ctx.getWords()
	if err != nil {
		log.Err(err).Msg("computeStats: ValidateMove error")
		return nil
	}

	stats := &pb.PuzzleStats{}

	// Basic counts
	stats.WordsFormed = int32(len(words))
	if len(words) > 0 {
		stats.MainWordLength = int32(len(words[0]))
	}
	stats.TilesPlayed = int32(m.TilesPlayed())
	stats.Score = int32(m.Score())

	// Equity advantage
	if len(ctx.moves) >= 2 {
		stats.EquityAdvantage = ctx.moves[0].Equity() - ctx.moves[1].Equity()
	}

	// Score advantage and top-score bingo detection
	best, secondScore := ctx.getBestByScore()
	stats.TopScorePlayTilesPlayed = int32(best.TilesPlayed())
	if best == ctx.moves[0] {
		stats.ScoreAdvantage = int32(ctx.moves[0].Score() - secondScore)
	} else {
		// negative: answer is not the top scorer
		stats.ScoreAdvantage = int32(ctx.moves[0].Score() - best.Score())
	}

	// Cross-word min/max lengths
	if len(words) >= 2 {
		maxCross := 0
		minCross := 1<<31 - 1 // max int32
		for _, w := range words[1:] {
			l := len(w)
			if l > maxCross {
				maxCross = l
			}
			if l < minCross {
				minCross = l
			}
		}
		stats.MaxCrossWordLength = int32(maxCross)
		stats.MinCrossWordLength = int32(minCross)
	}

	// Iterate over the play to compute per-tile stats
	rowStart, colStart, vertical := m.CoordsAndVertical()
	ri, ci := 0, 1
	if vertical {
		ri, ci = 1, 0
	}

	firstFreshIdx, lastFreshIdx := -1, -1

	for idx, tile := range m.Tiles() {
		row := rowStart + ri*idx
		col := colStart + ci*idx

		if tile == 0 {
			// played-through existing tile
			existing := brd.GetLetter(row, col)
			s := int32(ld.Score(existing))
			if s > stats.MaxPlayedThroughTileScore {
				stats.MaxPlayedThroughTileScore = s
			}
			continue
		}

		// fresh tile
		if firstFreshIdx == -1 {
			firstFreshIdx = idx
		}
		lastFreshIdx = idx

		bonus := brd.GetBonus(row, col)
		switch bonus {
		case board.Bonus3WS:
			stats.TwsCovered++
		case board.Bonus2WS:
			stats.DwsCovered++
		case board.Bonus3LS:
			stats.TlsCovered++
		case board.Bonus2LS:
			stats.DlsCovered++
		}

		// "X on DLS forming XI" pattern
		if bonus == board.Bonus2LS || bonus == board.Bonus3LS {
			crossCount := crossTileCountAt(brd, row, col, vertical)
			if crossCount >= 1 {
				faceVal := int32(ld.Score(tile.Unblank()))
				if faceVal > stats.MaxFreshTileFaceValueOnLetterBonusWithCrossword {
					stats.MaxFreshTileFaceValueOnLetterBonusWithCrossword = faceVal
				}
			}
		}
	}

	stats.BonusSquaresCovered = stats.TwsCovered + stats.DwsCovered + stats.TlsCovered + stats.DlsCovered

	// Hooks: cross-tile count at the first and last fresh tile (existing tiles perpendicular)
	if firstFreshIdx != -1 {
		firstRow := rowStart + ri*firstFreshIdx
		firstCol := colStart + ci*firstFreshIdx
		lastRow := rowStart + ri*lastFreshIdx
		lastCol := colStart + ci*lastFreshIdx

		frontCross := crossTileCountAt(brd, firstRow, firstCol, vertical)
		backCross := crossTileCountAt(brd, lastRow, lastCol, vertical)
		hookedLen := frontCross
		if backCross > hookedLen {
			hookedLen = backCross
		}
		stats.LongestHookedWordLength = int32(hookedLen)
	}

	// Extensions: existing word at front/back of play along the main axis
	endIdx := len(m.Tiles()) - 1
	frontExtLen := walkCount(brd, rowStart-ri, colStart-ci, -ri, -ci)
	backExtLen := walkCount(brd, rowStart+ri*(endIdx+1), colStart+ci*(endIdx+1), ri, ci)
	extLen := frontExtLen
	if backExtLen > extLen {
		extLen = backExtLen
	}
	stats.LongestExtendedWordLength = int32(extLen)

	// Max rack tile score
	rack := g.RackFor(g.PlayerOnTurn()).TilesOn()
	for _, ml := range rack {
		s := int32(ld.Score(ml))
		if s > stats.MaxRackTileScore {
			stats.MaxRackTileScore = s
		}
	}

	return stats
}

// crossTileCountAt returns the count of existing tiles on the board
// perpendicular to the main play axis at (row, col).
// vertical==true means the main axis is vertical, so the cross axis is horizontal.
func crossTileCountAt(brd *board.GameBoard, row, col int, vertical bool) int {
	var ri, ci int
	if vertical {
		ri, ci = 0, 1 // cross is horizontal
	} else {
		ri, ci = 1, 0 // cross is vertical
	}
	count := 0
	for r, c := row-ri, col-ci; brd.PosExists(r, c) && brd.HasLetter(r, c); r, c = r-ri, c-ci {
		count++
	}
	for r, c := row+ri, col+ci; brd.PosExists(r, c) && brd.HasLetter(r, c); r, c = r+ri, c+ci {
		count++
	}
	return count
}

// walkCount counts contiguous existing tiles starting at (row, col) stepping by (dr, dc).
// Returns 0 if the starting square is empty or out of bounds.
func walkCount(brd *board.GameBoard, row, col, dr, dc int) int {
	n := 0
	for r, c := row, col; brd.PosExists(r, c) && brd.HasLetter(r, c); r, c = r+dr, c+dc {
		n++
	}
	return n
}
