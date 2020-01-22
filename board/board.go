package board

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
)

type BoardDirection uint8
type WordDirection int

func (bd BoardDirection) String() string {
	if bd == HorizontalDirection {
		return "(horizontal)"
	} else if bd == VerticalDirection {
		return "(vertical)"
	}
	return "none"
}

const (
	HorizontalDirection BoardDirection = iota
	VerticalDirection
)

const (
	LeftDirection  WordDirection = -1
	RightDirection WordDirection = 1
)

// A GameBoard is the main board structure. It contains all of the Squares,
// with bonuses or filled letters, as well as cross-sets and cross-scores
// for computation. (See Appel & Jacobson paper for definition of the latter
// two terms)
type GameBoard struct {
	squares     [][]*Square
	transposed  bool
	tilesPlayed int
}

// MakeBoard creates a board from a description string.
func MakeBoard(desc []string) *GameBoard {
	// Turns an array of strings into the GameBoard structure type.
	rows := [][]*Square{}
	for _, s := range desc {
		row := []*Square{}
		for _, c := range s {
			row = append(row, &Square{letter: alphabet.EmptySquareMarker, bonus: BonusSquare(c)})
		}
		rows = append(rows, row)
	}
	g := &GameBoard{squares: rows}
	return g
}

// Dim is the dimension of the board. It assumes the board is square.
func (g *GameBoard) Dim() int {
	return len(g.squares)
}

func (g *GameBoard) GetBonus(row int, col int) BonusSquare {
	return g.squares[row][col].bonus
}

func (g *GameBoard) GetSquare(row int, col int) *Square {
	return g.squares[row][col]
}

func (g *GameBoard) SetLetter(row int, col int, letter alphabet.MachineLetter) {
	g.squares[row][col].letter = letter
}

func (g *GameBoard) GetLetter(row int, col int) alphabet.MachineLetter {
	return g.GetSquare(row, col).letter
}

func (g *GameBoard) GetCrossSet(row int, col int, dir BoardDirection) CrossSet {
	return *g.squares[row][col].getCrossSet(dir) // the actual value
}

func (g *GameBoard) ClearCrossSet(row int, col int, dir BoardDirection) {
	g.squares[row][col].getCrossSet(dir).clear()
}

func (g *GameBoard) SetCrossSetLetter(row int, col int, dir BoardDirection,
	ml alphabet.MachineLetter) {
	g.squares[row][col].getCrossSet(dir).set(ml)
}

func (g *GameBoard) GetCrossScore(row int, col int, dir BoardDirection) int {
	return g.squares[row][col].getCrossScore(dir)
}

// Transpose transposes the board, swapping rows and columns.
func (g *GameBoard) Transpose() {
	n := g.Dim()
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			g.squares[i][j], g.squares[j][i] = g.squares[j][i], g.squares[i][j]
		}
	}
	g.transposed = !g.transposed
}

func (g *GameBoard) IsTransposed() bool {
	return g.transposed
}

// SetAllCrosses sets the cross sets of every square to every acceptable letter.
func (g *GameBoard) SetAllCrosses() {
	// Assume square board. This should be an assertion somewhere.
	n := g.Dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.squares[i][j].hcrossSet.setAll()
			g.squares[i][j].vcrossSet.setAll()
		}
	}
}

// ClearAllCrosses disallows all letters on all squares (more or less).
func (g *GameBoard) ClearAllCrosses() {
	n := g.Dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.squares[i][j].hcrossSet.clear()
			g.squares[i][j].vcrossSet.clear()
		}
	}
}

// Clear clears the board.
func (g *GameBoard) Clear() {
	n := g.Dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.squares[i][j].letter = alphabet.EmptySquareMarker
		}
	}
	g.tilesPlayed = 0
	// We set all crosses because every letter is technically allowed
	// on every cross-set at the very beginning.
	g.SetAllCrosses()
	g.UpdateAllAnchors()
}

// IsEmpty returns if the board is empty.
func (g *GameBoard) IsEmpty() bool {
	return g.tilesPlayed == 0
}

func (g *GameBoard) updateAnchors(row int, col int, vertical bool) {
	if vertical {
		// This helps simplify the updateAnchorsForMove algorithm.
		row, col = col, row
	}
	// Always reset the anchors before applying anything else.
	g.squares[row][col].resetAnchors()
	var tileAbove, tileBelow, tileLeft, tileRight, tileHere bool
	if row > 0 {
		tileAbove = !g.squares[row-1][col].IsEmpty()
	}
	if col > 0 {
		tileLeft = !g.squares[row][col-1].IsEmpty()
	}
	if row < g.Dim()-1 {
		tileBelow = !g.squares[row+1][col].IsEmpty()
	}
	if col < g.Dim()-1 {
		tileRight = !g.squares[row][col+1].IsEmpty()
	}
	tileHere = !g.squares[row][col].IsEmpty()
	if tileHere {
		// The current square is not empty. It should only be an anchor
		// if it is the rightmost square of a word (actually, squares to
		// the left are probably ok, but not the leftmost square. Note
		// Gordon does not have this requirement, but the algorithm does
		// not work if we don't do this)
		if !tileRight {
			g.squares[row][col].setAnchor(HorizontalDirection)
		}
		// Apply the transverse logic too for the vertical anchor.
		if !tileBelow {
			g.squares[row][col].setAnchor(VerticalDirection)
		}
	} else {
		// If the square is empty, it should only be an anchor if the
		// squares to its left and right are empty, and at least one of
		// the squares in the top and bottom are NOT empty.
		if !tileLeft && !tileRight && (tileAbove || tileBelow) {
			g.squares[row][col].setAnchor(HorizontalDirection)
		}
		// (And apply the transverse logic for the vertical anchor)
		if !tileAbove && !tileBelow && (tileLeft || tileRight) {
			g.squares[row][col].setAnchor(VerticalDirection)
		}
	}
}

func (g *GameBoard) UpdateAllAnchors() {
	n := g.Dim()
	if g.tilesPlayed > 0 {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				g.updateAnchors(i, j, false)
			}
		}
	} else {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				g.squares[i][j].resetAnchors()
			}
		}
		rc := int(n / 2)
		// If the board is empty, set just one anchor, in the center square.
		g.squares[rc][rc].hAnchor = true
	}
}

// IsAnchor returns whether the row/col pair is an anchor in the given
// direction.
func (g *GameBoard) IsAnchor(row int, col int, dir BoardDirection) bool {
	return g.squares[row][col].anchor(dir)
}

func (g *GameBoard) posExists(row int, col int) bool {
	d := g.Dim()
	return row >= 0 && row < d && col >= 0 && col < d
}

// leftAndRightEmpty returns true if the squares at col - 1 and col + 1
// on this row are empty, checking carefully for boundary conditions.
func (g *GameBoard) leftAndRightEmpty(row int, col int) bool {
	if g.posExists(row, col-1) {
		if !g.squares[row][col-1].IsEmpty() {
			return false
		}
	}
	if g.posExists(row, col+1) {
		if !g.squares[row][col+1].IsEmpty() {
			return false
		}
	}
	return true
}

// wordEdge finds the edge of a word on the board, returning the column.
//
func (g *GameBoard) wordEdge(row int, col int, dir WordDirection) int {
	for g.posExists(row, col) && !g.squares[row][col].IsEmpty() {
		col += int(dir)
	}
	return col - int(dir)
}

func (g *GameBoard) traverseBackwardsForScore(row int, col int, bag *alphabet.Bag) int {
	score := 0
	for g.posExists(row, col) {
		ml := g.squares[row][col].letter
		if ml == alphabet.EmptySquareMarker {
			break
		}
		score += bag.Score(ml)
		col--
	}
	return score
}

func (g *GameBoard) traverseBackwards(row int, col int, nodeIdx uint32,
	checkLetterSet bool, leftMostCol int,
	gaddag *gaddag.SimpleGaddag) (uint32, bool) {
	// Traverse the letters on the board backwards (left). Return the index
	// of the node in the gaddag for the left-most letter, and a boolean
	// indicating if the gaddag path was valid.
	// If checkLetterSet is true, then we traverse until leftMostCol+1 and
	// check the letter set of this node to see if it includes the letter
	// at leftMostCol
	for g.posExists(row, col) {
		ml := g.squares[row][col].letter
		if ml == alphabet.EmptySquareMarker {
			break
		}

		if checkLetterSet && col == leftMostCol {
			if gaddag.InLetterSet(ml, nodeIdx) {
				return nodeIdx, true
			}
			// Give up early; if we're checking letter sets we only care about
			// this column.
			return nodeIdx, false
		}

		nodeIdx = gaddag.NextNodeIdx(nodeIdx, ml.Unblank())
		if nodeIdx == 0 {
			// There is no path in the gaddag for this word part; this
			// can occur if a phony was played and stayed on the board
			// and the phony has no extensions for example, or if it's
			// a real word with no further extensions.
			return nodeIdx, false
		}

		col--
	}

	return nodeIdx, true
}

func (g *GameBoard) updateAnchorsForMove(m *move.Move) {
	row, col, vertical := m.CoordsAndVertical()

	if vertical {
		// Transpose the logic, but NOT the board. The updateAnchors function
		// assumes the board is not transposed.
		col, row = row, col
	}

	// Update anchors all around the play.
	for i := col; i < len(m.Tiles())+col; i++ {
		g.updateAnchors(row, i, vertical)
		if row > 0 {
			g.updateAnchors(row-1, i, vertical)
		}
		if row < g.Dim()-1 {
			g.updateAnchors(row+1, i, vertical)
		}
	}

	if col-1 >= 0 {
		g.updateAnchors(row, col-1, vertical)
	}
	if len(m.Tiles())+col < g.Dim() {
		g.updateAnchors(row, col+len(m.Tiles()), vertical)
	}

}

func (g *GameBoard) placeMoveTiles(m *move.Move) {
	rowStart, colStart, vertical := m.CoordsAndVertical()
	var row, col int
	for idx, tile := range m.Tiles() {
		if tile == alphabet.PlayedThroughMarker {
			continue
		}
		if vertical {
			row = rowStart + idx
			col = colStart
		} else {
			col = colStart + idx
			row = rowStart
		}
		g.squares[row][col].letter = tile
	}
}

func (g *GameBoard) unplaceMoveTiles(m *move.Move) {
	rowStart, colStart, vertical := m.CoordsAndVertical()
	var row, col int
	for idx, tile := range m.Tiles() {
		if tile == alphabet.PlayedThroughMarker {
			continue
		}
		if vertical {
			row = rowStart + idx
			col = colStart
		} else {
			col = colStart + idx
			row = rowStart
		}
		g.squares[row][col].letter = alphabet.EmptySquareMarker
	}
}

// PlayMove plays a move on a board. It must place tiles on the board,
// regenerate cross-sets and cross-points, and recalculate anchors.
func (g *GameBoard) PlayMove(m *move.Move, gd *gaddag.SimpleGaddag,
	bag *alphabet.Bag) {

	// g.playHistory = append(g.playHistory, m.ShortDescription())
	if m.Action() != move.MoveTypePlay {
		return
	}
	g.placeMoveTiles(m)
	// Calculate anchors.
	g.updateAnchorsForMove(m)
	// Calculate cross-sets.
	g.updateCrossSetsForMove(m, gd, bag)
	g.tilesPlayed += m.TilesPlayed()
}

// ScoreWord scores the move at the given row and column. Note that this
// function is called when the board is potentially transposed, so we
// assume the row stays static as we iterate through the letters of the
// word.
func (g *GameBoard) ScoreWord(word alphabet.MachineWord, row, col, tilesPlayed int, crossDir BoardDirection, bag *alphabet.Bag) int {

	// letterScore:
	var ls int

	mainWordScore := 0
	crossScores := 0
	bingoBonus := 0
	if tilesPlayed == 7 {
		bingoBonus = 50
	}
	wordMultiplier := 1

	for idx, rn := range word {
		ml := alphabet.MachineLetter(rn)
		bonusSq := g.GetBonus(row, col+idx)
		letterMultiplier := 1
		thisWordMultiplier := 1
		freshTile := false
		if ml == alphabet.PlayedThroughMarker {
			ml = g.GetLetter(row, col+idx)
		} else {
			freshTile = true
			// Only count bonus if we are putting a fresh tile on it.
			switch bonusSq {
			case Bonus3WS:
				wordMultiplier *= 3
				thisWordMultiplier = 3
			case Bonus2WS:
				wordMultiplier *= 2
				thisWordMultiplier = 2
			case Bonus2LS:
				letterMultiplier = 2
			case Bonus3LS:
				letterMultiplier = 3
			}
			// else all the multipliers are 1.
		}
		cs := g.GetCrossScore(row, col+idx, crossDir)
		if ml >= alphabet.BlankOffset {
			// letter score is 0
			ls = 0
		} else {
			ls = bag.Score(ml)
		}

		mainWordScore += ls * letterMultiplier
		// We only add cross scores if the cross set of this square is non-trivial
		// (i.e. we have to be making an across word). Note that it's not enough
		// to check that the cross-score is 0 because we could have a blank.
		if freshTile && g.GetCrossSet(row, col+idx, crossDir) != TrivialCrossSet {
			crossScores += ls*letterMultiplier*thisWordMultiplier + cs*thisWordMultiplier
		}
	}
	return mainWordScore*wordMultiplier + crossScores + bingoBonus

}

// RestoreFromBackup restores the squares of this board from the backupBoard.
// func (g *GameBoard) RestoreFromBackup() {

// 	for ridx, r := range g.squaresBackup {
// 		for cidx, c := range r {
// 			g.squares[ridx][cidx].copyFrom(c)
// 		}
// 	}
// 	g.tilesPlayed = g.tilesPlayedBackup
// 	g.playHistory = append([]string{}, g.playHistoryBackup...)
// }

// GameHash returns a "Hash" of this game's play history.
// func (g *GameBoard) GameHash() string {
// 	joined := strings.Join(g.playHistory, ",")
// 	return fmt.Sprintf("%x", md5.Sum([]byte(joined)))
// }

// Copy returns a deep copy of this board.
func (g *GameBoard) Copy() *GameBoard {
	newg := &GameBoard{}
	squares := [][]*Square{}

	for _, r := range g.squares {
		row := []*Square{}
		for _, c := range r {
			s := &Square{}
			s.copyFrom(c)
			row = append(row, s)
		}
		squares = append(squares, row)
	}
	newg.squares = squares
	newg.transposed = g.transposed
	newg.tilesPlayed = g.tilesPlayed
	// newg.playHistory = append([]string{}, g.playHistory...)
	return newg
}

// CopyFrom copies the squares and other info from b back into g.
func (g *GameBoard) CopyFrom(b *GameBoard) {
	for ridx, r := range b.squares {
		for cidx, sq := range r {
			g.squares[ridx][cidx].copyFrom(sq)
		}
	}
	g.transposed = b.transposed
	g.tilesPlayed = b.tilesPlayed
}
