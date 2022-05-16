package board

import (
	"errors"
	"fmt"

	"github.com/domino14/macondo/alphabet"
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
	squares     [][]Square
	transposed  bool
	tilesPlayed int
	lastCopy    *GameBoard
}

// MakeBoard creates a board from a description string.
// Assumption: strings are ASCII.
func MakeBoard(desc []string) *GameBoard {
	// Turns an array of strings into the GameBoard structure type.
	rows := make([][]Square, len(desc))
	totalLen := 0
	for _, s := range desc {
		totalLen += len(s)
	}
	sqs := make([]Square, totalLen)
	sqp := 0
	for si, s := range desc {
		sqp += len(s)
		rows[si] = sqs[sqp-len(s) : sqp : sqp]
		for ci, c := range s {
			rows[si][ci] = Square{letter: alphabet.EmptySquareMarker, bonus: BonusSquare(c)}
		}
	}
	g := &GameBoard{squares: rows}
	// Call Clear to set all crosses.
	g.Clear()
	return g
}

func (g *GameBoard) TilesPlayed() int {
	return g.tilesPlayed
}

// Dim is the dimension of the board. It assumes the board is square.
func (g *GameBoard) Dim() int {
	return len(g.squares)
}

func (g *GameBoard) GetBonus(row int, col int) BonusSquare {
	return g.squares[row][col].bonus
}

func (g *GameBoard) GetSquare(row int, col int) *Square {
	return &g.squares[row][col]
}

func (g *GameBoard) SetLetter(row int, col int, letter alphabet.MachineLetter) {
	g.squares[row][col].letter = letter
}

func (g *GameBoard) GetLetter(row int, col int) alphabet.MachineLetter {
	return g.GetSquare(row, col).letter
}

func (g *GameBoard) HasLetter(row int, col int) bool {
	return !g.GetSquare(row, col).IsEmpty()
}

func (g *GameBoard) GetCrossSet(row int, col int, dir BoardDirection) CrossSet {
	return *g.squares[row][col].GetCrossSet(dir) // the actual value
}

func (g *GameBoard) ClearCrossSet(row int, col int, dir BoardDirection) {
	g.squares[row][col].GetCrossSet(dir).Clear()
}

func (g *GameBoard) SetCrossSetLetter(row int, col int, dir BoardDirection,
	ml alphabet.MachineLetter) {
	g.squares[row][col].GetCrossSet(dir).Set(ml)
}

func (g *GameBoard) GetCrossScore(row int, col int, dir BoardDirection) int {
	return g.squares[row][col].GetCrossScore(dir)
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
			g.squares[i][j].hcrossSet.SetAll()
			g.squares[i][j].vcrossSet.SetAll()
		}
	}
}

// ClearAllCrosses disallows all letters on all squares (more or less).
func (g *GameBoard) ClearAllCrosses() {
	n := g.Dim()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g.squares[i][j].hcrossSet.Clear()
			g.squares[i][j].vcrossSet.Clear()
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

func (g *GameBoard) PosExists(row int, col int) bool {
	d := g.Dim()
	return row >= 0 && row < d && col >= 0 && col < d
}

// LeftAndRightEmpty returns true if the squares at col - 1 and col + 1
// on this row are empty, checking carefully for boundary conditions.
func (g *GameBoard) LeftAndRightEmpty(row int, col int) bool {
	if g.PosExists(row, col-1) {
		if !g.squares[row][col-1].IsEmpty() {
			return false
		}
	}
	if g.PosExists(row, col+1) {
		if !g.squares[row][col+1].IsEmpty() {
			return false
		}
	}
	return true
}

// WordEdge finds the edge of a word on the board, returning the column.
//
func (g *GameBoard) WordEdge(row int, col int, dir WordDirection) int {
	for g.PosExists(row, col) && !g.squares[row][col].IsEmpty() {
		col += int(dir)
	}
	return col - int(dir)
}

func (g *GameBoard) TraverseBackwardsForScore(row int, col int, ld *alphabet.LetterDistribution) int {
	score := 0
	for g.PosExists(row, col) {
		ml := g.squares[row][col].letter
		if ml == alphabet.EmptySquareMarker {
			break
		}
		score += ld.Score(ml)
		col--
	}
	return score
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

func (g *GameBoard) PlaceMoveTiles(m *move.Move) {
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

func (g *GameBoard) UnplaceMoveTiles(m *move.Move) {
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
func (g *GameBoard) PlayMove(m *move.Move, ld *alphabet.LetterDistribution) {

	// g.playHistory = append(g.playHistory, m.ShortDescription())
	if m.Action() != move.MoveTypePlay {
		return
	}
	g.PlaceMoveTiles(m)
	// Calculate anchors.
	g.updateAnchorsForMove(m)
	g.tilesPlayed += m.TilesPlayed()
}

// ErrorIfIllegalPlay returns an error if the play is illegal, or nil otherwise.
// We are not checking the actual validity of the word, but whether it is a
// legal Crossword Game move.
func (g *GameBoard) ErrorIfIllegalPlay(row, col int, vertical bool,
	word alphabet.MachineWord) error {

	ri, ci := 0, 1
	if vertical {
		ri, ci = ci, ri
	}
	boardEmpty := g.IsEmpty()
	touchesCenterSquare := false
	bordersATile := false
	placedATile := false
	for idx, ml := range word {
		newrow, newcol := row+(ri*idx), col+(ci*idx)

		if boardEmpty && newrow == g.Dim()>>1 && newcol == g.Dim()>>1 {
			touchesCenterSquare = true
		}

		if newrow < 0 || newrow >= g.Dim() || newcol < 0 || newcol >= g.Dim() {
			return errors.New("play extends off of the board")
		}

		if ml == alphabet.PlayedThroughMarker {
			ml = g.GetLetter(newrow, newcol)
			if ml == alphabet.EmptySquareMarker {
				return errors.New("a played-through marker was specified, but " +
					"there is no tile at the given location")
			}
			bordersATile = true
		} else {
			ml = g.GetLetter(newrow, newcol)
			if ml != alphabet.EmptySquareMarker {
				return fmt.Errorf("tried to play through a letter already on "+
					"the board; please use the played-through marker (.) instead "+
					"(row %v col %v ml %v)", newrow, newcol, ml)
			}

			// We are placing a tile on this empty square. Check if we border
			// any other tiles.

			for d := -1; d <= 1; d += 2 {
				// only check perpendicular hooks
				checkrow, checkcol := newrow+ci*d, newcol+ri*d
				if g.PosExists(checkrow, checkcol) && g.GetLetter(checkrow, checkcol) != alphabet.EmptySquareMarker {
					bordersATile = true
				}
			}

			placedATile = true
		}
	}

	if boardEmpty && !touchesCenterSquare {
		return errors.New("the first play must touch the center square")
	}
	if !boardEmpty && !bordersATile {
		return errors.New("your play must border a tile already on the board")
	}
	if !placedATile {
		return errors.New("your play must place a new tile")
	}
	if len(word) < 2 {
		return errors.New("your play must include at least two letters")
	}
	{
		checkrow, checkcol := row-ri, col-ci
		if g.PosExists(checkrow, checkcol) && g.GetLetter(checkrow, checkcol) != alphabet.EmptySquareMarker {
			return errors.New("your play must include the whole word")
		}
	}
	{
		checkrow, checkcol := row+ri*len(word), col+ci*len(word)
		if g.PosExists(checkrow, checkcol) && g.GetLetter(checkrow, checkcol) != alphabet.EmptySquareMarker {
			return errors.New("your play must include the whole word")
		}
	}
	return nil
}

// FormedWords returns an array of all machine words formed by this move.
// The move is assumed to be of type Play
func (g *GameBoard) FormedWords(m *move.Move) ([]alphabet.MachineWord, error) {
	// Reserve space for main word.
	words := []alphabet.MachineWord{nil}
	mainWord := []alphabet.MachineLetter{}

	row, col, vertical := m.CoordsAndVertical()
	ri, ci := 0, 1
	if vertical {
		ri, ci = ci, ri
	}

	if m.Action() != move.MoveTypePlay {
		return nil, errors.New("function must be called with a tile placement play")
	}

	for idx, letter := range m.Tiles() {
		// For the purpose of checking words, all letters should be unblanked.
		letter = letter.Unblank()
		newrow, newcol := row+(ri*idx), col+(ci*idx)

		// This is the main word.
		if letter == alphabet.PlayedThroughMarker {
			letter = g.GetLetter(newrow, newcol).Unblank()
			mainWord = append(mainWord, letter)
			continue
		}
		mainWord = append(mainWord, letter)
		crossWord := g.formedCrossWord(!vertical, letter, newrow, newcol)
		if crossWord != nil {
			words = append(words, crossWord)
		}
	}
	// Prepend the main word to the slice. We do this to establish a convention
	// that this slice always contains the main formed word first.
	// Space for this is already reserved upfront to avoid unnecessary copying.
	words[0] = mainWord

	return words, nil
}

func (g *GameBoard) formedCrossWord(crossVertical bool, letter alphabet.MachineLetter,
	row, col int) alphabet.MachineWord {

	ri, ci := 0, 1
	if crossVertical {
		ri, ci = ci, ri
	}

	// Given the cross-word direction (crossVertical) and a letter located at row, col
	// find the cross-word that contains this letter (if any)
	// Look in the cross direction for newly played tiles.
	crossword := []alphabet.MachineLetter{}

	newrow := row - ri
	newcol := col - ci
	// top/left and bottom/right row/column pairs.
	var tlr, tlc, brr, brc int

	// Find the top or left edge.
	for g.PosExists(newrow, newcol) && !g.squares[newrow][newcol].IsEmpty() {
		newrow -= ri
		newcol -= ci
	}
	newrow += ri
	newcol += ci
	tlr = newrow
	tlc = newcol

	// Find bottom or right edge
	newrow, newcol = row, col
	newrow += ri
	newcol += ci
	for g.PosExists(newrow, newcol) && !g.squares[newrow][newcol].IsEmpty() {
		newrow += ri
		newcol += ci
	}
	newrow -= ri
	newcol -= ci
	// what a ghetto function, sorry future me
	brr = newrow
	brc = newcol

	for rowiter, coliter := tlr, tlc; rowiter <= brr && coliter <= brc; rowiter, coliter = rowiter+ri, coliter+ci {
		if rowiter == row && coliter == col {
			crossword = append(crossword, letter.Unblank())
		} else {
			crossword = append(crossword, g.GetLetter(rowiter, coliter).Unblank())
		}
	}
	if len(crossword) < 2 {
		// there are no 1-letter words, Josh >:(
		return nil
	}
	return crossword
}

// ScoreWord scores the move at the given row and column. Note that this
// function is called when the board is potentially transposed, so we
// assume the row stays static as we iterate through the letters of the
// word.
func (g *GameBoard) ScoreWord(word alphabet.MachineWord, row, col, tilesPlayed int,
	crossDir BoardDirection, ld *alphabet.LetterDistribution) int {

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
			case Bonus4WS:
				wordMultiplier *= 4
				thisWordMultiplier = 4
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
			case Bonus4LS:
				letterMultiplier = 4
			}
			// else all the multipliers are 1.
		}
		cs := g.GetCrossScore(row, col+idx, crossDir)
		if ml >= alphabet.BlankOffset {
			// letter score is 0
			ls = 0
		} else {
			ls = ld.Score(ml)
		}

		mainWordScore += ls * letterMultiplier
		// We only add cross scores if we are making an "across" word).
		// Note that we look up and down because the word is always horizontal
		// in this routine (board might or might not be transposed).
		actualCrossWord := (row > 0 && g.HasLetter(row-1, col+idx)) || (row < g.Dim()-1 && g.HasLetter(row+1, col+idx))

		if freshTile && actualCrossWord {
			crossScores += ls*letterMultiplier*thisWordMultiplier + cs*thisWordMultiplier
		}
	}
	return mainWordScore*wordMultiplier + crossScores + bingoBonus

}

// Copy returns a deep copy of this board.
func (g *GameBoard) Copy() *GameBoard {
	newg := &GameBoard{}
	newg.squares = make([][]Square, len(g.squares))

	totalLen := 0
	for _, r := range g.squares {
		totalLen += len(r)
	}
	sqs := make([]Square, totalLen)
	sqp := 0
	for ri, r := range g.squares {
		sqp += len(r)
		newg.squares[ri] = sqs[sqp-len(r) : sqp : sqp]
		for ci, c := range r {
			newg.squares[ri][ci].copyFrom(&c)
		}
	}
	newg.transposed = g.transposed
	newg.tilesPlayed = g.tilesPlayed
	// newg.playHistory = append([]string{}, g.playHistory...)
	return newg
}

func (g *GameBoard) SaveCopy() {
	g.lastCopy = g.Copy()
}

func (g *GameBoard) RestoreFromCopy() {
	g.CopyFrom(g.lastCopy)
	g.lastCopy = nil
}

// CopyFrom copies the squares and other info from b back into g.
func (g *GameBoard) CopyFrom(b *GameBoard) {
	for ridx, r := range b.squares {
		for cidx, sq := range r {
			g.squares[ridx][cidx].copyFrom(&sq)
		}
	}
	g.transposed = b.transposed
	g.tilesPlayed = b.tilesPlayed
}

func (g *GameBoard) GetTilesPlayed() int {
	return g.tilesPlayed
}

func (g *GameBoard) TestSetTilesPlayed(n int) {
	g.tilesPlayed = n
}
