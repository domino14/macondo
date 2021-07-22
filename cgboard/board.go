package cgboard

import (
	"errors"
	"fmt"
	"os"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"
)

var (
	ColorSupport = os.Getenv("MACONDO_DISABLE_COLOR") != "on"
)

type BonusSquare byte

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

const (
	// Bonus3WS is a triple word score
	Bonus3WS BonusSquare = 61 // =  (hex 3D)
	// Bonus3LS is a triple letter score
	Bonus3LS BonusSquare = 34 // "  (hex 22)
	// Bonus2LS is a double letter score
	Bonus2LS BonusSquare = 39 // '  (hex 27)
	// Bonus2WS is a double word score
	Bonus2WS BonusSquare = 45 // -  (hex 2D)

	NoBonus BonusSquare = 32 // space (hex 20)
)

func (b BonusSquare) displayString() string {
	repr := string(rune(b))
	if !ColorSupport {
		return repr
	}
	switch b {

	case Bonus3WS:
		return fmt.Sprintf("\033[31m%s\033[0m", repr)
	case Bonus2WS:
		return fmt.Sprintf("\033[35m%s\033[0m", repr)
	case Bonus3LS:
		return fmt.Sprintf("\033[34m%s\033[0m", repr)
	case Bonus2LS:
		return fmt.Sprintf("\033[36m%s\033[0m", repr)
	default:
		return "?"
	}
}

// GameBoard will store a one-dimensional array of tiles played.
type GameBoard struct {
	squares     []alphabet.MachineLetter
	bonuses     []BonusSquare
	transposed  bool
	tilesPlayed int
	dim         int
	lastCopy    *GameBoard

	// Store cross-scores with the board to avoid recalculating, but cross-sets
	// are a movegen detail and do not belong here!
	// vCrossScores []int
	// hCrossScores []int
}

func MakeBoard(desc []string) *GameBoard {
	// Turns an array of strings into the GameBoard structure type.
	// Assume all strings are the same length
	totalLen := 0
	for _, s := range desc {
		totalLen += len(s)
	}
	sqs := make([]alphabet.MachineLetter, totalLen)
	bs := make([]BonusSquare, totalLen)
	sqi := 0
	for _, s := range desc {
		for _, c := range s {
			bs[sqi] = BonusSquare(byte(c))
			sqi++
		}

	}
	g := &GameBoard{squares: sqs, bonuses: bs, dim: len(desc)}
	return g
}

func (g *GameBoard) TilesPlayed() int {
	return g.tilesPlayed
}

// Dim is the dimension of the board. It assumes the board is square.
func (g *GameBoard) Dim() int {
	return g.dim
}

func (g *GameBoard) GetBonus(row int, col int) BonusSquare {
	return g.bonuses[row*g.dim+col]
}

func (g *GameBoard) SetLetter(row int, col int, letter alphabet.MachineLetter) {
	g.squares[row*g.dim+col] = letter
}

func (g *GameBoard) GetLetter(row int, col int) alphabet.MachineLetter {
	return g.squares[row*g.dim+col]
}

func (g *GameBoard) HasLetter(row int, col int) bool {
	return g.GetLetter(row, col) != alphabet.EmptySquareMarker
}

func (g *GameBoard) SquareDisplayString(row, col int, alph *alphabet.Alphabet) string {
	disp := " "
	pos := row*g.dim + col
	letter := g.squares[pos]
	bonus := g.bonuses[pos]
	if bonus == NoBonus {
		if letter != alphabet.EmptySquareMarker {
			disp = string(letter.UserVisible(alph))
		}
	} else {
		disp = bonus.displayString()
	}
	return disp
}

// Transpose transposes the board. It doesn't actually change the layout of the
// squares; the iterating functions must switch rows and columns themselves.
func (g *GameBoard) Transpose() {
	g.transposed = !g.transposed
}

func (g *GameBoard) IsTransposed() bool {
	return g.transposed
}

// Clear clears the board.
func (g *GameBoard) Clear() {
	for i := 0; i < len(g.squares); i++ {
		g.squares[i] = alphabet.EmptySquareMarker
	}
	g.tilesPlayed = 0
}

// IsEmpty returns if the board is empty.
func (g *GameBoard) IsEmpty() bool {
	return g.tilesPlayed == 0
}

func (g *GameBoard) PosExists(row int, col int) bool {
	d := g.Dim()
	return row >= 0 && row < d && col >= 0 && col < d
}

// LeftAndRightEmpty returns true if the squares at col - 1 and col + 1
// on this row are empty, checking carefully for boundary conditions.
func (g *GameBoard) LeftAndRightEmpty(row int, col int) bool {
	if g.PosExists(row, col-1) {
		if g.HasLetter(row, col-1) {
			return false
		}
	}
	if g.PosExists(row, col+1) {
		if g.HasLetter(row, col+1) {
			return false
		}
	}
	return true
}

// WordEdge finds the edge of a word on the board, returning the column.
func (g *GameBoard) WordEdge(row int, col int, dir WordDirection) int {
	for g.PosExists(row, col) && g.HasLetter(row, col) {
		col += int(dir)
	}
	return col - int(dir)
}

func (g *GameBoard) TraverseBackwardsForScore(row int, col int, ld *alphabet.LetterDistribution) int {
	score := 0
	for g.PosExists(row, col) {
		ml := g.GetLetter(row, col)
		if ml == alphabet.EmptySquareMarker {
			break
		}
		score += ld.Score(ml)
		col--
	}
	return score
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
		g.SetLetter(row, col, tile)
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
		g.SetLetter(row, col, alphabet.EmptySquareMarker)
	}
}

// PlayMove plays a move on a board.
func (g *GameBoard) PlayMove(m *move.Move, ld *alphabet.LetterDistribution) {
	if m.Action() != move.MoveTypePlay {
		return
	}
	g.PlaceMoveTiles(m)
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
	for g.PosExists(newrow, newcol) && g.HasLetter(newrow, newcol) {
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
	for g.PosExists(newrow, newcol) && g.HasLetter(newrow, newcol) {
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

// Copy returns a deep copy of this board.
func (g *GameBoard) Copy() *GameBoard {
	newg := &GameBoard{}
	newg.squares = make([]alphabet.MachineLetter, len(g.squares))
	newg.bonuses = make([]BonusSquare, len(g.bonuses))
	for i, s := range g.squares {
		newg.squares[i] = s
	}
	for i, b := range g.bonuses {
		newg.bonuses[i] = b
	}
	newg.transposed = g.transposed
	newg.tilesPlayed = g.tilesPlayed
	newg.dim = g.dim

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
	for i, r := range b.squares {
		g.squares[i] = r
		g.bonuses[i] = b.bonuses[i]
	}
	g.transposed = b.transposed
	g.tilesPlayed = b.tilesPlayed
	g.dim = b.dim
}

func (g *GameBoard) GetTilesPlayed() int {
	return g.tilesPlayed
}

func (g *GameBoard) TestSetTilesPlayed(n int) {
	g.tilesPlayed = n
}
