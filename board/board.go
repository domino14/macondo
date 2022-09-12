package board

import (
	"errors"
	"fmt"
	"os"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/move"
	"github.com/rs/zerolog/log"
)

var (
	ColorSupport = os.Getenv("MACONDO_DISABLE_COLOR") != "on"
)

type BonusSquare byte
type CrossSet uint64

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
	// Bonus4WS is a quadruple word score
	Bonus4WS BonusSquare = '~'
	// Bonus4LS is a quadruple letter score
	Bonus4LS BonusSquare = '^'
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
	case Bonus4WS:
		return fmt.Sprintf("\033[33m%s\033[0m", repr)
	case Bonus3WS:
		return fmt.Sprintf("\033[31m%s\033[0m", repr)
	case Bonus2WS:
		return fmt.Sprintf("\033[35m%s\033[0m", repr)
	case Bonus4LS:
		return fmt.Sprintf("\033[95m%s\033[0m", repr)
	case Bonus3LS:
		return fmt.Sprintf("\033[34m%s\033[0m", repr)
	case Bonus2LS:
		return fmt.Sprintf("\033[36m%s\033[0m", repr)
	case NoBonus:
		return " "
	default:
		return "?"
	}
}

// GameBoard will store a one-dimensional array of tiles played.
type GameBoard struct {
	squares     []alphabet.MachineLetter
	bonuses     []BonusSquare
	tilesPlayed int
	dim         int
	lastCopy    *GameBoard

	// Store cross-scores with the board to avoid recalculating, but cross-sets
	// are a movegen detail and do not belong here!
	hCrossScores []int
	vCrossScores []int
	// The rest of these are definitely movegen details and they
	// really should not be here. However, let's do this one step at a time.
	hCrossSets []CrossSet
	vCrossSets []CrossSet
	hAnchors   []bool
	vAnchors   []bool

	rowMul int
	colMul int
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
	vc := make([]int, totalLen)
	hc := make([]int, totalLen)
	hcs := make([]CrossSet, totalLen)
	vcs := make([]CrossSet, totalLen)
	hAs := make([]bool, totalLen)
	vAs := make([]bool, totalLen)
	sqi := 0
	for _, s := range desc {
		for _, c := range s {
			bs[sqi] = BonusSquare(byte(c))
			sqs[sqi] = alphabet.EmptySquareMarker
			sqi++
		}

	}
	g := &GameBoard{
		squares:      sqs,
		bonuses:      bs,
		dim:          len(desc),
		vCrossScores: vc,
		hCrossScores: hc,
		hCrossSets:   hcs,
		vCrossSets:   vcs,
		hAnchors:     hAs,
		vAnchors:     vAs,
		rowMul:       len(desc),
		colMul:       1,
	}
	return g
}

func (g *GameBoard) TilesPlayed() int {
	return g.tilesPlayed
}

// Dim is the dimension of the board. It assumes the board is square.
func (g *GameBoard) Dim() int {
	return g.dim
}

// Transpose the board in-place. We should copy transposed boards in the future.
func (g *GameBoard) Transpose() {
	// for i := 0; i < g.dim; i++ {
	// 	for j := i + 1; j < g.dim; j++ {
	// 		rm := i*g.dim + j
	// 		cm := j*g.dim + i
	// 		g.squares[rm], g.squares[cm] = g.squares[cm], g.squares[rm]
	// 		g.hCrossScores[rm], g.hCrossScores[cm] = g.hCrossScores[cm], g.hCrossScores[rm]
	// 		g.vCrossScores[rm], g.vCrossScores[cm] = g.vCrossScores[cm], g.vCrossScores[rm]
	// 		g.hCrossSets[rm], g.hCrossSets[cm] = g.hCrossSets[cm], g.hCrossSets[rm]
	// 		g.vCrossSets[rm], g.vCrossSets[cm] = g.vCrossSets[cm], g.vCrossSets[rm]
	// 		g.hAnchors[rm], g.hAnchors[cm] = g.hAnchors[cm], g.hAnchors[rm]
	// 		g.vAnchors[rm], g.vAnchors[cm] = g.vAnchors[cm], g.vAnchors[rm]
	// 		// ignore bonuses.
	// 	}
	// }
	g.rowMul, g.colMul = g.colMul, g.rowMul
}

func (g *GameBoard) getSqIdx(row, col int) int {
	return row*g.rowMul + col*g.colMul
}

func (g *GameBoard) GetBonus(row int, col int) BonusSquare {
	// No need to check for transpositions as bonuses are rotationally invariant
	// (but implement the transpose check to ease merge resolution)
	return g.bonuses[g.getSqIdx(row, col)]
}

func (g *GameBoard) SetLetter(row int, col int, letter alphabet.MachineLetter) {
	g.squares[g.getSqIdx(row, col)] = letter
}

func (g *GameBoard) GetLetter(row int, col int) alphabet.MachineLetter {
	return g.squares[g.getSqIdx(row, col)]
}

func (g *GameBoard) GetCrossScore(row int, col int, dir BoardDirection) int {
	pos := g.getSqIdx(row, col)

	switch dir {
	case HorizontalDirection:
		return g.hCrossScores[pos]
	case VerticalDirection:
		return g.vCrossScores[pos]
	default:
		log.Error().Msgf("Unknown direction: %v\n", dir)
		return 0
	}
}

func (g *GameBoard) SetCrossScore(row, col, score int, dir BoardDirection) {
	pos := g.getSqIdx(row, col)
	switch dir {
	case HorizontalDirection:
		g.hCrossScores[pos] = score
	case VerticalDirection:
		g.vCrossScores[pos] = score
	default:
		log.Error().Msgf("Unknown direction: %v\n", dir)
	}
}

func (g *GameBoard) ResetCrossScores() {
	for i := range g.hCrossScores {
		g.hCrossScores[i] = 0
	}
	for i := range g.vCrossScores {
		g.vCrossScores[i] = 0
	}
}

func (g *GameBoard) SetCrossSet(row int, col int, cs CrossSet,
	dir BoardDirection) {
	pos := g.getSqIdx(row, col)
	switch dir {
	case HorizontalDirection:
		g.hCrossSets[pos] = cs
	case VerticalDirection:
		g.vCrossSets[pos] = cs
	default:
		log.Error().Msgf("Unknown direction: %v\n", dir)
	}
}

func (g *GameBoard) HasLetter(row int, col int) bool {
	return g.GetLetter(row, col) != alphabet.EmptySquareMarker
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
		if ml > alphabet.BlankOffset {
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
	newg.squares = make([]alphabet.MachineLetter, len(g.squares))
	newg.bonuses = make([]BonusSquare, len(g.bonuses))
	newg.vCrossScores = make([]int, len(g.vCrossScores))
	newg.hCrossScores = make([]int, len(g.hCrossScores))
	newg.hCrossSets = make([]CrossSet, len(g.hCrossSets))
	newg.vCrossSets = make([]CrossSet, len(g.vCrossSets))
	newg.hAnchors = make([]bool, len(g.vCrossSets))
	newg.vAnchors = make([]bool, len(g.vCrossSets))

	copy(newg.squares, g.squares)
	copy(newg.bonuses, g.bonuses)
	copy(newg.vCrossScores, g.vCrossScores)
	copy(newg.hCrossScores, g.hCrossScores)
	copy(newg.vCrossSets, g.vCrossSets)
	copy(newg.hCrossSets, g.hCrossSets)
	copy(newg.vAnchors, g.vAnchors)
	copy(newg.hAnchors, g.hAnchors)

	newg.tilesPlayed = g.tilesPlayed
	newg.dim = g.dim
	newg.rowMul = g.rowMul
	newg.colMul = g.colMul
	return newg
}

func (g *GameBoard) RestoreFromCopy() {
	g.CopyFrom(g.lastCopy)
	g.lastCopy = nil
}

// CopyFrom copies the squares and other info from b back into g.
func (g *GameBoard) CopyFrom(b *GameBoard) {
	copy(g.squares, b.squares)
	copy(g.bonuses, b.bonuses)
	copy(g.vCrossScores, b.vCrossScores)
	copy(g.hCrossScores, b.hCrossScores)
	copy(g.vCrossSets, b.vCrossSets)
	copy(g.hCrossSets, b.hCrossSets)
	copy(g.vAnchors, b.vAnchors)
	copy(g.hAnchors, b.hAnchors)
	g.tilesPlayed = b.tilesPlayed
	g.dim = b.dim
	g.rowMul = b.rowMul
	g.colMul = b.colMul
}

func (g *GameBoard) GetSquares() []alphabet.MachineLetter {
	return g.squares
}

func (g *GameBoard) GetTilesPlayed() int {
	return g.tilesPlayed
}

func (g *GameBoard) TestSetTilesPlayed(n int) {
	g.tilesPlayed = n
}
