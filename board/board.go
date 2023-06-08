package board

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
	"github.com/rs/zerolog/log"
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
	squares     []tilemapping.MachineLetter
	bonuses     []BonusSquare
	tilesPlayed int
	dim         int
	lastCopy    *GameBoard

	// Store cross-scores with the board to avoid recalculating, but cross-sets
	// are a movegen detail and do not belong here!
	vCrossScores []int
	hCrossScores []int
	// The rest of these are definitely movegen details and they
	// really should not be here. However, let's do this one step at a time.
	hCrossSets []CrossSet
	vCrossSets []CrossSet
	hAnchors   []bool
	vAnchors   []bool

	rowMul int
	colMul int
}

// MakeBoard creates a board from a description string.
// Assumption: strings are ASCII.
func MakeBoard(desc []string) *GameBoard {
	// Turns an array of strings into the GameBoard structure type.
	// Assume all strings are the same length
	totalLen := 0
	for _, s := range desc {
		totalLen += len(s)
	}
	sqs := make([]tilemapping.MachineLetter, totalLen)
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
			sqs[sqi] = 0
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
	// Call Clear to set all crosses.
	g.Clear()
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
	// (I feel ok making this assumption for now)
	return g.bonuses[g.getSqIdx(row, col)]
}

func (g *GameBoard) SetLetter(row int, col int, letter tilemapping.MachineLetter) {
	g.squares[g.getSqIdx(row, col)] = letter
}

func (g *GameBoard) GetLetter(row int, col int) tilemapping.MachineLetter {
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

func (g *GameBoard) GetCrossSet(row int, col int, dir BoardDirection) CrossSet {
	pos := g.getSqIdx(row, col)

	switch dir {
	case HorizontalDirection:
		return g.hCrossSets[pos]
	case VerticalDirection:
		return g.vCrossSets[pos]
	default:
		log.Error().Msgf("Unknown direction: %v\n", dir)
		return 0
	}
}

func (g *GameBoard) ClearCrossSet(row int, col int, dir BoardDirection) {
	pos := g.getSqIdx(row, col)
	switch dir {
	case HorizontalDirection:
		g.hCrossSets[pos] = 0
	case VerticalDirection:
		g.vCrossSets[pos] = 0
	default:
		log.Error().Msgf("Unknown direction: %v\n", dir)
	}
}

func (g *GameBoard) SetCrossSetLetter(row int, col int, dir BoardDirection,
	ml tilemapping.MachineLetter) {
	pos := g.getSqIdx(row, col)
	switch dir {
	case HorizontalDirection:
		g.hCrossSets[pos].Set(ml)
	case VerticalDirection:
		g.vCrossSets[pos].Set(ml)
	default:
		log.Error().Msgf("Unknown direction: %v\n", dir)
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

// SetAllCrosses sets the cross sets of every square to every acceptable letter.
func (g *GameBoard) SetAllCrosses() {
	for i := range g.hCrossScores {
		g.hCrossSets[i].SetAll()
	}
	for i := range g.vCrossScores {
		g.vCrossSets[i].SetAll()
	}
}

// ClearAllCrosses disallows all letters on all squares (more or less).
func (g *GameBoard) ClearAllCrosses() {
	for i := range g.hCrossScores {
		g.hCrossSets[i].Clear()
	}
	for i := range g.vCrossScores {
		g.vCrossSets[i].Clear()
	}
}

func (g *GameBoard) HasLetter(row int, col int) bool {
	return g.GetLetter(row, col) != 0
}

// Clear clears the board.
func (g *GameBoard) Clear() {
	for i := 0; i < len(g.squares); i++ {
		g.squares[i] = 0
	}
	g.tilesPlayed = 0
	// We set all crosses because every letter is technically allowed
	// on every cross-set at the very beginning.
	g.SetAllCrosses()
	g.ResetCrossScores()
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
	pos := g.getSqIdx(row, col)
	g.hAnchors[pos] = false
	g.vAnchors[pos] = false
	var tileAbove, tileBelow, tileLeft, tileRight, tileHere bool
	if row > 0 {
		tileAbove = g.HasLetter(row-1, col)
	}
	if col > 0 {
		tileLeft = g.HasLetter(row, col-1)
	}
	if row < g.Dim()-1 {
		tileBelow = g.HasLetter(row+1, col)
	}
	if col < g.Dim()-1 {
		tileRight = g.HasLetter(row, col+1)
	}
	tileHere = g.HasLetter(row, col)
	if tileHere {
		// The current square is not empty. It should only be an anchor
		// if it is the rightmost square of a word (actually, squares to
		// the left are probably ok, but not the leftmost square. Note
		// Gordon does not have this requirement, but the algorithm does
		// not work if we don't do this)
		if !tileRight {
			g.hAnchors[pos] = true
		}
		// Apply the transverse logic too for the vertical anchor.
		if !tileBelow {
			g.vAnchors[pos] = true
		}
	} else {
		// If the square is empty, it should only be an anchor if the
		// squares to its left and right are empty, and at least one of
		// the squares in the top and bottom are NOT empty.
		if !tileLeft && !tileRight && (tileAbove || tileBelow) {
			g.hAnchors[pos] = true
		}
		// (And apply the transverse logic for the vertical anchor)
		if !tileAbove && !tileBelow && (tileLeft || tileRight) {
			g.vAnchors[pos] = true
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
				pos := g.getSqIdx(i, j)
				g.hAnchors[pos] = false
				g.vAnchors[pos] = false
			}
		}
		rc := int(n / 2)
		// If the board is empty, set just one anchor, in the center square.
		g.hAnchors[g.getSqIdx(rc, rc)] = true
	}
}

// IsAnchor returns whether the row/col pair is an anchor in the given
// direction.
func (g *GameBoard) IsAnchor(row int, col int, dir BoardDirection) bool {
	pos := g.getSqIdx(row, col)
	switch dir {
	case HorizontalDirection:
		return g.hAnchors[pos]
	case VerticalDirection:
		return g.vAnchors[pos]

	default:
		log.Error().Msgf("Unknown direction: %v\n", dir)
	}
	return false
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

func (g *GameBoard) TraverseBackwardsForScore(row int, col int, ld *tilemapping.LetterDistribution) int {
	score := 0
	for g.PosExists(row, col) {
		ml := g.GetLetter(row, col)
		if ml == 0 {
			break
		}
		score += ld.Score(ml)
		col--
	}
	return score
}

func (g *GameBoard) updateAnchorsForMove(m move.PlayMaker) {
	row := m.RowStart()
	col := m.ColStart()
	vertical := m.Vertical()
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

func (g *GameBoard) PlaceMoveTiles(m move.PlayMaker) {
	rowStart := m.RowStart()
	colStart := m.ColStart()
	vertical := m.Vertical()
	var row, col int
	for idx, tile := range m.Tiles() {
		if tile == 0 {
			continue
		}
		if vertical {
			row = rowStart + idx
			col = colStart
		} else {
			col = colStart + idx
			row = rowStart
		}
		g.squares[g.getSqIdx(row, col)] = tile
	}
}

func (g *GameBoard) UnplaceMoveTiles(m *move.Move) {
	rowStart, colStart, vertical := m.CoordsAndVertical()
	var row, col int
	for idx, tile := range m.Tiles() {
		if tile == 0 {
			continue
		}
		if vertical {
			row = rowStart + idx
			col = colStart
		} else {
			col = colStart + idx
			row = rowStart
		}
		g.squares[g.getSqIdx(row, col)] = 0
	}
}

// PlayMove plays a move on a board. It must place tiles on the board,
// regenerate cross-sets and cross-points, and recalculate anchors.
func (g *GameBoard) PlayMove(m move.PlayMaker, ld *tilemapping.LetterDistribution) {

	// g.playHistory = append(g.playHistory, m.ShortDescription())
	if m.Type() != move.MoveTypePlay {
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
	word tilemapping.MachineWord) error {

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

		if ml == 0 {
			ml = g.GetLetter(newrow, newcol)
			if ml == 0 {
				return errors.New("a played-through marker was specified, but " +
					"there is no tile at the given location")
			}
			bordersATile = true
		} else {
			ml = g.GetLetter(newrow, newcol)
			if ml != 0 {
				return fmt.Errorf("tried to play through a letter already on "+
					"the board; please use the played-through marker (.) instead "+
					"(row %v col %v ml %v)", newrow, newcol, ml)
			}

			// We are placing a tile on this empty square. Check if we border
			// any other tiles.

			for d := -1; d <= 1; d += 2 {
				// only check perpendicular hooks
				checkrow, checkcol := newrow+ci*d, newcol+ri*d
				if g.PosExists(checkrow, checkcol) && g.GetLetter(checkrow, checkcol) != 0 {
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
		if g.PosExists(checkrow, checkcol) && g.GetLetter(checkrow, checkcol) != 0 {
			return errors.New("your play must include the whole word")
		}
	}
	{
		checkrow, checkcol := row+ri*len(word), col+ci*len(word)
		if g.PosExists(checkrow, checkcol) && g.GetLetter(checkrow, checkcol) != 0 {
			return errors.New("your play must include the whole word")
		}
	}
	return nil
}

// FormedWords returns an array of all machine words formed by this move.
// The move is assumed to be of type Play
func (g *GameBoard) FormedWords(m move.PlayMaker) ([]tilemapping.MachineWord, error) {
	// Reserve space for main word.
	words := []tilemapping.MachineWord{nil}
	mainWord := []tilemapping.MachineLetter{}

	row, col, vertical := m.RowStart(), m.ColStart(), m.Vertical()
	ri, ci := 0, 1
	if vertical {
		ri, ci = ci, ri
	}

	if m.Type() != move.MoveTypePlay {
		return nil, errors.New("function must be called with a tile placement play")
	}

	for idx, letter := range m.Tiles() {
		// For the purpose of checking words, all letters should be unblanked.
		letter = letter.Unblank()
		newrow, newcol := row+(ri*idx), col+(ci*idx)

		// This is the main word.
		if letter == 0 {
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

func (g *GameBoard) formedCrossWord(crossVertical bool, letter tilemapping.MachineLetter,
	row, col int) tilemapping.MachineWord {

	ri, ci := 0, 1
	if crossVertical {
		ri, ci = ci, ri
	}

	// Given the cross-word direction (crossVertical) and a letter located at row, col
	// find the cross-word that contains this letter (if any)
	// Look in the cross direction for newly played tiles.
	crossword := []tilemapping.MachineLetter{}

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

// ScoreWord scores the move at the given row and column. Note that this
// function is called when the board is potentially transposed, so we
// assume the row stays static as we iterate through the letters of the
// word.
func (g *GameBoard) ScoreWord(word tilemapping.MachineWord, row, col, tilesPlayed int,
	crossDir BoardDirection, ld *tilemapping.LetterDistribution) int {

	// letterScore:
	var ls int

	mainWordScore := 0
	crossScores := 0
	bingoBonus := 0
	if tilesPlayed == 7 {
		bingoBonus = 50
	}
	wordMultiplier := 1

	for idx, ml := range word {
		bonusSq := g.GetBonus(row, col+idx)
		letterMultiplier := 1
		thisWordMultiplier := 1
		freshTile := false
		if ml == 0 {
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
		if ml.IsBlanked() {
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
	newg.squares = make([]tilemapping.MachineLetter, len(g.squares))
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
	// newg.playHistory = append([]string{}, g.playHistory...)
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
	g.rowMul = b.rowMul
	g.colMul = b.colMul
}

func (g *GameBoard) GetSquares() []tilemapping.MachineLetter {
	return g.squares
}

func (g *GameBoard) GetTilesPlayed() int {
	return g.tilesPlayed
}

func (g *GameBoard) TestSetTilesPlayed(n int) {
	g.tilesPlayed = n
}

// ToFEN converts the game board to a FEN string, which is the board component
// of the CGP data format. See cgp directory for more info.
func (g *GameBoard) ToFEN(alph *tilemapping.TileMapping) string {
	var bd strings.Builder
	for i := 0; i < g.dim; i++ {
		var r strings.Builder
		zeroCt := 0
		for j := 0; j < g.dim; j++ {
			l := g.GetLetter(i, j)
			if l == 0 {
				zeroCt++
				continue
			}
			// Otherwise, it's a letter.
			if zeroCt > 0 {
				r.WriteString(strconv.Itoa(zeroCt))
				zeroCt = 0
			}
			r.WriteString(l.UserVisible(alph, false))
		}
		if zeroCt > 0 {
			r.WriteString(strconv.Itoa(zeroCt))
		}
		bd.WriteString(r.String())
		if i != g.dim-1 {
			bd.WriteString("/")
		}
	}
	return bd.String()
}
