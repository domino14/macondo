package move

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
)

// MoveType is a type of move; a play, an exchange, pass, etc.
type MoveType uint8

const (
	MoveTypePlay MoveType = iota
	MoveTypeExchange
	MoveTypePass
	MoveTypeChallenge
	MoveTypePhonyTilesReturned
	MoveTypeChallengeBonus
	MoveTypeUnsuccessfulChallengePass

	MoveTypeEndgameTiles
	MoveTypeLostTileScore
	MoveTypeLostScoreOnTime
)

// Move is a move. It can have a score, position, equity, etc. It doesn't
// have to be a scoring move.
type Move struct {
	action      MoveType
	score       int
	equity      float64
	desc        string
	coords      string //25 so far
	tiles       alphabet.MachineWord
	leave       alphabet.MachineWord
	rowStart    int
	colStart    int
	vertical    bool
	bingo       bool
	tilesPlayed int
	alph        *alphabet.Alphabet
	valuation   float32
	// visited is used during endgame iterative deepening.
	visited bool
	// If this move has a duplicate, it will be here. It only applies
	// for single-tile plays.
	dupeOf *Move
}

var reVertical, reHorizontal *regexp.Regexp

func init() {
	reVertical = regexp.MustCompile(`^(?P<col>[A-Z])(?P<row>[0-9]+)$`)
	reHorizontal = regexp.MustCompile(`^(?P<row>[0-9]+)(?P<col>[A-Z])$`)
}

// String provides a string just for debugging purposes.
func (m *Move) String() string {
	switch m.action {
	case MoveTypePlay:
		return fmt.Sprintf(
			"<%p action: play word: %v %v score: %v tp: %v leave: %v equity: %.3f valu: %.3f>",
			m,
			m.coords, m.TilesString(), m.score,
			m.tilesPlayed, m.LeaveString(), m.equity, m.valuation)
	case MoveTypePass:
		return fmt.Sprintf("<%p action: pass leave: %v equity: %.3f valu: %.3f>",
			m,
			m.LeaveString(), m.equity, m.valuation)
	case MoveTypeExchange:
		return fmt.Sprintf(
			"<%p action: exchange %v score: %v tp: %v leave: %v equity: %.3f>",
			m,
			m.TilesString(), m.score, m.tilesPlayed,
			m.LeaveString(), m.equity)
	case MoveTypeChallenge:
		return fmt.Sprintf("<%p action: challenge leave: %v equity: %.3f valu: %.3f>",
			m,
			m.LeaveString(), m.equity, m.valuation)
	}
	return fmt.Sprint("<Unhandled move>")

}

func (m *Move) MoveTypeString() string {
	// Return the moveype as a string
	switch m.action {
	case MoveTypePlay:
		return "Play"
	case MoveTypePass:
		return "Pass"
	case MoveTypeExchange:
		return "Exchange"
	case MoveTypeChallenge:
		return "Challenge"
	}
	return fmt.Sprint("UNHANDLED")
}

func (m *Move) TilesString() string {
	return m.tiles.UserVisible(m.alph)
}

func (m *Move) LeaveString() string {
	return m.leave.UserVisible(m.alph)
}

// ShortDescription provides a short description, useful for logging or
// user display.
func (m *Move) ShortDescription() string {
	switch m.action {
	case MoveTypePlay:
		return fmt.Sprintf("%v %v", m.coords, m.TilesString())
	case MoveTypePass:
		return "(Pass)"
	case MoveTypeExchange:
		return fmt.Sprintf("(exch %v)", m.TilesString())
	case MoveTypeChallenge:
		return "(Challenge!)"
	}
	return fmt.Sprint("UNHANDLED")
}

// FullRack returns the entire rack that the move was made from. This
// can be calculated from the tiles it uses and the leave.
func (m *Move) FullRack() string {
	rack := []rune(m.LeaveString())
	for _, ml := range m.tiles {
		switch {
		case ml >= alphabet.BlankOffset:
			rack = append(rack, alphabet.BlankToken)
		case ml == alphabet.BlankMachineLetter:
			// Only if you exchange the blank
			rack = append(rack, alphabet.BlankToken)
		case ml == alphabet.PlayedThroughMarker || ml == alphabet.EmptySquareMarker:
			// do nothing

		default:
			rack = append(rack, m.alph.Letter(ml))
		}
	}
	sort.Slice(rack, func(i, j int) bool {
		return rack[i] < rack[j]
	})
	return string(rack)
}

func (m *Move) Action() MoveType {
	return m.action
}

// TilesPlayed returns the number of tiles played by this move.
func (m *Move) TilesPlayed() int {
	return m.tilesPlayed
}

// NewScoringMove creates a scoring *Move and returns it.
func NewScoringMove(score int, tiles alphabet.MachineWord,
	leave alphabet.MachineWord, vertical bool, tilesPlayed int,
	alph *alphabet.Alphabet, rowStart int, colStart int, coords string) *Move {

	move := &Move{
		action: MoveTypePlay, score: score, tiles: tiles, leave: leave, vertical: vertical,
		bingo: tilesPlayed == 7, tilesPlayed: tilesPlayed, alph: alph,
		rowStart: rowStart, colStart: colStart, coords: coords,
	}
	return move
}

// NewScoringMoveSimple takes in user-visible strings. Consider moving to this
// (it is a little slower, though, so maybe only for tests)
func NewScoringMoveSimple(score int, coords string, word string, leave string,
	alph *alphabet.Alphabet) *Move {

	row, col, vertical := FromBoardGameCoords(coords)

	tiles, err := alphabet.ToMachineWord(word, alph)
	if err != nil {
		log.Error().Err(err).Msg("")
		return nil
	}
	leaveMW, err := alphabet.ToMachineWord(leave, alph)
	if err != nil {
		log.Error().Err(err).Msg("")
		return nil
	}
	tilesPlayed := 0
	for _, t := range tiles {
		if t != alphabet.PlayedThroughMarker {
			tilesPlayed++
		}
	}

	move := &Move{
		action:      MoveTypePlay,
		score:       score,
		tiles:       tiles,
		leave:       leaveMW,
		vertical:    vertical,
		bingo:       tilesPlayed == 7,
		tilesPlayed: tilesPlayed,
		alph:        alph,
		rowStart:    row,
		colStart:    col,
		coords:      coords,
	}
	return move
}

// NewExchangeMove creates an exchange.
func NewExchangeMove(tiles alphabet.MachineWord, leave alphabet.MachineWord,
	alph *alphabet.Alphabet) *Move {
	move := &Move{
		action:      MoveTypeExchange,
		score:       0,
		tiles:       tiles,
		leave:       leave,
		tilesPlayed: len(tiles), // tiles exchanged, really..
		alph:        alph,
	}
	return move
}

func NewBonusScoreMove(t MoveType, tiles alphabet.MachineWord, score int) *Move {
	move := &Move{
		action: t,
		score:  score,
		tiles:  tiles,
	}
	return move
}

func NewLostScoreMove(t MoveType, rack alphabet.MachineWord, score int) *Move {
	move := &Move{
		action: t,
		tiles:  rack,
		score:  -score,
	}
	return move
}

func NewUnsuccessfulChallengePassMove(leave alphabet.MachineWord, alph *alphabet.Alphabet) *Move {
	return &Move{
		action: MoveTypeUnsuccessfulChallengePass,
		leave:  leave,
		alph:   alph,
	}
}

// Alphabet is the alphabet used by this move
func (m *Move) Alphabet() *alphabet.Alphabet {
	return m.alph
}

// Equity is the equity of this move.
func (m *Move) Equity() float64 {
	return m.equity
}

// SetEquity sets the equity of this move. It is calculated outside this package.
func (m *Move) SetEquity(e float64) {
	m.equity = e
}

// Valuation is the "value" of this move. This is an internal value that is used
// in calculating endgames and other such metrics.
func (m *Move) Valuation() float32 {
	return m.valuation
}

// SetValuation sets the valuation of this move. It is calculated outside of this package.
func (m *Move) SetValuation(v float32) {
	m.valuation = v
}

func (m *Move) Score() int {
	return m.score
}

func (m *Move) SetDupe(o *Move) {
	m.dupeOf = o
}

func (m *Move) HasDupe() bool {
	return m.dupeOf != nil
}

func (m *Move) Dupe() *Move {
	return m.dupeOf
}

func (m *Move) SetVisited(v bool) { m.visited = v }
func (m *Move) Visited() bool     { return m.visited }

func (m *Move) Leave() alphabet.MachineWord {
	return m.leave
}

func (m *Move) Tiles() alphabet.MachineWord {
	return m.tiles
}

func (m *Move) UniqueSingleTileKey() int {
	// Find the tile.
	var idx int
	var ml alphabet.MachineLetter
	for idx, ml = range m.tiles {
		if ml != alphabet.PlayedThroughMarker {
			break
		}
	}

	var row, col int
	row = m.rowStart
	col = m.colStart
	// We want to get the coordinate of the tile that is on the board itself.
	if m.vertical {
		row += idx
	} else {
		col += idx
	}
	// A unique, fast to compute key for this play.
	return row + alphabet.MaxAlphabetSize*col +
		alphabet.MaxAlphabetSize*alphabet.MaxAlphabetSize*int(ml)
}

func (m *Move) CoordsAndVertical() (int, int, bool) {
	return m.rowStart, m.colStart, m.vertical
}

func (m *Move) BoardCoords() string {
	return m.coords
}

// ToBoardGameCoords onverts the row, col, and orientation of the play to
// a coordinate like 5F or G4.
func ToBoardGameCoords(row int, col int, vertical bool) string {
	colCoords := string(rune('A' + col))
	rowCoords := strconv.Itoa(int(row + 1))
	var coords string
	if vertical {
		coords = colCoords + rowCoords
	} else {
		coords = rowCoords + colCoords
	}
	return coords
}

// FromBoardGameCoords does the inverse operation of ToBoardGameCoords above.
func FromBoardGameCoords(c string) (int, int, bool) {
	vMatches := reVertical.FindStringSubmatch(c)
	var row, col int
	var vertical bool
	if len(vMatches) == 3 {
		// It's vertical
		row, _ = strconv.Atoi(vMatches[2])
		col = int(vMatches[1][0] - 'A')
		vertical = true
		return row - 1, col, vertical
	}
	hMatches := reHorizontal.FindStringSubmatch(c)
	if len(hMatches) == 3 {
		row, _ = strconv.Atoi(hMatches[1])
		col = int(hMatches[2][0] - 'A')
		vertical = false
		return row - 1, col, vertical
	}

	return 0, 0, false
}

// NewPassMove creates a pass with the given leave.
func NewPassMove(leave alphabet.MachineWord, alph *alphabet.Alphabet) *Move {
	return &Move{
		action: MoveTypePass,
		leave:  leave,
		alph:   alph,
	}
}

// NewChallengeMove creates a challenge with the given leave.
func NewChallengeMove(leave alphabet.MachineWord, alph *alphabet.Alphabet) *Move {
	return &Move{
		action: MoveTypeChallenge,
		leave:  leave,
		alph:   alph,
	}
}
