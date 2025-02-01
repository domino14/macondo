package move

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
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

	MoveTypeUnset
)

// Move is a move. It can have a score, position, equity, etc. It doesn't
// have to be a scoring move.
type Move struct {
	// The ordering here should only be changed if it makes the structure smaller.
	// This Move should be kept as small as possible.
	tiles tilemapping.MachineWord
	score int

	rowStart    int
	colStart    int
	tilesPlayed int

	equity float64

	action   MoveType
	vertical bool

	alph *tilemapping.TileMapping
}

var reVertical, reHorizontal *regexp.Regexp

func init() {
	reVertical = regexp.MustCompile(`^(?P<col>[A-Za-z])(?P<row>[0-9]+)$`)
	reHorizontal = regexp.MustCompile(`^(?P<row>[0-9]+)(?P<col>[A-Za-z])$`)
}

func (m *Move) equalPositions(o *Move, alsoCheckTransposition bool) bool {
	if m.rowStart == o.rowStart && m.colStart == o.colStart && m.vertical == o.vertical {
		return true
	}
	if alsoCheckTransposition {
		if m.rowStart == o.colStart && m.colStart == o.rowStart && m.vertical != o.vertical {
			return true
		}
	}
	return false
}

func (m *Move) Equals(o *Move, alsoCheckTransposition bool) bool {
	if m.score != o.score {
		return false
	}
	if m.action != o.action {
		return false
	}
	if m.tilesPlayed != o.tilesPlayed {
		return false
	}
	if !m.equalPositions(o, alsoCheckTransposition) {
		return false
	}
	if len(m.tiles) != len(o.tiles) {
		return false
	}
	for idx, i := range m.tiles {
		if o.tiles[idx] != i {
			return false
		}
	}

	return true
}

func (m *Move) Set(tiles tilemapping.MachineWord, score int,
	rowStart, colStart, tilesPlayed int, vertical bool, action MoveType,
	alph *tilemapping.TileMapping) {

	m.tiles = tiles
	m.score = score
	m.rowStart = rowStart
	m.colStart = colStart
	m.tilesPlayed = tilesPlayed
	m.vertical = vertical
	// everything else can be calculated.
	m.action = action
	m.alph = alph
}

func (m *Move) SetAction(action MoveType) {
	m.action = action
}

func (m *Move) SetAlphabet(alph *tilemapping.TileMapping) {
	m.alph = alph
}

func (m *Move) SetScore(s int) {
	m.score = s
}

// CopyFrom performs a copy of other.
func (m *Move) CopyFrom(other *Move) {
	m.action = other.action
	if cap(m.tiles) < len(other.tiles) {
		m.tiles = make([]tilemapping.MachineLetter, len(other.tiles))
	}
	m.tiles = m.tiles[:len(other.tiles)]
	copy(m.tiles, other.tiles)
	m.alph = other.alph
	m.score = other.score

	m.rowStart = other.rowStart
	m.colStart = other.colStart
	m.tilesPlayed = other.tilesPlayed
	m.vertical = other.vertical

	m.equity = other.equity
}

// String provides a string just for debugging purposes.
func (m *Move) String() string {
	switch m.action {
	case MoveTypePlay:
		return fmt.Sprintf(
			"<%p action: play word: %v %v score: %v tp: %v equity: %.3f>",
			m,
			m.BoardCoords(), m.TilesString(), m.score,
			m.tilesPlayed, m.equity)
	case MoveTypePass:
		return fmt.Sprintf("<%p action: pass equity: %.3f>",
			m,
			m.equity)
	case MoveTypeExchange:
		return fmt.Sprintf(
			"<%p action: exchange %v score: %v tp: %v equity: %.3f>",
			m,
			m.TilesStringExchange(), m.score, m.tilesPlayed,
			m.equity)
	case MoveTypeChallenge:
		return fmt.Sprintf("<%p action: challenge equity: %.3f>",
			m,
			m.equity)
	}
	return "<Unhandled move>"

}

func (m *Move) SetEmpty() {
	m.action = MoveTypeUnset
}

func (m *Move) IsEmpty() bool {
	return m.action == MoveTypeUnset
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
	return m.tiles.UserVisiblePlayedTiles(m.alph)
}

func (m *Move) TilesStringExchange() string {
	return m.tiles.UserVisible(m.alph)
}

// ShortDescription provides a short description, useful for logging or
// user display.
func (m *Move) ShortDescription() string {
	switch m.action {
	case MoveTypePlay:
		return fmt.Sprintf("%3v %s", m.BoardCoords(), m.TilesString())
	case MoveTypePass:
		return "(Pass)"
	case MoveTypeExchange:
		return fmt.Sprintf("(exch %s)", m.TilesStringExchange())
	case MoveTypeChallenge:
		return "(Challenge!)"
	}
	return fmt.Sprint("UNHANDLED")
}

func (m *Move) Action() MoveType {
	return m.action
}

// TilesPlayed returns the number of tiles played by this move.
func (m *Move) TilesPlayed() int {
	return m.tilesPlayed
}

func (m *Move) BingoPlayed() bool {
	return (m.action == MoveTypePlay) && (m.tilesPlayed == 7)
}

// NewScoringMove creates a scoring *Move and returns it.
func NewScoringMove(score int, tiles tilemapping.MachineWord,
	vertical bool, tilesPlayed int,
	alph *tilemapping.TileMapping, rowStart int, colStart int) *Move {

	move := &Move{
		action: MoveTypePlay, score: score, tiles: tiles, vertical: vertical,
		tilesPlayed: tilesPlayed, alph: alph,
		rowStart: rowStart, colStart: colStart,
	}
	return move
}

// NewScoringMoveSimple takes in user-visible strings. Consider moving to this
// (it is a little slower, though, so maybe only for tests)
func NewScoringMoveSimple(score int, coords string, word string,
	alph *tilemapping.TileMapping) *Move {

	row, col, vertical := FromBoardGameCoords(coords)

	tiles, err := tilemapping.ToMachineWord(word, alph)
	if err != nil {
		log.Error().Err(err).Msg("")
		return nil
	}
	tilesPlayed := 0
	for _, t := range tiles {
		if t != 0 {
			tilesPlayed++
		}
	}

	move := &Move{
		action:      MoveTypePlay,
		score:       score,
		tiles:       tiles,
		vertical:    vertical,
		tilesPlayed: tilesPlayed,
		alph:        alph,
		rowStart:    row,
		colStart:    col,
	}
	return move
}

// NewExchangeMove creates an exchange.
func NewExchangeMove(tiles tilemapping.MachineWord, alph *tilemapping.TileMapping) *Move {
	move := &Move{
		action:      MoveTypeExchange,
		score:       0,
		tiles:       tiles,
		tilesPlayed: len(tiles), // tiles exchanged, really..
		alph:        alph,
	}
	return move
}

func NewBonusScoreMove(t MoveType, tiles tilemapping.MachineWord, score int) *Move {
	move := &Move{
		action: t,
		score:  score,
		tiles:  tiles,
	}
	return move
}

func NewLostScoreMove(t MoveType, rack tilemapping.MachineWord, score int) *Move {
	move := &Move{
		action: t,
		tiles:  rack,
		score:  -score,
	}
	return move
}

func NewUnsuccessfulChallengePassMove(alph *tilemapping.TileMapping) *Move {
	return &Move{
		action: MoveTypeUnsuccessfulChallengePass,
		alph:   alph,
	}
}

// Alphabet is the alphabet used by this move
func (m *Move) Alphabet() *tilemapping.TileMapping {
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

func (m *Move) Score() int {
	return m.score
}

func (m *Move) Tiles() tilemapping.MachineWord {
	return m.tiles
}

func (m *Move) PlayLength() int {
	return len(m.tiles)
}

func (m *Move) CoordsAndVertical() (int, int, bool) {
	return m.rowStart, m.colStart, m.vertical
}

func (m *Move) BoardCoords() string {
	return ToBoardGameCoords(m.rowStart, m.colStart, m.vertical)
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
		col = int(strings.ToUpper(vMatches[1])[0] - 'A')
		vertical = true
		return row - 1, col, vertical
	}
	hMatches := reHorizontal.FindStringSubmatch(c)
	if len(hMatches) == 3 {
		row, _ = strconv.Atoi(hMatches[1])
		col = int(strings.ToUpper(hMatches[2])[0] - 'A')
		vertical = false
		return row - 1, col, vertical
	}

	return 0, 0, false
}

// NewPassMove creates a pass
func NewPassMove(alph *tilemapping.TileMapping) *Move {
	return &Move{
		action: MoveTypePass,
		alph:   alph,
	}
}

// NewChallengeMove creates a challenge
func NewChallengeMove(alph *tilemapping.TileMapping) *Move {
	return &Move{
		action: MoveTypeChallenge,
		alph:   alph,
	}
}

func MinimallyEqual(m1 *Move, m2 *Move) bool {
	if m1.Action() != m2.Action() {
		return false
	}
	if m1.TilesPlayed() != m2.TilesPlayed() {
		return false
	}
	if len(m1.Tiles()) != len(m2.Tiles()) {
		return false
	}
	r1, c1, v1 := m1.CoordsAndVertical()
	r2, c2, v2 := m2.CoordsAndVertical()
	if r1 != r2 || c1 != c2 || v1 != v2 {
		return false
	}
	for idx, i := range m1.Tiles() {
		if m2.Tiles()[idx] != i {
			return false
		}
	}
	return true
}
