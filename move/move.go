package move

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/tilemapping"
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
	leave tilemapping.MachineWord
	score int

	rowStart    int
	colStart    int
	tilesPlayed int

	equity float64

	estimatedValue int16 // used only for endgames

	action        MoveType
	vertical      bool
	uglyBagOfData any

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

func (m *Move) Equals(o *Move, alsoCheckTransposition, ignoreLeave bool) bool {
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
	if !ignoreLeave && len(m.leave) != len(o.leave) {
		return false
	}
	for idx, i := range m.tiles {
		if o.tiles[idx] != i {
			return false
		}
	}
	if !ignoreLeave {
		for idx, i := range m.leave {
			if o.leave[idx] != i {
				return false
			}
		}
	}
	return true
}

func (m *Move) Set(tiles tilemapping.MachineWord, leave tilemapping.MachineWord, score int,
	rowStart, colStart, tilesPlayed int, vertical bool, action MoveType,
	alph *tilemapping.TileMapping) {

	m.tiles = tiles
	m.leave = leave
	m.score = score
	m.rowStart = rowStart
	m.colStart = colStart
	m.tilesPlayed = tilesPlayed
	m.vertical = vertical
	// everything else can be calculated.
	m.action = action
	m.alph = alph
}

func (m *Move) SetLeave(leave tilemapping.MachineWord) {
	m.leave = leave
}

func (m *Move) SetAction(action MoveType) {
	m.action = action
}

func (m *Move) SetAlphabet(alph *tilemapping.TileMapping) {
	m.alph = alph
}

// CopyFrom performs a copy of other.
func (m *Move) CopyFrom(other *Move) {
	m.action = other.action
	if cap(m.tiles) < len(other.tiles) {
		m.tiles = make([]tilemapping.MachineLetter, len(other.tiles))
	}
	m.tiles = m.tiles[:len(other.tiles)]
	if cap(m.leave) < len(other.leave) {
		m.leave = make([]tilemapping.MachineLetter, len(other.leave))
	}
	m.leave = m.leave[:len(other.leave)]
	copy(m.tiles, other.tiles)
	copy(m.leave, other.leave)
	m.alph = other.alph
	m.score = other.score

	m.rowStart = other.rowStart
	m.colStart = other.colStart
	m.tilesPlayed = other.tilesPlayed
	m.vertical = other.vertical

	m.estimatedValue = other.estimatedValue
	m.equity = other.equity
}

// String provides a string just for debugging purposes.
func (m *Move) String() string {
	switch m.action {
	case MoveTypePlay:
		return fmt.Sprintf(
			"<%p action: play word: %v %v score: %v tp: %v leave: %v equity: %.3f valu: %d>",
			m,
			m.BoardCoords(), m.TilesString(), m.score,
			m.tilesPlayed, m.LeaveString(), m.equity, m.estimatedValue)
	case MoveTypePass:
		return fmt.Sprintf("<%p action: pass leave: %v equity: %.3f valu: %d>",
			m,
			m.LeaveString(), m.equity, m.estimatedValue)
	case MoveTypeExchange:
		return fmt.Sprintf(
			"<%p action: exchange %v score: %v tp: %v leave: %v equity: %.3f>",
			m,
			m.TilesStringExchange(), m.score, m.tilesPlayed,
			m.LeaveString(), m.equity)
	case MoveTypeChallenge:
		return fmt.Sprintf("<%p action: challenge leave: %v equity: %.3f valu: %d>",
			m,
			m.LeaveString(), m.equity, m.estimatedValue)
	}
	return "<Unhandled move>"

}

func (m *Move) SetEmpty() {
	m.action = MoveTypeUnset
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

func (m *Move) LeaveString() string {
	return m.leave.UserVisible(m.alph)
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

func (m *Move) LessShortDescription() string {
	switch m.action {
	case MoveTypePlay:
		return fmt.Sprintf("%3v %s (%s)", m.BoardCoords(), m.TilesString(), m.LeaveString())
	case MoveTypePass:
		return fmt.Sprintf("(Pass) (%s)", m.LeaveString())
	case MoveTypeExchange:
		return fmt.Sprintf("(exch %s) (%s)", m.TilesStringExchange(), m.LeaveString())
	case MoveTypeChallenge:
		return "(Challenge!)"
	}
	return fmt.Sprint("UNHANDLED")
}

// FullRack returns the entire rack that the move was made from. This
// can be calculated from the tiles it uses and the leave.
func (m *Move) FullRack() string {

	rack := []tilemapping.MachineLetter{}
	for _, ml := range m.tiles {
		switch {
		case ml.IsBlanked():
			rack = append(rack, 0)
		case ml == 0:
			if m.action == MoveTypeExchange {
				// Only if you exchange the blank
				rack = append(rack, 0)
			}
			// Otherwise, don't add this to the rack representation. It
			// is a played-through marker.
		default:
			rack = append(rack, ml)
		}
	}
	for _, ml := range m.leave {
		rack = append(rack, ml)
	}
	sort.Slice(rack, func(i, j int) bool {
		return rack[i] < rack[j]
	})
	return tilemapping.MachineWord(rack).UserVisible(m.Alphabet())
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

func (m *Move) SetUglyBagOfData(b any) {
	m.uglyBagOfData = b
}

func (m *Move) UglyBagOfData() any {
	return m.uglyBagOfData
}

// NewScoringMove creates a scoring *Move and returns it.
func NewScoringMove(score int, tiles tilemapping.MachineWord,
	leave tilemapping.MachineWord, vertical bool, tilesPlayed int,
	alph *tilemapping.TileMapping, rowStart int, colStart int) *Move {

	move := &Move{
		action: MoveTypePlay, score: score, tiles: tiles, leave: leave, vertical: vertical,
		tilesPlayed: tilesPlayed, alph: alph,
		rowStart: rowStart, colStart: colStart,
	}
	return move
}

// NewScoringMoveSimple takes in user-visible strings. Consider moving to this
// (it is a little slower, though, so maybe only for tests)
func NewScoringMoveSimple(score int, coords string, word string, leave string,
	alph *tilemapping.TileMapping) *Move {

	row, col, vertical := FromBoardGameCoords(coords)

	tiles, err := tilemapping.ToMachineWord(word, alph)
	if err != nil {
		log.Error().Err(err).Msg("")
		return nil
	}
	leaveMW, err := tilemapping.ToMachineWord(leave, alph)
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
		leave:       leaveMW,
		vertical:    vertical,
		tilesPlayed: tilesPlayed,
		alph:        alph,
		rowStart:    row,
		colStart:    col,
	}
	return move
}

// NewExchangeMove creates an exchange.
func NewExchangeMove(tiles tilemapping.MachineWord, leave tilemapping.MachineWord,
	alph *tilemapping.TileMapping) *Move {
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

func NewUnsuccessfulChallengePassMove(leave tilemapping.MachineWord, alph *tilemapping.TileMapping) *Move {
	return &Move{
		action: MoveTypeUnsuccessfulChallengePass,
		leave:  leave,
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

// EstimatedValue is an internal value that is used in calculating endgames and related metrics.
func (m *Move) EstimatedValue() int16 {
	return m.estimatedValue
}

// SetEstimatedValue sets the estimated value of this move. It is calculated
// outside of this package.
func (m *Move) SetEstimatedValue(v int16) {
	m.estimatedValue = v
}

// AddEstimatedValue adds an estimate to the existing estimated value of this
// estimate. Estimate.
func (m *Move) AddEstimatedValue(v int16) {
	m.estimatedValue += v
}

func (m *Move) Score() int {
	return m.score
}

func (m *Move) Leave() tilemapping.MachineWord {
	return m.leave
}

func (m *Move) Tiles() tilemapping.MachineWord {
	return m.tiles
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

// NewPassMove creates a pass with the given leave.
func NewPassMove(leave tilemapping.MachineWord, alph *tilemapping.TileMapping) *Move {
	return &Move{
		action: MoveTypePass,
		leave:  leave,
		alph:   alph,
	}
}

// NewChallengeMove creates a challenge with the given leave.
func NewChallengeMove(leave tilemapping.MachineWord, alph *tilemapping.TileMapping) *Move {
	return &Move{
		action: MoveTypeChallenge,
		leave:  leave,
		alph:   alph,
	}
}
