// Package game encapsulates the main mechanics for a Crossword Game. It
// interacts heavily with the protobuf data structures.
package game

import (
	crypto_rand "crypto/rand"
	"encoding/binary"
	"math/rand"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	pb "github.com/domino14/macondo/rpc/api/proto"
	"github.com/lithammer/shortuuid"
)

// // History is a wrapper around pb.GameHistory to allow for additional
// // methods and an actual array of positions.
// type History struct {
// 	pb.GameHistory

// }

// Position is a wrapper around pb.GamePosition that has additional
// methods and values associated with it, such as the current board state.
type Position struct {
	pb.GamePosition

	board *board.GameBoard
	bag   *alphabet.Bag
	// The pb.GamePosition doesn't know how to deal with alphabet.Rack,
	// so we add an extra field here just for the alphabet rack,
	// mapped 1-to-1 to the pb.GamePosition.Players array.
	playerRacks []*alphabet.Rack
}

// Game is the actual internal game structure that controls the entire
// business logic of the game; drawing, making moves, etc. The two
// structures above are basically data entities.
// Note: a Game doesn't care how it is played. It is just rules for gameplay.
// AI players, human players, etc will play a game outside of the scope of
// this module.
type Game struct {
	gaddag  *gaddag.SimpleGaddag
	alph    *alphabet.Alphabet
	playing bool

	randSeed   int64
	randSource *rand.Rand

	history *pb.GameHistory
	// An array of Position, corresponding to the internal protobuf array
	// of pb.GameTurn.
	positions []*Position
}

// CalculateCoordsFromStringPosition turns a "position" on the board such as
// H7 and turns it into a numeric row, col, and direction.
func CalculateCoordsFromStringPosition(evt *pb.GameEvent) {
	// Note that this evt.Position has nothing to do with the type Position
	// we are defining in this package.
	row, col, vertical := move.FromBoardGameCoords(evt.Position)
	if vertical {
		evt.Direction = pb.GameEvent_VERTICAL
	} else {
		evt.Direction = pb.GameEvent_HORIZONTAL
	}
	evt.Row = int32(row)
	evt.Column = int32(col)
}

func newHistory(players []*pb.PlayerInfo) *History {
	his := &pb.GameHistory{}
	his.Players = players
	his.Uuid = shortuuid.New()
	return his
}

// NewGame is how one instantiates a brand new game. It instantiates a history,
// and deals tiles, creating the first position in the history as well.
func NewGame(cfg *config.Config, players []*pb.PlayerInfo, boardLayout []string,
	lexiconName string, letterDistributionName string) (*Game, error) {

	var b [8]byte
	_, err := crypto_rand.Read(b[:])
	if err != nil {
		panic("cannot seed math/rand package with cryptographically secure random number generator")
	}

	game := &Game{}
	game.randSeed = int64(binary.LittleEndian.Uint64(b[:]))
	game.randSource = rand.New(rand.NewSource(game.randSeed))
	game.history = newHistory(players)

	// Initialize a new position.
	position := &Position{}

	// 0. create a random seed?
	// 1. pick a random player to go first.
	// 2. set position onturn, turnnum, playing, board
	// 3. deal out tiles, set position bag to have 86 tiles in it
	// 4. set position playerRacks to be the player racks, and the position
	//   internal pb struct for the racks to be the string repr of these
	// 5. add position to game.positions
	// 6. return game.

	first := game.randSource.Intn(2)
	second := (first + 1) % 2

	// Flip the array if first is not 0.
	if first != 0 {
		players[first], players[second] = players[second], players[first]
	}

	game.board =

}
