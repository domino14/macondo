// Package game encapsulates the main mechanics for a Crossword Game. It
// interacts heavily with the protobuf data structures.
package game

import (
	crypto_rand "crypto/rand"
	"encoding/binary"
	"math/rand"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	pb "github.com/domino14/macondo/rpc/api/proto"
	"github.com/lithammer/shortuuid"
	"github.com/rs/zerolog/log"
)

// RuleDefiner is an interface that is used for passing a set of rules
// to a game.
type RuleDefiner interface {
	Gaddag() *gaddag.SimpleGaddag
	Board() *board.GameBoard
	Bag() *alphabet.Bag
}

// // History is a wrapper around pb.GameHistory to allow for additional
// // methods and an actual array of positions.
// type History struct {
// 	pb.GameHistory

// }

// Position is a wrapper around pb.GamePosition that has additional
// methods and values associated with it, such as the current board state.
// type Position struct {
// 	pb.GamePosition

// 	// The pb.GamePosition doesn't know how to deal with alphabet.Rack,
// 	// so we add an extra field here just for the alphabet rack,
// 	// mapped 1-to-1 to the pb.GamePosition.Players array.
// 	playerRacks []*alphabet.Rack
// }

type playerState struct {
	pb.PlayerInfo

	rack        *alphabet.Rack
	rackLetters string
	points      int
}

func (p *playerState) resetScore() {
	p.points = 0
}

func (p *playerState) SetRackTiles(tiles []alphabet.MachineLetter, alph *alphabet.Alphabet) {
	p.rack.Set(tiles)
	p.rackLetters = alphabet.MachineWord(tiles).UserVisible(alph)
}

type playerStates []*playerState

func (p playerStates) resetScore() {
	for idx := range p {
		p[idx].resetScore()
	}
}

// Game is the actual internal game structure that controls the entire
// business logic of the game; drawing, making moves, etc. The two
// structures above are basically data entities.
// Note: a Game doesn't care how it is played. It is just rules for gameplay.
// AI players, human players, etc will play a game outside of the scope of
// this module.
type Game struct {
	gaddag *gaddag.SimpleGaddag
	alph   *alphabet.Alphabet
	// board and bag will contain the latest (current) versions of these.
	board *board.GameBoard
	bag   *alphabet.Bag

	playing bool

	randSeed   int64
	randSource *rand.Rand

	// Although these are in the position, we keep track of them here for
	// ease in backing up and restoring state. We could write logic to
	// unload / load these from the last Position but that is annoying
	// and error-prone.
	scorelessTurns int
	onturn         int
	turnnum        int
	players        playerStates

	history *pb.GameHistory
	// An array of Position, corresponding to the internal protobuf array
	// of GameHistory's GameTurn.
	// positions []*pb.GamePosition
	// playerRacks are the latest known versions of the player racks.
	// playerRacks []*alphabet.Rack

	stateStack []*stateBackup
	stackPtr   int
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

func newHistory(players []*pb.PlayerInfo) *pb.GameHistory {
	his := &pb.GameHistory{}
	his.Players = players
	his.Uuid = shortuuid.New()
	return his
}

// NewGame is how one instantiates a brand new game. It instantiates a history,
// and deals tiles, creating the first position in the history as well, and
// essentially starting the game.
func NewGame(rules RuleDefiner, playerinfo []*pb.PlayerInfo) (*Game, error) {

	var b [8]byte
	_, err := crypto_rand.Read(b[:])
	if err != nil {
		panic("cannot seed math/rand package with cryptographically secure random number generator")
	}

	game := &Game{}
	game.gaddag = rules.Gaddag()
	game.alph = game.gaddag.GetAlphabet()
	game.randSeed = int64(binary.LittleEndian.Uint64(b[:]))
	game.randSource = rand.New(rand.NewSource(game.randSeed))
	game.history = newHistory(playerinfo)
	game.bag = rules.Bag().Copy()
	game.board = rules.Board().Copy()

	// Initialize a new position.
	// position := &pb.GamePosition{}

	// 0. create a random seed?
	// 1. pick a random player to go first.
	// 2. set position onturn, turnnum, playing, board
	// 3. deal out tiles, set position bag to have 86 tiles in it
	// 4. set position playerRacks to be the player racks, and the position
	//   internal pb struct for the racks to be the string repr of these
	// 5. add position to game.positions
	// 6. return game

	first := game.randSource.Intn(2)
	second := (first + 1) % 2

	// Flip the array if first is not 0.
	if first != 0 {
		playerinfo[first], playerinfo[second] = playerinfo[second], playerinfo[first]
		playerinfo[first].Number = 1
		playerinfo[second].Number = 2
	}

	// Player index 0 always starts. The above flip ensures that player is not
	// always the first one passed into this function.
	// position.Onturn = 0
	// position.Playing = true
	// position.Turnnum = 0
	// position.Players = make([]*pb.PlayerState, len(playerinfo))

	game.players = make([]*playerState, len(playerinfo))

	for i := 0; i < len(playerinfo); i++ {
		tiles, _ := game.bag.Draw(7)
		pstate := &playerState{}
		pstate.Nickname = playerinfo[i].Nickname
		pstate.RealName = playerinfo[i].RealName
		pstate.Number = playerinfo[i].Number
		pstate.rack = alphabet.NewRack(game.alph)
		pstate.SetRackTiles(tiles, game.alph)
		pstate.points = 0

		game.players[i] = pstate

	}

	return game, nil
}

// PlayMove plays a move on the board. This function is meant to be used
// by simulators as it implements a subset of possible moves.
// XXX: It doesn't implement special things like challenge bonuses, etc.
// XXX: Will this still be true, or should this function do it all?
func (g *Game) PlayMove(m *move.Move, backup bool) {
	if backup {
		g.backupState()
	}

	switch m.Action() {
	case move.MoveTypePlay:
		g.board.PlayMove(m, g.gaddag, g.bag.LetterDistribution())
		score := m.Score()
		if score != 0 {
			g.scorelessTurns = 0
		}
		g.players[g.onturn].points += score

		drew := g.bag.DrawAtMost(m.TilesPlayed())
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].SetRackTiles(tiles, g.alph)
		if g.players[g.onturn].rack.NumTiles() == 0 {
			g.playing = false
			unplayedPts := g.calculateRackPts((g.onturn+1)%len(g.players)) * 2
			g.players[g.onturn].points += unplayedPts
		}

	case move.MoveTypePass:
		g.scorelessTurns++

	case move.MoveTypeExchange:
		drew, err := g.bag.Exchange([]alphabet.MachineLetter(m.Tiles()))
		if err != nil {
			panic(err)
		}
		tiles := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].SetRackTiles(tiles, g.alph)
		g.scorelessTurns++
	}

	if g.scorelessTurns == 6 {
		g.playing = false
		// Take away pts on each player's rack.
		for i := 0; i < len(g.players); i++ {
			pts := g.calculateRackPts(i)
			g.players[i].points -= pts
		}
	}
	g.onturn = (g.onturn + 1) % len(g.players)
	g.turnnum++
}

func (g *Game) calculateRackPts(onturn int) int {
	rack := g.players[onturn].rack
	return rack.ScoreOn(g.bag.LetterDistribution())
}

// SetRackFor sets the player's current rack. It throws an error if
// the rack is impossible to set from the current bag. It puts tiles
// back in the bag if needed.
func (g *Game) SetRackFor(playerIdx int, rack *alphabet.Rack) error {
	g.bag.PutBack(g.RackFor(playerIdx).TilesOn())
	err := g.bag.RemoveTiles(rack.TilesOn())
	if err != nil {
		log.Error().Msgf("Unable to set rack: %v", err)
		return err
	}
	g.players[playerIdx].rack = rack
	g.players[playerIdx].rackLetters = rack.String()
	return nil
}

// SetRandomRack sets the player's rack to a random rack drawn from the bag.
// It tosses the current rack back in first. This is used for simulations.
func (g *Game) SetRandomRack(playerIdx int) {
	tiles := g.bag.Redraw(g.RackFor(playerIdx).TilesOn())
	g.players[playerIdx].SetRackTiles(tiles, g.alph)
}

// RackFor returns the rack for the player with the passed-in index
func (g *Game) RackFor(playerIdx int) *alphabet.Rack {
	return g.players[playerIdx].rack
}
