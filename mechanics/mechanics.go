// Package mechanics implements the rules / mechanics of Crossword game.
// It contains all the logic for the actual gameplay of Crossword Game,
// which, as we said before, features all sorts of things like wingos
// and blonks.
package mechanics

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/google/uuid"
)

// A player plays crossword game. This is a very minimal structure that only
// keeps track of things such as rack and points. We will use a more overarching
// Player structure elsewhere with strategy, endgame solver, etc.
type player struct {
	rack        *alphabet.Rack
	rackLetters string // user-visible for ease in logging
	name        string
	points      int
}

type players []player

// XWordGame encapsulates the various components of a crossword game. At
// any given time it can be thought of as the current state of a game.
type XWordGame struct {
	onturn             int
	turnnum            int
	board              *board.GameBoard
	bag                *alphabet.Bag
	gaddag             *gaddag.SimpleGaddag
	alph               *alphabet.Alphabet
	playing            bool
	scorelessTurns     int
	numPossibleLetters int
	players            []*player
	uuid               uuid.UUID

	stateStack []*backedupState
	stackPtr   int
}

// String returns a helpful string representation of this state.
func (s *XWordGame) String() string {
	ret := ""
	for idx, p := range s.players {
		if idx == s.onturn {
			ret += "*"
		}
		ret += fmt.Sprintf("%v holding %v (%v)", p.name, p.rackLetters,
			p.points)
		ret += " - "
	}
	ret += fmt.Sprintf(" | pl=%v slt=%v", s.playing, s.scorelessTurns)
	return ret
}

type backedupState struct {
	board          *board.GameBoard
	bag            *alphabet.Bag
	playing        bool
	scorelessTurns int
	players        []*player
	lastWasPass    bool
}

// Init initializes the crossword game and seeds the random number generator.
func (g *XWordGame) Init(gd *gaddag.SimpleGaddag, dist *alphabet.LetterDistribution) {
	g.numPossibleLetters = int(gd.GetAlphabet().NumLetters())
	g.board = board.MakeBoard(board.CrosswordGameBoard)
	g.alph = gd.GetAlphabet()
	g.bag = dist.MakeBag(g.alph)
	g.gaddag = gd
	g.players = []*player{
		&player{alphabet.NewRack(g.alph), "", "player1", 0},
		&player{alphabet.NewRack(g.alph), "", "player2", 0},
	}
	// The strategy and move generator are not part of the "game mechanics".
	// These should be a level up. This module is just for the gameplay side
	// of things, taking turns, logic, etc.
}

// StartGame resets everything and deals out the first set of tiles.
func (g *XWordGame) StartGame() {
	g.uuid = uuid.New()
	// reset movegen outside of this function.

	for i := 0; i < len(g.players); i++ {
		rack, _ := g.bag.Draw(7)
		g.players[i].rackLetters = alphabet.MachineWord(rack).UserVisible(g.alph)
		g.players[i].points = 0
		g.players[i].rack.Set(rack)
	}
	g.onturn = 0
	g.turnnum = 0
	g.playing = true
}

func (ps *players) copyFrom(other players) {
	for idx := range other {
		// Note: this ugly pointer nonsense is purely to make this as fast
		// as possible and avoid allocations.
		(*ps)[idx].rack.CopyFrom(other[idx].rack)
		(*ps)[idx].rackLetters = other[idx].rackLetters
		(*ps)[idx].name = other[idx].name
		(*ps)[idx].points = other[idx].points
	}
}

func copyPlayers(ps []*player) []*player {
	// Make a deep copy of the player slice.
	p := make([]*player, len(ps))
	for idx, porig := range ps {
		p[idx] = &player{
			name:        porig.name,
			points:      porig.points,
			rack:        porig.rack.Copy(),
			rackLetters: porig.rackLetters,
		}
	}
	return p
}

// PlayMove plays a move on the board.
func (g *XWordGame) PlayMove(m *move.Move, backup bool) {
	// If backup is on, we should back up a lot of the relevant state.
	// This allows us to backtrack / undo moves for simulations/etc.

	if backup {
		g.backupState()
	}

	switch m.Action() {
	case move.MoveTypePlay:
		g.board.PlayMove(m, g.gaddag, g.bag)

		score := m.Score()
		if score != 0 {
			g.scorelessTurns = 0
		}
		g.players[g.onturn].points += score
		// log.Printf("[DEBUG] Player %v played %v for %v points (equity %v, total score %v)", game.onturn, m,
		// 	score, m.Equity(), game.players[game.onturn].points)
		// Draw new tiles.
		drew := g.bag.DrawAtMost(m.TilesPlayed())
		rack := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].rack.Set(rack)
		g.players[g.onturn].rackLetters = alphabet.MachineWord(rack).UserVisible(g.alph)

		if g.players[g.onturn].rack.NumTiles() == 0 {
			// log.Printf("[DEBUG] Player %v played off all their tiles. Game over!",
			// 	game.onturn)
			g.playing = false
			unplayedPts := g.calculateRackPts((g.onturn+1)%len(g.players)) * 2
			// log.Printf("[DEBUG] Player %v gets %v points from unplayed tiles",
			// 	game.onturn, unplayedPts)
			g.players[g.onturn].points += unplayedPts
		} else {
			// log.Printf("[DEBUG] Player %v drew new tiles: %v, rack is now %v",
			// 	game.onturn, string(drew), rack)
		}
	case move.MoveTypePass:
		// log.Printf("[DEBUG] Player %v passed", game.onturn)
		g.scorelessTurns++

	case move.MoveTypeExchange:
		// XXX: Gross; the bag should be full of MachineLetter.
		drew, err := g.bag.Exchange([]alphabet.MachineLetter(m.Tiles()))
		if err != nil {
			panic(err)
		}
		rack := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].rack.Set(rack)
		g.players[g.onturn].rackLetters = alphabet.MachineWord(rack).UserVisible(g.alph)
		g.scorelessTurns++
	}
	if g.scorelessTurns == 6 {
		// log.Printf("[DEBUG] Game ended after 6 scoreless turns")
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

// UnplayLastMove is a tricky but crucial function for any sort of simming /
// minimax search / etc. It restores the state after playing a move, without
// having to store a giant amount of data. The alternative is to store the entire
// game state with every node which quickly becomes unfeasible.
func (g *XWordGame) UnplayLastMove() {
	// Things that need to be restored after a move:
	// [x] The current user turn
	// [x] The bag tiles
	// [x] The board state (including cross-checks / anchors)
	// [x] The scores
	// [x] The player racks
	// The clock (in the future? May never be needed)
	// [x] The scoreless turns
	// [x] Turn number

	// Pop the last element, essentially.
	b := g.stateStack[g.stackPtr-1]
	g.stackPtr--

	// Turn number and on turn do not need to be restored from backup
	// as they're assumed to increase logically after every turn. Just
	// decrease them.
	g.turnnum--
	g.onturn = (g.onturn + (len(g.players) - 1)) % len(g.players)

	g.board.CopyFrom(b.board)
	g.bag.CopyFrom(b.bag)
	g.playing = b.playing
	g.players = copyPlayers(b.players)
	g.scorelessTurns = b.scorelessTurns
}

func (g *XWordGame) backupState() {
	// st := &backedupState{
	// 	board:          g.board.Copy(),
	// 	bag:            g.bag.Copy(),
	// 	playing:        g.playing,
	// 	scorelessTurns: g.scorelessTurns,
	// 	players:        copyPlayers(g.players),
	// }
	st := g.stateStack[g.stackPtr]
	// Copy directly.
	st.board.CopyFrom(g.board)
	st.bag.CopyFrom(g.bag)
	st.playing = g.playing
	st.scorelessTurns = g.scorelessTurns
	st.players = copyPlayers(g.players)
	g.stackPtr++
}

func (g *XWordGame) calculateRackPts(onturn int) int {
	// Calculate the number of pts on the player with the `onturn` rack.
	rack := g.players[onturn].rack
	return rack.ScoreOn(g.bag)
}

func (g *XWordGame) NumPlayers() int {
	return len(g.players)
}

func (g *XWordGame) Bag() *alphabet.Bag {
	return g.bag
}

func (g *XWordGame) Board() *board.GameBoard {
	return g.board
}

func (g *XWordGame) Gaddag() *gaddag.SimpleGaddag {
	return g.gaddag
}

func (g *XWordGame) SetRackFor(playerID int, rack *alphabet.Rack) {
	g.players[playerID].rack = rack
	g.players[playerID].rackLetters = rack.String()
}

func (g *XWordGame) SetPointsFor(playerID int, pts int) {
	g.players[playerID].points = pts
}

func (g *XWordGame) SetNameFor(playerID int, name string) {
	g.players[playerID].name = name
}

func (g *XWordGame) RackFor(playerID int) *alphabet.Rack {
	return g.players[playerID].rack
}

func (g *XWordGame) RackLettersFor(playerID int) string {
	return g.players[playerID].rackLetters
}

func (g *XWordGame) PointsFor(playerID int) int {
	return g.players[playerID].points
}

func (g *XWordGame) Uuid() uuid.UUID {
	return g.uuid
}

// Turn returns the current turn number.
func (g *XWordGame) Turn() int {
	return g.turnnum
}

// PlayerOnTurn returns the current player index whose turn it is.
func (g *XWordGame) PlayerOnTurn() int {
	return g.onturn
}

func (g *XWordGame) SetPlayerOnTurn(playerID int) {
	g.onturn = playerID
}

func (g *XWordGame) Playing() bool {
	return g.playing
}

func (g *XWordGame) SetPlaying(p bool) {
	g.playing = p
}

func (g *XWordGame) Alphabet() *alphabet.Alphabet {
	return g.alph
}

func (g *XWordGame) SetStateStackLength(l int) {
	g.stateStack = make([]*backedupState, l)
	for idx := range g.stateStack {
		// Initialize each element of the stack now to avoid having
		// allocations and GC.
		g.stateStack[idx] = &backedupState{
			board:          g.board.Copy(),
			bag:            g.bag.Copy(),
			playing:        g.playing,
			scorelessTurns: g.scorelessTurns,
			players:        copyPlayers(g.players),
		}
	}
}
