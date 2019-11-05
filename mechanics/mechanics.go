// Package mechanics implements the rules / mechanics of Crossword game.
// It contains all the logic for the actual gameplay of Crossword Game,
// which, as we said before, features all sorts of things like wingos
// and blonks.
package mechanics

import (
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
	players            []player
	uuid               uuid.UUID
}

// Init initializes the crossword game and seeds the random number generator.
func (g *XWordGame) Init(gd *gaddag.SimpleGaddag, dist *alphabet.LetterDistribution) {
	g.numPossibleLetters = int(gd.GetAlphabet().NumLetters())
	g.board = board.MakeBoard(board.CrosswordGameBoard)
	g.alph = gd.GetAlphabet()
	g.bag = dist.MakeBag(g.alph)
	g.gaddag = gd
	g.players = []player{
		{alphabet.NewRack(g.alph), "", "player1", 0},
		{alphabet.NewRack(g.alph), "", "player2", 0},
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

// PlayMove plays a move on the board.
func (g *XWordGame) PlayMove(m *move.Move) {
	switch m.Action() {
	case move.MoveTypePlay:
		g.board.PlayMove(m, g.gaddag, g.bag, false)
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

func (g *XWordGame) SetRackFor(playerID int, rack *alphabet.Rack) {
	g.players[playerID].rack = rack
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

func (g *XWordGame) Playing() bool {
	return g.playing
}

func (g *XWordGame) Alphabet() *alphabet.Alphabet {
	return g.alph
}
