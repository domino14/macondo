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

// A Player plays crossword game. This is a very minimal structure that only
// keeps track of things such as rack and points. We will use a more overarching
// Player structure elsewhere with strategy, endgame solver, etc.
type Player struct {
	Nickname     string `json:"nick"`
	RealName     string `json:"real_name"`
	PlayerNumber uint8  `json:"p_number"`

	rack        *alphabet.Rack
	rackLetters string
	points      int
}

type players []*Player

func (p *Player) resetScore() {
	p.points = 0
}

func (p players) resetScore() {
	for idx := range p {
		p[idx].resetScore()
	}
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
	players            players
	uuid               uuid.UUID
	turnHistory        []Turn

	stateStack []*backedupState
	stackPtr   int
}

// String returns a helpful string representation of this state.
func (g *XWordGame) String() string {
	ret := ""
	for idx, p := range g.players {
		if idx == g.onturn {
			ret += "*"
		}
		ret += fmt.Sprintf("%v holding %v (%v)", p.Nickname, p.rackLetters,
			p.points)
		ret += " - "
	}
	ret += fmt.Sprintf(" | pl=%v slt=%v", g.playing, g.scorelessTurns)
	return ret
}

type backedupState struct {
	board          *board.GameBoard
	bag            *alphabet.Bag
	playing        bool
	scorelessTurns int
	players        players
	lastWasPass    bool
}

// Init initializes the crossword game and seeds the random number generator.
func (g *XWordGame) Init(gd *gaddag.SimpleGaddag, dist *alphabet.LetterDistribution) {
	g.numPossibleLetters = int(gd.GetAlphabet().NumLetters())
	g.board = board.MakeBoard(board.CrosswordGameBoard)
	g.alph = gd.GetAlphabet()
	g.bag = dist.MakeBag(g.alph)
	g.gaddag = gd
	g.players = []*Player{
		&Player{"player1", "player1", 0, alphabet.NewRack(g.alph), "", 0},
		&Player{"player2", "player2", 1, alphabet.NewRack(g.alph), "", 0},
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
		(*ps)[idx].Nickname = other[idx].Nickname
		(*ps)[idx].RealName = other[idx].RealName
		(*ps)[idx].PlayerNumber = other[idx].PlayerNumber
		(*ps)[idx].points = other[idx].points
	}
}

func copyPlayers(ps players) players {
	// Make a deep copy of the player slice.
	p := make([]*Player, len(ps))
	for idx, porig := range ps {
		p[idx] = &Player{
			Nickname:     porig.Nickname,
			RealName:     porig.RealName,
			PlayerNumber: porig.PlayerNumber,
			points:       porig.points,
			rack:         porig.rack.Copy(),
			rackLetters:  porig.rackLetters,
		}
	}
	return p
}

// UpdateTurnHistory should be called after PlayMove, but only for places
// where we are interacting with the game. Note that PlayMove also gets
// called when doing sims / endgame lookups, so we don't want to be doing
// expensive updates and backups on turn history during these moments.
func (g *XWordGame) UpdateTurnHistory(m *move.Move) {
	// switch m.Action() {
	// case move.MoveTypePlay:
	// 	g.turnHistory = append(g.turnHistory, newPlacementTurn(m, g.players[pnum]))
	// case move.MoveTypePass:
	// 	g.turnHistory = append(g.turnHistory, newPassTurn(m))
	// case move.MoveTypeExchange:
	// 	g.turnHistory = append(g.turnHistory, newExchangeTurn(m))
	// }
}

// PlayMove plays a move on the board. This function is meant to be used
// by simulators as it implements a subset of possible moves. It doesnt
// implement special things like challenge bonuses, etc.
func (g *XWordGame) PlayMove(m *move.Move, backup bool) {
	// If backup is on, we should back up a lot of the relevant state.
	// This allows us to backtrack / undo moves for simulations/etc.

	if backup {
		g.backupState()
	}

	// Note that we are not backing up the turn history. This would be kind
	// of expensive and unneeded; we only use backup with sims and the like.
	switch m.Action() {
	case move.MoveTypePlay:
		g.board.PlayMove(m, g.gaddag, g.bag)
		score := m.Score()
		if score != 0 {
			g.scorelessTurns = 0
		}
		g.players[g.onturn].points += score
		// log.Debug().Msgf("Player %v played %v for %v points (total score %v)",
		// 	g.onturn, m, score, g.players[g.onturn].points)
		// Draw new tiles.
		drew := g.bag.DrawAtMost(m.TilesPlayed())
		rack := append(drew, []alphabet.MachineLetter(m.Leave())...)
		g.players[g.onturn].rack.Set(rack)
		g.players[g.onturn].rackLetters = alphabet.MachineWord(rack).UserVisible(g.alph)

		if g.players[g.onturn].rack.NumTiles() == 0 {
			g.playing = false
			unplayedPts := g.calculateRackPts((g.onturn+1)%len(g.players)) * 2
			g.players[g.onturn].points += unplayedPts
		}

	case move.MoveTypePass:
		// log.Printf("[DEBUG] Player %v passed", game.onturn)
		g.scorelessTurns++

	case move.MoveTypeExchange:
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
	// [x] Turn history

	// Pop the last element, essentially.
	b := g.stateStack[g.stackPtr-1]
	g.stackPtr--

	// Turn number and on turn do not need to be restored from backup
	// as they're assumed to increase logically after every turn. Just
	// decrease them. Similarly, pop the turn history.
	g.turnnum--
	g.onturn = (g.onturn + (len(g.players) - 1)) % len(g.players)

	g.board.CopyFrom(b.board)
	g.bag.CopyFrom(b.bag)
	g.playing = b.playing
	g.players.copyFrom(b.players)
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
	st.players.copyFrom(g.players)
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

// CurrentSpread returns the spread of the current game state.
func (g *XWordGame) CurrentSpread() int {
	other := (g.onturn + 1) % len(g.players)
	return g.players[g.onturn].points - g.players[other].points
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
