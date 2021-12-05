package game

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

type BackupMode int

const (
	// NoBackup never performs game backups. It can be used for autoplay
	// that has absolutely no input.
	NoBackup BackupMode = iota
	// SimulationMode keeps a stack of game copies, using these for doing
	// endgame and other types of simulations.
	SimulationMode
	// InteractiveGameplayMode keeps just one backup after every turn. This is needed
	// in order to get challenges working, which would roll back the game to
	// an earlier state if a word is challenged off.
	InteractiveGameplayMode
)

// stateBackup is a subset of Game, meant only for backup purposes.
type stateBackup struct {
	board          *board.GameBoard
	bag            *alphabet.Bag
	playing        pb.PlayState
	scorelessTurns int
	onturn         int
	turnnum        int
	players        playerStates
}

func (g *Game) SetBackupMode(m BackupMode) {
	g.backupMode = m
}

func (g *Game) backupState() {
	if g.backupMode == InteractiveGameplayMode {
		g.stackPtr = 0
	}
	st := g.stateStack[g.stackPtr]

	st.board.CopyFrom(g.board)
	st.bag.CopyFrom(g.bag)
	st.playing = g.playing
	st.scorelessTurns = g.scorelessTurns
	st.players.copyFrom(g.players)
	if g.backupMode == SimulationMode {
		st.onturn = g.onturn
		st.turnnum = g.turnnum
		g.stackPtr++
	}
}

func copyPlayers(ps playerStates) playerStates {
	// Make a deep copy of the player slice.
	p := make([]*playerState, len(ps))
	for idx, porig := range ps {
		p[idx] = &playerState{
			PlayerInfo: pb.PlayerInfo{
				Nickname: porig.Nickname,
				RealName: porig.RealName,
			},
			points:      porig.points,
			bingos:      porig.bingos,
			turns:       porig.turns,
			rack:        porig.rack.Copy(),
			rackLetters: porig.rackLetters,
		}
	}
	return p
}

func (ps *playerStates) copyFrom(other playerStates) {
	for idx := range other {
		// Note: this ugly pointer nonsense is purely to make this as fast
		// as possible and avoid allocations.
		(*ps)[idx].rack.CopyFrom(other[idx].rack)
		(*ps)[idx].rackLetters = other[idx].rackLetters
		(*ps)[idx].Nickname = other[idx].Nickname
		(*ps)[idx].RealName = other[idx].RealName
		// XXX: Do I have to copy all the other auto-generated protobuf nonsense fields?
		(*ps)[idx].points = other[idx].points
		(*ps)[idx].bingos = other[idx].bingos
		(*ps)[idx].turns = other[idx].turns
	}
}

func (g *Game) SetStateStackLength(length int) {
	g.stateStack = make([]*stateBackup, length)
	for idx := range g.stateStack {
		// Initialize each element of the stack now to avoid having
		// allocations and GC.
		g.stateStack[idx] = &stateBackup{
			board:          g.board.Copy(),
			bag:            g.bag.Copy(),
			playing:        g.playing,
			scorelessTurns: g.scorelessTurns,
			players:        copyPlayers(g.players),
		}
	}
}

// UnplayLastMove is a tricky but crucial function for any sort of simming /
// minimax search / etc. It restores the state after playing a move, without
// having to store a giant amount of data. The alternative is to store the entire
// game state with every node which quickly becomes unfeasible.
func (g *Game) UnplayLastMove() {
	// Pop the last element, essentially.
	var b *stateBackup
	if g.backupMode == SimulationMode {
		b = g.stateStack[g.stackPtr-1]
		g.stackPtr--
		// Turn number and on turn do not need to be restored from backup
		// as they're assumed to increase logically after every turn. Just
		// decrease them.
		g.turnnum--
		g.onturn = (g.onturn + (len(g.players) - 1)) % len(g.players)
	} else {
		b = g.stateStack[0]
	}

	g.board.CopyFrom(b.board)
	g.bag.CopyFrom(b.bag)
	g.playing = b.playing
	g.players.copyFrom(b.players)
	g.scorelessTurns = b.scorelessTurns
}

// ResetToFirstState unplays all moves on the stack.
func (g *Game) ResetToFirstState() {
	// This is like a fast-backward -- it unplays all of the moves on the
	// stack.

	b := g.stateStack[0]
	g.onturn = b.onturn
	g.turnnum = b.turnnum
	g.stackPtr = 0

	g.board.CopyFrom(b.board)
	g.bag.CopyFrom(b.bag)
	g.playing = b.playing
	g.players.copyFrom(b.players)
	g.scorelessTurns = b.scorelessTurns
}

// Copy creates a deep copy of Game for the most part. The lexicon and
// alphabet are not deep-copied because these are not expected to change.
// The history is not copied because this only changes with the main Game,
// and not these copies.
func (g *Game) Copy() *Game {

	copy := &Game{
		config:            g.config,
		onturn:            g.onturn,
		turnnum:           g.turnnum,
		board:             g.board.Copy(),
		bag:               g.bag.Copy(),
		lexicon:           g.lexicon,
		crossSetGen:       g.crossSetGen,
		alph:              g.alph,
		playing:           g.playing,
		scorelessTurns:    g.scorelessTurns,
		maxScorelessTurns: g.maxScorelessTurns,
		players:           copyPlayers(g.players),
		// stackPtr only changes during a sim, etc. This Copy should
		// only be called at the beginning of everything.
		stackPtr: 0,
	}
	// Also set the copy's stack.
	copy.SetStateStackLength(len(g.stateStack))
	return copy
}
