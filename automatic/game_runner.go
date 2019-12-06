// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/endgame"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
)

const (
	LeaveFile = "leave_values_112719.idx.gz"
)

// GameRunner is the master struct here for the automatic game logic.
type GameRunner struct {
	game          *mechanics.XWordGame
	movegen       *movegen.GordonGenerator
	logchan       chan string
	gamechan      chan string
	endgameSolver *endgame.Solver
}

// Init initializes the runner
func (r *GameRunner) Init(gd *gaddag.SimpleGaddag) {
	r.game = &mechanics.XWordGame{}
	r.game.Init(gd, alphabet.EnglishLetterDistribution())
	strategy := strategy.NewExhaustiveLeaveStrategy(r.game.Bag(), gd.LexiconName(),
		gd.GetAlphabet(), LeaveFile)
	r.movegen = movegen.NewGordonGenerator(r.game, strategy)
}

func (r *GameRunner) StartGame() {
	r.movegen.Reset()
	r.game.StartGame()
}

// PlayBestStaticTurn generates the best static move for the player and
// plays it on the board.
func (r *GameRunner) PlayBestStaticTurn(playerID int) {
	opp := (playerID + 1) % r.game.NumPlayers()
	r.movegen.SetOppRack(r.game.RackFor(opp))
	r.movegen.GenAll(r.game.RackFor(playerID))
	bestPlay := r.movegen.Plays()[0]
	// save rackLetters for logging.
	rackLetters := r.game.RackLettersFor(playerID)
	tilesRemaining := r.game.Bag().TilesRemaining()

	r.game.PlayMove(bestPlay, false)

	if r.logchan != nil {
		r.logchan <- fmt.Sprintf("%v,%v,%v,%v,%v,%v,%v,%v,%v,%.3f,%v\n",
			playerID,
			r.game.Uuid(),
			r.game.Turn(),
			rackLetters,
			bestPlay.ShortDescription(),
			bestPlay.Score(),
			r.game.PointsFor(playerID),
			bestPlay.TilesPlayed(),
			bestPlay.Leave().UserVisible(r.game.Alphabet()),
			bestPlay.Equity(),
			tilesRemaining)
	}
}
