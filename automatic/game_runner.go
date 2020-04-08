// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"fmt"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
)

// GameRunner is the master struct here for the automatic game logic.
type GameRunner struct {
	game    *mechanics.XWordGame
	movegen movegen.MoveGenerator

	config        *config.Config
	logchan       chan string
	gamechan      chan string
	endgameSolver *endgame.Solver
	aiplayers     [2]player.AIPlayer
}

// Init initializes the runner
func (r *GameRunner) Init(gd *gaddag.SimpleGaddag) {
	r.game = &mechanics.XWordGame{}
	// XXX: This needs to be configurable.
	ld := alphabet.EnglishLetterDistribution(gd.GetAlphabet())
	r.game.Init(gd, ld)
	strategy := strategy.NewExhaustiveLeaveStrategy(r.game.Bag(), gd.LexiconName(),
		gd.GetAlphabet(), r.config.StrategyParamsPath)
	r.movegen = movegen.NewGordonGenerator(gd, r.game.Board(), ld)
	r.aiplayers[0] = player.NewRawEquityPlayer(strategy)
	r.aiplayers[1] = player.NewRawEquityPlayer(strategy)
}

func (r *GameRunner) StartGame() {
	r.game.Board().Clear()
	r.game.Bag().Refill()
	r.game.StartGame()
}

func (r *GameRunner) genBestStaticTurn(playerID int) *move.Move {
	return player.GenBestStaticTurn(r.game, r.movegen, r.aiplayers[playerID], playerID)
}

// PlayBestStaticTurn generates the best static move for the player and
// plays it on the board.
func (r *GameRunner) PlayBestStaticTurn(playerID int) {
	bestPlay := r.genBestStaticTurn(playerID)
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
