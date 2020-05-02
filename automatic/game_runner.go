// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"fmt"

	"github.com/domino14/macondo/ai/player"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"

	pb "github.com/domino14/macondo/rpc/api/proto"
)

// GameRunner is the master struct here for the automatic game logic.
type GameRunner struct {
	game     *game.Game
	gaddag   *gaddag.SimpleGaddag
	movegen  movegen.MoveGenerator
	alphabet *alphabet.Alphabet

	config    *config.Config
	logchan   chan string
	gamechan  chan string
	aiplayers [2]player.AIPlayer
}

// NewGameRunner just instantiates and initializes a game runner.
func NewGameRunner(logchan chan string, config *config.Config) *GameRunner {
	r := &GameRunner{logchan: logchan, config: config}
	r.Init()
	return r
}

// Init initializes the runner
func (r *GameRunner) Init() {
	rules, err := game.NewGameRules(r.config, board.CrosswordGameBoard,
		r.config.DefaultLexicon, r.config.DefaultLetterDistribution)
	if err != nil {
		panic(err)
	}

	players := []*pb.PlayerInfo{
		&pb.PlayerInfo{Nickname: "p1", RealName: "Player 1", Number: 1},
		&pb.PlayerInfo{Nickname: "p2", RealName: "Player 2", Number: 2},
	}

	r.game, err = game.NewGame(rules, players)
	if err != nil {
		panic(err)
	}
	r.gaddag = rules.Gaddag()
	r.alphabet = r.gaddag.GetAlphabet()
	strategy := strategy.NewExhaustiveLeaveStrategy(r.gaddag.LexiconName(),
		r.alphabet, r.config.StrategyParamsPath)
	r.movegen = movegen.NewGordonGenerator(r.gaddag, r.game.Board(),
		rules.LetterDistribution())
	r.aiplayers[0] = player.NewRawEquityPlayer(strategy)
	r.aiplayers[1] = player.NewRawEquityPlayer(strategy)
}

func (r *GameRunner) StartGame() {
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

	r.game.PlayMove(bestPlay, false, false)

	if r.logchan != nil {
		r.logchan <- fmt.Sprintf("%v,%v,%v,%v,%v,%v,%v,%v,%v,%.3f,%v\n",
			playerID,
			r.game.Uid(),
			r.game.Turn(),
			rackLetters,
			bestPlay.ShortDescription(),
			bestPlay.Score(),
			r.game.PointsFor(playerID),
			bestPlay.TilesPlayed(),
			bestPlay.Leave().UserVisible(r.alphabet),
			bestPlay.Equity(),
			tilesRemaining)
	}
}
