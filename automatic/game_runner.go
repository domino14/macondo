// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"fmt"
	"strings"

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

const (
	ExhaustiveLeavePlayer = "exhaustiveleave"
	NoLeavePlayer         = "noleave"
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
	r.Init(ExhaustiveLeavePlayer, ExhaustiveLeavePlayer)
	return r
}

// Init initializes the runner
func (r *GameRunner) Init(player1 string, player2 string) {
	rules, err := game.NewGameRules(r.config, board.CrosswordGameBoard,
		r.config.DefaultLexicon, r.config.DefaultLetterDistribution)
	if err != nil {
		panic(err)
	}

	realName1 := player1
	realName2 := player2
	if player1 == player2 {
		realName2 = realName1 + "2"
	}

	players := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: realName1, Number: 1},
		{Nickname: "p2", RealName: realName2, Number: 2},
	}

	r.game, err = game.NewGame(rules, players)
	if err != nil {
		panic(err)
	}
	r.gaddag = rules.Gaddag()
	r.alphabet = r.gaddag.GetAlphabet()

	r.movegen = movegen.NewGordonGenerator(r.gaddag, r.game.Board(),
		rules.LetterDistribution())

	var strat strategy.Strategizer
	for idx, pinfo := range players {
		if strings.HasPrefix(pinfo.RealName, ExhaustiveLeavePlayer) {
			strat = strategy.NewExhaustiveLeaveStrategy(r.gaddag.LexiconName(),
				r.alphabet, r.config.StrategyParamsPath)
		}
		if strings.HasPrefix(pinfo.RealName, NoLeavePlayer) {
			strat = strategy.NewNoLeaveStrategy()
		}
		r.aiplayers[idx] = player.NewRawEquityPlayer(strat)
	}
}

func (r *GameRunner) StartGame() {
	r.game.StartGame()
}

func (r *GameRunner) genBestStaticTurn(playerIdx int) *move.Move {
	return player.GenBestStaticTurn(r.game, r.movegen, r.aiplayers[playerIdx], playerIdx)
}

// PlayBestStaticTurn generates the best static move for the player and
// plays it on the board.
func (r *GameRunner) PlayBestStaticTurn(playerIdx int) {
	bestPlay := r.genBestStaticTurn(playerIdx)
	// save rackLetters for logging.
	rackLetters := r.game.RackLettersFor(playerIdx)
	tilesRemaining := r.game.Bag().TilesRemaining()

	r.game.PlayMove(bestPlay, false, false)

	if r.logchan != nil {
		r.logchan <- fmt.Sprintf("%v,%v,%v,%v,%v,%v,%v,%v,%v,%.3f,%v\n",
			playerIdx,
			r.game.Uid(),
			r.game.Turn(),
			rackLetters,
			bestPlay.ShortDescription(),
			bestPlay.Score(),
			r.game.PointsFor(playerIdx),
			bestPlay.TilesPlayed(),
			bestPlay.Leave().UserVisible(r.alphabet),
			bestPlay.Equity(),
			tilesRemaining)
	}
}
