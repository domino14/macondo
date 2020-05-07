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

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
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

	lexicon   string
	config    *config.Config
	logchan   chan string
	gamechan  chan string
	aiplayers [2]player.AIPlayer
}

// NewGameRunner just instantiates and initializes a game runner.
func NewGameRunner(logchan chan string, config *config.Config) *GameRunner {
	r := &GameRunner{logchan: logchan, config: config, lexicon: config.DefaultLexicon}
	r.Init(ExhaustiveLeavePlayer, ExhaustiveLeavePlayer, "", "")
	return r
}

// Init initializes the runner
func (r *GameRunner) Init(player1, player2, leavefile1, leavefile2 string) error {
	// XXX: there should be a data structure for the combination
	// of a lexicon and a letter distribution. For now the following
	// will not work for non-english lexicons, so this needs to be fixed
	// in the future.
	rules, err := game.NewGameRules(r.config, board.CrosswordGameBoard,
		r.lexicon, r.config.DefaultLetterDistribution)
	if err != nil {
		return err
	}

	realName1 := player1 + "-1"
	realName2 := player2 + "-2"

	players := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: realName1},
		{Nickname: "p2", RealName: realName2},
	}

	r.game, err = game.NewGame(rules, players)
	if err != nil {
		return err
	}
	r.gaddag = rules.Gaddag()
	r.alphabet = r.gaddag.GetAlphabet()

	r.movegen = movegen.NewGordonGenerator(r.gaddag, r.game.Board(),
		rules.LetterDistribution())

	var strat strategy.Strategizer
	for idx, pinfo := range players {
		var leavefile string
		if idx == 0 {
			leavefile = leavefile1
		} else if idx == 1 {
			leavefile = leavefile2
		}
		if strings.HasPrefix(pinfo.RealName, ExhaustiveLeavePlayer) {
			strat, err = strategy.NewExhaustiveLeaveStrategy(r.gaddag.LexiconName(),
				r.alphabet, r.config.StrategyParamsPath, leavefile)
			if err != nil {
				return err
			}
		}
		if strings.HasPrefix(pinfo.RealName, NoLeavePlayer) {
			strat = strategy.NewNoLeaveStrategy()
		}
		r.aiplayers[idx] = player.NewRawEquityPlayer(strat)
	}
	return nil
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
	nickOnTurn := r.game.NickOnTurn()
	r.game.PlayMove(bestPlay, false, false)

	if r.logchan != nil {
		r.logchan <- fmt.Sprintf("%v,%v,%v,%v,%v,%v,%v,%v,%v,%.3f,%v\n",
			nickOnTurn,
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
