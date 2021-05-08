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
	"github.com/domino14/macondo/runner"
	"github.com/domino14/macondo/strategy"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

const (
	ExhaustiveLeavePlayer = "exhaustiveleave"
	NoLeavePlayer         = "noleave"
)

// GameRunner is the master struct here for the automatic game logic.
type GameRunner struct {
	game *game.Game
	// gaddag    gaddag.GenericDawg
	movegenp1 movegen.MoveGenerator
	movegenp2 movegen.MoveGenerator
	alphabet  *alphabet.Alphabet

	lexicon   string
	config    *config.Config
	logchan   chan string
	gamechan  chan string
	aiplayers [2]player.AIPlayer
}

// NewGameRunner just instantiates and initializes a game runner.
func NewGameRunner(logchan chan string, config *config.Config) *GameRunner {
	r := &GameRunner{logchan: logchan, config: config, lexicon: config.DefaultLexicon}
	r.Init(ExhaustiveLeavePlayer, ExhaustiveLeavePlayer, "", "", "", "", "", "")
	return r
}

// Init initializes the runner
func (r *GameRunner) Init(player1, player2, leavefile1, leavefile2, pegfile1, pegfile2,
	dictfile1, dictfile2 string) error {
	// XXX: there should be a data structure for the combination
	// of a lexicon and a letter distribution. For now the following
	// will not work for non-english lexicons, so this needs to be fixed
	// in the future.
	rules, err := runner.NewAIGameRules(r.config, board.CrosswordGameBoard,
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

	var lexiconName string
	for idx, v := range []string{dictfile1, dictfile2} {
		var gd *gaddag.SimpleGaddag
		var err error
		if v == "" {
			gd, err = gaddag.Get(r.config, r.lexicon)
			if err != nil {
				return err
			}
		} else {
			gd, err = gaddag.LoadGaddag(v)
			if err != nil {
				return err
			}
		}
		lexiconName = gd.LexiconName()
		// Assume they would have the same alphabet.
		r.alphabet = gd.GetAlphabet()
		mg := movegen.NewGordonGenerator(gd, r.game.Board(), rules.LetterDistribution())
		if idx == 0 {
			r.movegenp1 = mg
		} else {
			r.movegenp2 = mg
		}
	}

	var strat strategy.Strategizer
	for idx, pinfo := range players {
		var leavefile, pegfile string
		if idx == 0 {
			leavefile = leavefile1
			pegfile = pegfile1
		} else if idx == 1 {
			leavefile = leavefile2
			pegfile = pegfile2
		}
		if strings.HasPrefix(pinfo.RealName, ExhaustiveLeavePlayer) {
			strat, err = strategy.NewExhaustiveLeaveStrategy(lexiconName,
				r.alphabet, r.config, leavefile, pegfile)
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
	mg := r.movegenp1
	if playerIdx == 1 {
		mg = r.movegenp2
	}
	return player.GenBestStaticTurn(r.game, mg, r.aiplayers[playerIdx], playerIdx)
}

// PlayBestStaticTurn generates the best static move for the player and
// plays it on the board.
func (r *GameRunner) PlayBestStaticTurn(playerIdx int) {
	bestPlay := r.genBestStaticTurn(playerIdx)
	// save rackLetters for logging.
	rackLetters := r.game.RackLettersFor(playerIdx)
	tilesRemaining := r.game.Bag().TilesRemaining()
	nickOnTurn := r.game.NickOnTurn()
	r.game.PlayMove(bestPlay, false, 0)

	if r.logchan != nil {
		r.logchan <- fmt.Sprintf("%v,%v,%v,%v,%v,%v,%v,%v,%v,%.3f,%v,%v\n",
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
			tilesRemaining,
			r.game.PointsFor((playerIdx+1)%2))
	}
}
