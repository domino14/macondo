// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"fmt"
	"strings"

	"github.com/domino14/macondo/ai/player"
	airunner "github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

const (
	ExhaustiveLeavePlayer = "exhaustiveleave"
	NoLeavePlayer         = "noleave"
)

// GameRunner is the master struct here for the automatic game logic.
type GameRunner struct {
	game     *game.Game
	gaddag   gaddag.GenericDawg
	movegens [2]movegen.MoveGenerator
	alphabet *alphabet.Alphabet

	lexicon            string
	letterDistribution string
	config             *config.Config
	logchan            chan string
	gamechan           chan string
	aiplayers          [2]player.AIPlayer
}

// NewGameRunner just instantiates and initializes a game runner.
func NewGameRunner(logchan chan string, config *config.Config) *GameRunner {
	r := &GameRunner{logchan: logchan, config: config, lexicon: config.DefaultLexicon, letterDistribution: config.DefaultLetterDistribution}
	r.Init(ExhaustiveLeavePlayer, ExhaustiveLeavePlayer, "", "", "", "", pb.BotRequest_HASTY_BOT, pb.BotRequest_HASTY_BOT)
	return r
}

// Init initializes the runner
func (r *GameRunner) Init(player1, player2, leavefile1, leavefile2, pegfile1, pegfile2 string,
	botcode1, botcode2 pb.BotRequest_BotCode) error {

	rules, err := airunner.NewAIGameRules(r.config, board.CrosswordGameLayout, game.VarClassic,
		r.lexicon, r.letterDistribution)
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

	gd, err := gaddag.Get(r.config, r.lexicon)
	if err != nil {
		return err
	}

	r.gaddag = gd

	r.alphabet = r.gaddag.GetAlphabet()

	// We use two movegens so that each movegen can have independent
	// play recorders.
	r.movegens = [2]movegen.MoveGenerator{
		movegen.NewGordonGenerator(r.gaddag.(*gaddag.SimpleGaddag), r.game.Board(),
			rules.LetterDistribution()),
		movegen.NewGordonGenerator(r.gaddag.(*gaddag.SimpleGaddag), r.game.Board(),
			rules.LetterDistribution()),
	}
	r.game.SetAddlState(r.movegen.(*movegen.GordonGenerator).State())

	var strat strategy.Strategizer
	for idx, pinfo := range players {
		var leavefile, pegfile string
		var botcode pb.BotRequest_BotCode
		if idx == 0 {
			leavefile = leavefile1
			pegfile = pegfile1
			botcode = botcode1
		} else if idx == 1 {
			leavefile = leavefile2
			pegfile = pegfile2
			botcode = botcode2
		}
		log.Info().Msgf("botcode %v", botcode)
		if strings.HasPrefix(pinfo.RealName, ExhaustiveLeavePlayer) {
			strat, err = strategy.NewExhaustiveLeaveStrategy(r.gaddag.LexiconName(),
				r.alphabet, r.config, leavefile, pegfile)
			if err != nil {
				return err
			}
		}
		if strings.HasPrefix(pinfo.RealName, NoLeavePlayer) {
			strat = strategy.NewNoLeaveStrategy()
		}
		r.aiplayers[idx] = player.NewRawEquityPlayer(strat, botcode)
	}
	return nil
}

func (r *GameRunner) StartGame() {
	if frand.Intn(2) == 1 {
		r.game.FlipPlayers()
		// XXX: probably associate the movegen with the aiplayer in the future.
		r.aiplayers[0], r.aiplayers[1] = r.aiplayers[1], r.aiplayers[0]
		r.movegens[0], r.movegens[1] = r.movegens[1], r.movegens[0]
	}
	r.game.StartGame()
}

func (r *GameRunner) Game() *game.Game {
	return r.game
}

func (r *GameRunner) genBestStaticTurn(playerIdx int) *move.Move {
	return player.GenBestStaticTurn(r.game, r.movegens[playerIdx], r.aiplayers[playerIdx], playerIdx)
}

func (r *GameRunner) genBestMoveForBot(playerIdx int) *move.Move {
	if r.aiplayers[playerIdx].GetBotType() == pb.BotRequest_HASTY_BOT {
		// For HastyBot we only need to generate one single best static turn.
		return r.genBestStaticTurn(playerIdx)
	}
	return airunner.GenerateMoves(
		r.game, r.aiplayers[playerIdx], r.movegens[playerIdx], r.config, 1)[0]
}

// PlayBestStaticTurn generates the best static move for the player and
// plays it on the board.
func (r *GameRunner) PlayBestStaticTurn(playerIdx int, addToHistory bool) {
	bestPlay := r.genBestMoveForBot(playerIdx)
	// save rackLetters for logging.
	rackLetters := r.game.RackLettersFor(playerIdx)
	tilesRemaining := r.game.Bag().TilesRemaining()
	nickOnTurn := r.game.NickOnTurn()
	r.game.PlayMove(bestPlay, addToHistory, 0)

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
