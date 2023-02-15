// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"fmt"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// GameRunner is the master struct here for the automatic game logic.
type GameRunner struct {
	game     *game.Game
	gaddag   gaddag.GenericDawg
	alphabet *alphabet.Alphabet

	lexicon            string
	letterDistribution string
	config             *config.Config
	logchan            chan string
	gamechan           chan string
	aiplayers          [2]aiturnplayer.AITurnPlayer
}

// NewGameRunner just instantiates and initializes a game runner.
func NewGameRunner(logchan chan string, config *config.Config) *GameRunner {
	r := &GameRunner{logchan: logchan, config: config, lexicon: config.DefaultLexicon, letterDistribution: config.DefaultLetterDistribution}
	r.Init([]AutomaticRunnerPlayer{
		{"", "", pb.BotRequest_HASTY_BOT},
		{"", "", pb.BotRequest_HASTY_BOT},
	})

	return r
}

type AutomaticRunnerPlayer struct {
	LeaveFile string
	PEGFile   string
	BotCode   pb.BotRequest_BotCode
}

// Init initializes the runner
func (r *GameRunner) Init(players []AutomaticRunnerPlayer) error {

	rules, err := game.NewBasicGameRules(r.config, r.lexicon, board.CrosswordGameLayout, r.letterDistribution, game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		return err
	}

	pnames := playerNames(players)

	playerInfos := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: pnames[0]},
		{Nickname: "p2", RealName: pnames[1]},
	}

	r.game, err = game.NewGame(rules, playerInfos)
	if err != nil {
		return err
	}

	gd, err := gaddag.Get(r.config, r.lexicon)
	if err != nil {
		return err
	}

	r.gaddag = gd
	r.alphabet = r.gaddag.GetAlphabet()

	for idx := range players {
		leavefile := players[idx].LeaveFile
		pegfile := players[idx].PEGFile
		botcode := players[idx].BotCode
		log.Info().Msgf("botcode %v", botcode)
		var calcs []equity.EquityCalculator
		if botcode == pb.BotRequest_NO_LEAVE_BOT {
			calc := equity.NewNoLeaveCalculator()
			calcs = []equity.EquityCalculator{calc}
		} else {
			calc, err := equity.NewCombinedStaticCalculator(r.gaddag.LexiconName(),
				r.config, leavefile, pegfile)
			if err != nil {
				return err
			}
			calcs = []equity.EquityCalculator{calc}
		}
		tp, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(r.game, r.config, calcs)
		if err != nil {
			return err
		}
		btp := &aiturnplayer.BotTurnPlayer{
			AIStaticTurnPlayer: *tp,
		}
		btp.SetBotType(botcode)
		btp.SetGame(r.game)
		r.aiplayers[idx] = btp
	}
	return nil
}

func (r *GameRunner) StartGame() {
	if frand.Intn(2) == 1 {
		r.game.FlipPlayers()
		r.aiplayers[0], r.aiplayers[1] = r.aiplayers[1], r.aiplayers[0]
	}
	r.game.StartGame()
}

func (r *GameRunner) Game() *game.Game {
	return r.game
}

func (r *GameRunner) genBestStaticTurn(playerIdx int) *move.Move {
	return aiturnplayer.GenBestStaticTurn(r.game, r.aiplayers[playerIdx], playerIdx)
}

func (r *GameRunner) genBestMoveForBot(playerIdx int) *move.Move {
	if r.aiplayers[playerIdx].GetBotType() == pb.BotRequest_HASTY_BOT {
		// For HastyBot we only need to generate one single best static turn.
		return r.genBestStaticTurn(playerIdx)
	}
	// Otherwise use the bot's GenerateMoves function.
	return r.aiplayers[playerIdx].GenerateMoves(1)[0]
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
