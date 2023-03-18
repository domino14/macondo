// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"context"
	"fmt"
	"time"

	"github.com/domino14/macondo/ai/bot"
	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
	"github.com/rs/zerolog/log"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

var MaxTimePerTurn = 15 * time.Second
var MaxTimePerEndgame = 10 * time.Second

// GameRunner is the master struct here for the automatic game logic.
type GameRunner struct {
	game     *game.Game
	gaddag   gaddag.WordGraph
	alphabet *tilemapping.TileMapping

	lexicon            string
	letterDistribution string
	config             *config.Config
	logchan            chan string
	gamechan           chan string
	aiplayers          [2]aiturnplayer.AITurnPlayer
	order              [2]int
}

// NewGameRunner just instantiates and initializes a game runner.
func NewGameRunner(logchan chan string, config *config.Config) *GameRunner {
	r := &GameRunner{logchan: logchan, config: config, lexicon: config.DefaultLexicon, letterDistribution: config.DefaultLetterDistribution}
	r.Init([]AutomaticRunnerPlayer{
		{"", "", pb.BotRequest_HASTY_BOT, 0},
		{"", "", pb.BotRequest_HASTY_BOT, 0},
	})

	return r
}

type AutomaticRunnerPlayer struct {
	LeaveFile   string
	PEGFile     string
	BotCode     pb.BotRequest_BotCode
	MinSimPlies int
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

	gd, err := kwg.Get(r.config, r.lexicon)
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

		conf := &bot.BotConfig{
			Config:            *r.config,
			PEGAdjustmentFile: pegfile,
			LeavesFile:        leavefile,
			MinSimPlies:       players[idx].MinSimPlies,
		}

		btp, err := bot.NewBotTurnPlayerFromGame(r.game, conf, botcode)
		if err != nil {
			return err
		}

		r.aiplayers[idx] = btp
	}
	r.order = [2]int{0, 1}
	return nil
}

func (r *GameRunner) StartGame(gidx int) {
	// r.order must be {0, 1} if gidx is even, and {1, 0} if odd
	flip := false
	if gidx%2 == 1 {
		if r.order[0] == 0 {
			flip = true
		}
	} else {
		if r.order[1] == 0 {
			flip = true
		}
	}

	if flip {
		r.game.FlipPlayers()
		r.aiplayers[0], r.aiplayers[1] = r.aiplayers[1], r.aiplayers[0]
		r.order[0], r.order[1] = r.order[1], r.order[0]
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
	maxTime := MaxTimePerTurn
	if r.game.Bag().TilesRemaining() == 0 {
		log.Debug().Msg("runner-bag-is-empty")
		maxTime = MaxTimePerEndgame
	}
	// Otherwise use the bot's GenerateMoves function.
	ctx, cancel := context.WithTimeout(context.Background(), maxTime)
	defer cancel()
	m, err := r.aiplayers[playerIdx].BestPlay(ctx)
	if err != nil {
		log.Err(err).Msg("generating best move for bot")
	}
	return m
}

// PlayBestTurn generates the best move for the player and plays it on the board.
func (r *GameRunner) PlayBestTurn(playerIdx int, addToHistory bool) error {
	bestPlay := r.genBestMoveForBot(playerIdx)
	// save rackLetters for logging.
	rackLetters := r.game.RackLettersFor(playerIdx)
	tilesRemaining := r.game.Bag().TilesRemaining()
	nickOnTurn := r.game.NickOnTurn()
	err := r.game.PlayMove(bestPlay, addToHistory, 0)
	if err != nil {
		return err
	}
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
	return nil
}
