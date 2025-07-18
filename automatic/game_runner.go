// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"context"
	"fmt"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

var MaxTimePerTurn = 30 * time.Second
var MaxTimePerEndgame = 15 * time.Second

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
func NewGameRunner(logchan chan string, cfg *config.Config) *GameRunner {
	r := &GameRunner{
		logchan:            logchan,
		config:             cfg,
		lexicon:            cfg.GetString(config.ConfigDefaultLexicon),
		letterDistribution: cfg.GetString(config.ConfigDefaultLetterDistribution),
	}
	r.Init([]AutomaticRunnerPlayer{
		{"", "", pb.BotRequest_HASTY_BOT, nil, 0, false},
		{"", "", pb.BotRequest_HASTY_BOT, nil, 0, false},
	})

	return r
}

type AutomaticRunnerPlayer struct {
	LeaveFile            string
	PEGFile              string
	BotCode              pb.BotRequest_BotCode
	BotSpec              *pb.BotSpec
	MinSimPlies          int
	StochasticStaticEval bool
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

	gd, err := kwg.GetKWG(r.config.WGLConfig(), r.lexicon)
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
			Config:               *r.config,
			PEGAdjustmentFile:    pegfile,
			LeavesFile:           leavefile,
			MinSimPlies:          players[idx].MinSimPlies,
			StochasticStaticEval: players[idx].StochasticStaticEval,
			BotSpec:              players[idx].BotSpec,
		}

		btp, err := bot.NewBotTurnPlayerFromGame(r.game, conf, botcode)
		if err != nil {
			return err
		}
		btp.MoveGenerator().(*movegen.GordonGenerator).SetGame(r.game)
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
	r.aiplayers[0].Reset()
	r.aiplayers[1].Reset()
	if r.aiplayers[0].GetBotType() == pb.BotRequest_FAST_ML_BOT ||
		r.aiplayers[1].GetBotType() == pb.BotRequest_FAST_ML_BOT {

		// If we are using a ML bot, it needs to have backup mode enabled
		// as it evaluates board positions by playing and unplaying moves.
		r.game.SetBackupMode(game.InteractiveGameplayMode)
		r.game.SetStateStackLength(1)
	}
}

func (r *GameRunner) Game() *game.Game {
	return r.game
}

func (r *GameRunner) genBestStaticTurn(playerIdx int) *move.Move {
	return aiturnplayer.GenBestStaticTurn(r.game, r.aiplayers[playerIdx], playerIdx)
}

func (r *GameRunner) genStochasticStaticTurn(playerIdx int) *move.Move {
	return aiturnplayer.GenStochasticStaticTurn(r.game, r.aiplayers[playerIdx], playerIdx)
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
	log.Debug().Int("playerIdx", playerIdx).
		Str("bestPlay", bestPlay.ShortDescription()).Msg("play-best-turn")

	// save rackLetters for logging.
	rackLetters := r.game.RackLettersFor(playerIdx)
	tilesRemaining := r.game.Bag().TilesRemaining()
	nickOnTurn := r.game.NickOnTurn()
	err := r.game.PlayMove(bestPlay, addToHistory, 0)
	if err != nil {
		return err
	}
	// Tell both players about the last move.
	r.aiplayers[0].AddLastMove(bestPlay)
	r.aiplayers[1].AddLastMove(bestPlay)

	// XXX: HERE need to get these somewhere.
	r.aiplayers[0].GetPertinentLogs()

	if r.logchan != nil {
		r.logchan <- fmt.Sprintf("%v,%v,%v,%v,%v,%v,%v,%v,%v,%.3f,%v,%v\n",
			r.game.Uid(),
			nickOnTurn,
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
