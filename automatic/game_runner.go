// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"
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

	// Shadow verification
	verifyShadow     bool
	divergenceCount  int64 // atomic counter
	totalTurnsPlayed int64 // atomic counter
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
		{"", "", pb.BotRequest_HASTY_BOT, 0, false},
		{"", "", pb.BotRequest_HASTY_BOT, 0, false},
	})

	return r
}

type AutomaticRunnerPlayer struct {
	LeaveFile            string
	PEGFile              string
	BotCode              pb.BotRequest_BotCode
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
	r.aiplayers[0].SetLastMoves(nil)
	r.aiplayers[1].SetLastMoves(nil)
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

	// Verify shadow algorithm if enabled (before playing the move)
	if r.verifyShadow && r.aiplayers[playerIdx].GetBotType() == pb.BotRequest_HASTY_BOT {
		r.verifyShadowEquity(playerIdx, bestPlay)
	}

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

// SetVerifyShadow enables shadow move generation verification.
func (r *GameRunner) SetVerifyShadow(verify bool) {
	r.verifyShadow = verify
}

// DivergenceCount returns the number of turns where shadow diverged.
func (r *GameRunner) DivergenceCount() int64 {
	return atomic.LoadInt64(&r.divergenceCount)
}

// TotalTurnsPlayed returns the total number of turns verified.
func (r *GameRunner) TotalTurnsPlayed() int64 {
	return atomic.LoadInt64(&r.totalTurnsPlayed)
}

// verifyShadowEquity verifies that shadow movegen produces the same top equity
// as non-shadow movegen. Since shadow is now enabled by default, this function
// disables shadow to compare against the normal (shadow-enabled) play.
func (r *GameRunner) verifyShadowEquity(playerIdx int, shadowPlay *move.Move) {
	atomic.AddInt64(&r.totalTurnsPlayed, 1)

	mg := r.aiplayers[playerIdx].MoveGenerator().(*movegen.GordonGenerator)
	rack := r.game.RackFor(playerIdx)
	oppRack := r.game.RackFor(1 - playerIdx)
	unseen := int(oppRack.NumTiles()) + r.game.Bag().TilesRemaining()
	exchAllowed := unseen-game.RackTileLimit >= r.game.ExchangeLimit()

	// Run with shadow DISABLED for verification (normal play used shadow)
	mg.SetShadowEnabled(false)
	mg.SetTopPlayOnlyRecorder()
	mg.SetMaxCanExchange(game.MaxCanExchange(unseen-game.RackTileLimit, r.game.ExchangeLimit()))
	mg.GenAll(rack, exchAllowed)

	nonShadowPlay := mg.Plays()[0]
	nonShadowEquity := nonShadowPlay.Equity()
	shadowEquity := shadowPlay.Equity()

	// Re-enable shadow for next turn
	mg.SetShadowEnabled(true)

	// Compare equities with small epsilon for float comparison
	if math.Abs(shadowEquity-nonShadowEquity) > 0.001 {
		atomic.AddInt64(&r.divergenceCount, 1)
		log.Warn().
			Float64("shadowEquity", shadowEquity).
			Float64("nonShadowEquity", nonShadowEquity).
			Str("shadowPlay", shadowPlay.ShortDescription()).
			Str("nonShadowPlay", nonShadowPlay.ShortDescription()).
			Str("rack", rack.String()).
			Int("turn", r.game.Turn()).
			Str("gameID", r.game.Uid()).
			Msg("shadow-divergence")
	}
}
