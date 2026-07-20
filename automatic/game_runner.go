// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"context"
	"fmt"
	"runtime"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/ai/externalengine"
	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
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
	// threads is the number of concurrent games in the parent autoplay run.
	// Used to divide the TT memory fraction and default endgame thread counts.
	threads int
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
		{BotCode: pb.BotRequest_HASTY_BOT},
		{BotCode: pb.BotRequest_HASTY_BOT},
	})

	return r
}

type AutomaticRunnerPlayer struct {
	LeaveFile            string
	PEGFile              string
	BotCode              pb.BotRequest_BotCode
	MinSimPlies          int
	SimThreads           int
	StochasticStaticEval bool
	InferenceTau                float64
	InferenceTimeSecs           int
	InferenceSimIters           int
	InferenceMaxEnumeratedLeaves int
	OracleInference             bool
	ExternalEngine              *externalengine.Config // nil unless BotCode == CUSTOM_BOT
	// EndgameThreads / PreendgameThreads: threads inside a single solve.
	// 0 means "default" — the runner resolves it to max(1, NumCPU/concurrentGames).
	EndgameThreads    int
	PreendgameThreads int
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

	// Create one transposition table shared by both players in this game.
	// Players take turns and never solve concurrently, so sharing is safe.
	// Sized at totalFraction/threads so N concurrent games sum to ~totalFraction.
	concurrentGames := r.threads
	if concurrentGames < 1 {
		concurrentGames = 1
	}
	totalFraction := r.config.GetFloat64(config.ConfigTtableMemFraction)
	perGameFraction := totalFraction / float64(concurrentGames)
	// Default per-solve thread count: spread CPU across concurrent games.
	defaultSolveThreads := max(1, runtime.NumCPU()/concurrentGames)

	// One TT per game, shared across both players (allocated lazily by each
	// bot's first Solve, but we pre-create it here so both players use the
	// same instance).
	gameTT := &negamax.TranspositionTable{}

	for idx := range players {
		leavefile := players[idx].LeaveFile
		pegfile := players[idx].PEGFile
		botcode := players[idx].BotCode
		log.Info().Msgf("botcode %v", botcode)

		if botcode == pb.BotRequest_CUSTOM_BOT {
			if players[idx].ExternalEngine == nil {
				return fmt.Errorf("player %d: CUSTOM_BOT requires ExternalEngine config", idx)
			}
			ep, err := externalengine.NewFromGame(r.game, r.config, *players[idx].ExternalEngine)
			if err != nil {
				return err
			}
			ep.MoveGenerator().(*movegen.GordonGenerator).SetGame(r.game)
			r.aiplayers[idx] = ep
			continue
		}

		// Resolve per-solve thread counts: use explicit value from config if
		// set, otherwise spread CPU evenly across concurrent games.
		simThreads := players[idx].SimThreads
		if simThreads == 0 {
			simThreads = defaultSolveThreads
		}
		egThreads := players[idx].EndgameThreads
		if egThreads == 0 {
			egThreads = defaultSolveThreads
		}
		pegThreads := players[idx].PreendgameThreads
		if pegThreads == 0 {
			pegThreads = defaultSolveThreads
		}

		conf := &bot.BotConfig{
			Config:               *r.config,
			PEGAdjustmentFile:    pegfile,
			LeavesFile:           leavefile,
			MinSimPlies:          players[idx].MinSimPlies,
			SimThreads:           simThreads,
			StochasticStaticEval: players[idx].StochasticStaticEval,
			InferenceTau:                players[idx].InferenceTau,
			InferenceTimeSecs:           players[idx].InferenceTimeSecs,
			InferenceSimIters:           players[idx].InferenceSimIters,
			InferenceMaxEnumeratedLeaves: players[idx].InferenceMaxEnumeratedLeaves,
			OracleInference:             players[idx].OracleInference,
			EndgameTTable:               gameTT,
			EndgameTTableFraction:       perGameFraction,
			EndgameThreads:              egThreads,
			PreendgameThreads:           pegThreads,
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
	r.StartGameWithSeed(gidx, [32]byte{})
}

func (r *GameRunner) StartGameWithSeed(gidx int, seed [32]byte) {
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
	// Seed before starting if seed is non-zero
	var zeroSeed [32]byte
	if seed != zeroSeed {
		r.game.SeedBag(seed)
	}
	r.game.StartGame()
	// Set deterministic game ID if seeded
	if seed != zeroSeed {
		r.game.SetUidFromSeed(seed)
	}
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

// BotTypeFor returns the bot code for the given game-level player index.
// Accounts for player flips done by StartGameWithSeed.
func (r *GameRunner) BotTypeFor(playerIdx int) pb.BotRequest_BotCode {
	return r.aiplayers[playerIdx].GetBotType()
}

func (r *GameRunner) genBestStaticTurn(playerIdx int) *move.Move {
	return aiturnplayer.GenBestStaticTurn(r.game, r.aiplayers[playerIdx], playerIdx)
}

func (r *GameRunner) genStochasticStaticTurn(playerIdx int) *move.Move {
	return aiturnplayer.GenStochasticStaticTurn(r.game, r.aiplayers[playerIdx], playerIdx)
}

func (r *GameRunner) genBestMoveForBot(playerIdx int) (*move.Move, error) {
	if r.aiplayers[playerIdx].GetBotType() == pb.BotRequest_HASTY_BOT {
		// For HastyBot we only need to generate one single best static turn.
		return r.genBestStaticTurn(playerIdx), nil
	}
	var ctx context.Context
	var cancel context.CancelFunc
	if r.aiplayers[playerIdx].GetBotType() == pb.BotRequest_CUSTOM_BOT {
		ctx, cancel = context.WithCancel(context.Background())
	} else {
		maxTime := MaxTimePerTurn
		if r.game.Bag().TilesRemaining() == 0 {
			log.Debug().Msg("runner-bag-is-empty")
			maxTime = MaxTimePerEndgame
		}
		ctx, cancel = context.WithTimeout(context.Background(), maxTime)
	}
	defer cancel()
	return r.aiplayers[playerIdx].BestPlay(ctx)
}

// PlayBestTurn generates the best move for the player and plays it on the board.
func (r *GameRunner) PlayBestTurn(playerIdx int, addToHistory bool) error {
	bestPlay, err := r.genBestMoveForBot(playerIdx)
	if err != nil {
		return fmt.Errorf("player %d: %w", playerIdx, err)
	}
	log.Debug().Int("playerIdx", playerIdx).
		Str("bestPlay", bestPlay.ShortDescription()).Msg("play-best-turn")

	// save rackLetters for logging.
	rackLetters := r.game.RackLettersFor(playerIdx)
	tilesRemaining := r.game.Bag().TilesRemaining()
	nickOnTurn := r.game.NickOnTurn()
	err = r.game.PlayMove(bestPlay, addToHistory, 0)
	if err != nil {
		return err
	}
	// Tell both players about the last move.
	r.aiplayers[0].AddLastMove(bestPlay)
	r.aiplayers[1].AddLastMove(bestPlay)

	if r.logchan != nil {
		inferCount := ""
		if btp, ok := r.aiplayers[playerIdx].(*bot.BotTurnPlayer); ok {
			ic := btp.LastInferenceCount()
			if ic >= 0 {
				inferCount = fmt.Sprintf(",%v", ic)
			}
		}
		r.logchan <- fmt.Sprintf("%v,%v,%v,%v,%v,%v,%v,%v,%v,%.3f,%v,%v%v\n",
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
			r.game.PointsFor((playerIdx+1)%2),
			inferCount)
	}
	return nil
}
