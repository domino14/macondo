// Package automatic contains all the logic for the actual gameplay
// of Crossword Game, which, as we said before, features all sorts of
// things like wingos and blonks.
package automatic

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
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

	// gamePairs makes each seeded game draw tiles from a fixed bag order, so
	// that the same seed played twice with the seats swapped deals both bots
	// the same tiles. See game/pairedbag.go.
	gamePairs bool
	// movesPlayed records the game in progress move by move so that the two
	// games of a pair can be compared. Only filled in while pairing.
	movesPlayed []*move.Move
	recordMoves bool

	// logInference is true when some bot in this run infers, so every row of the
	// per-turn log carries the inference columns -- empty ones for the bot that
	// does not infer. Keeping the rows the same width is what makes the file a
	// table rather than two interleaved shapes.
	logInference bool

	// Random openings (see OpeningConfig). openingPlies is this game's K, drawn
	// at StartGameWithSeed; openingRng is seeded from the game seed when there
	// is one, so a replayed or paired game samples the same opening.
	opening      OpeningConfig
	openingPlies int
	openingRng   *rand.Rand
}

// OpeningConfig makes each game's first K plies sampled instead of played by
// the bots, K ~ round(Exp(Mean)) per game. A sampled ply is a softmax over
// static equity (Temperature in points) across the TopN best plays, or with
// probability UniformProb a uniform draw over every legal play and exchange.
// Both seats get it, whatever their bot type. The per-turn log carries K on
// every row of the game (the `openingplies` column), so a training consumer
// can take the position after the last sampled ply as its first eligible one
// and know that no sampled move sits in the future of any label.
//
// This is the label-clean version of temperature self-play: the randomness
// creates unusual states, but the play that decides their labels is the
// bots' own.
type OpeningConfig struct {
	Mean        float64
	Temperature float64
	TopN        int
	UniformProb float64
}

// Enabled reports whether any opening plies are ever sampled.
func (o OpeningConfig) Enabled() bool { return o.Mean > 0 }

// drawOpeningPlies draws K ~ round(Exp(mean)).
func drawOpeningPlies(mean float64, rng *rand.Rand) int {
	if mean <= 0 {
		return 0
	}
	return int(math.Round(rng.ExpFloat64() * mean))
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
	FixedSimPlies        int
	TritonModel          string
	SimThreads           int
	StochasticStaticEval bool
	// InferenceTau pins the softmax temperature for P(play | leave). Leave it
	// zero for the engine default (SoftmaxTemperature); a value here applies to
	// the whole game.
	InferenceTau                 float64
	InferenceTimeSecs            int
	InferenceSimIters            int
	InferenceMaxEnumeratedLeaves int
	InferenceBudget              int
	// QuickEndgamePlies > 0: close endgames (|spread| <= QuickEndgameMargin,
	// 0 = always) are played with a quick N-ply search capped at
	// QuickEndgameCapMs per move. See AutoplayPlayerConfig.
	QuickEndgamePlies  int
	QuickEndgameMargin int
	QuickEndgameCapMs  int
	// QuickEndgameNoPrune: no pruned word graph for the quick solves.
	QuickEndgameNoPrune bool
	OracleInference     bool
	ExternalEngine      *externalengine.Config // nil unless BotCode == CUSTOM_BOT
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
			Config:                       *r.config,
			PEGAdjustmentFile:            pegfile,
			LeavesFile:                   leavefile,
			MinSimPlies:                  players[idx].MinSimPlies,
			FixedSimPlies:                players[idx].FixedSimPlies,
			TritonModel:                  players[idx].TritonModel,
			SimThreads:                   simThreads,
			StochasticStaticEval:         players[idx].StochasticStaticEval,
			InferenceTau:                 players[idx].InferenceTau,
			InferenceTimeSecs:            players[idx].InferenceTimeSecs,
			InferenceSimIters:            players[idx].InferenceSimIters,
			InferenceMaxEnumeratedLeaves: players[idx].InferenceMaxEnumeratedLeaves,
			InferenceBudget:              players[idx].InferenceBudget,
			QuickEndgamePlies:            players[idx].QuickEndgamePlies,
			QuickEndgameMargin:           players[idx].QuickEndgameMargin,
			QuickEndgameCapMs:            players[idx].QuickEndgameCapMs,
			QuickEndgameNoPrune:          players[idx].QuickEndgameNoPrune,
			OracleInference:              players[idx].OracleInference,
			EndgameTTable:                gameTT,
			EndgameTTableFraction:        perGameFraction,
			EndgameThreads:               egThreads,
			PreendgameThreads:            pegThreads,
		}

		btp, err := bot.NewBotTurnPlayerFromGame(r.game, conf, botcode)
		if err != nil {
			return err
		}
		btp.MoveGenerator().(*movegen.GordonGenerator).SetGame(r.game)
		r.aiplayers[idx] = btp

	}
	r.order = [2]int{0, 1}
	for idx := range players {
		if bot.HasInfer(players[idx].BotCode) {
			r.logInference = true
		}
	}
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
	// Paired draws are only worth anything on top of a seed: the bag order and
	// the exchange re-inserts both come out of the seeded RNG.
	r.game.SetPairedBagMode(r.gamePairs && seed != zeroSeed)
	if seed != zeroSeed {
		r.game.SeedBag(seed)
	}
	if r.opening.Enabled() {
		if seed != zeroSeed {
			// The seed pins the opening too, so a pair's two halves and a
			// replayed run sample the same opening moves.
			r.openingRng = rand.New(rand.NewSource(int64(binary.LittleEndian.Uint64(seed[:8]))))
		} else if r.openingRng == nil {
			r.openingRng = rand.New(rand.NewSource(rand.Int63()))
		}
		r.openingPlies = drawOpeningPlies(r.opening.Mean, r.openingRng)
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
	maxTime := MaxTimePerTurn
	endgame := r.game.Bag().TilesRemaining() == 0
	if endgame {
		log.Debug().Msg("runner-bag-is-empty")
		maxTime = MaxTimePerEndgame
	}
	var ctx context.Context
	var cancel context.CancelFunc
	if r.aiplayers[playerIdx].GetBotType() == pb.BotRequest_CUSTOM_BOT {
		// The external engine process manages its own time.
		ctx, cancel = context.WithCancel(context.Background())
	} else if r.game.PairedBagMode() && !endgame {
		// A wall-clock budget makes the bot's choice depend on how busy the
		// machine happened to be, which is the sort of noise game pairs exist
		// to get rid of. A sim stops on its own iteration cutoff, so dropping
		// the deadline still leaves something to stop it. The endgame solver
		// searches until it is interrupted, so it keeps its budget -- and stays
		// the one part of a paired game that does not replay exactly.
		ctx, cancel = context.WithCancel(context.Background())
	} else {
		ctx, cancel = context.WithTimeout(context.Background(), maxTime)
	}
	defer cancel()
	m, err := r.aiplayers[playerIdx].BestPlay(ctx)
	if bt := r.aiplayers[playerIdx].GetBotType(); err != nil &&
		bt != pb.BotRequest_CUSTOM_BOT && bt != pb.BotRequest_FAST_ML_BOT {
		// A built-in bot that fails (a solver error, say) plays its best
		// static move, so the game still finishes and counts. An external
		// engine or the ML bot failing is a broken setup (for the ML bot, an
		// unserved model, which would otherwise quietly play HastyBot's
		// moves), and stops the run.
		log.Err(err).Int("playerIdx", playerIdx).Msg("bot-error-playing-static-move")
		return r.genBestStaticTurn(playerIdx), nil
	}
	return m, err
}

// sampleOpeningMove picks this turn's move under the opening config instead
// of asking the bot: a uniform draw over every legal play (UniformProb), else
// a softmax over static equity across the TopN best plays.
func (r *GameRunner) sampleOpeningMove(playerIdx int) *move.Move {
	// Any static-capable player can sample: the built-in bots and the
	// external engine (which keeps a static generator of its own).
	ai := r.aiplayers[playerIdx].(interface {
		MoveGenerator() movegen.MoveGenerator
		GenerateMoves(int) []*move.Move
	})
	// HastyBot leaves its generator recording only the top play (with shadow
	// pruning); sampling needs every play, ranked. HastyBot sets its recorder
	// again on its next turn, so this need not be undone.
	ai.MoveGenerator().SetPlayRecorder(movegen.AllPlaysRecorder)
	if r.opening.UniformProb > 0 && r.openingRng.Float64() < r.opening.UniformProb {
		all := ai.GenerateMoves(math.MaxInt32)
		return all[r.openingRng.Intn(len(all))]
	}
	topN := r.opening.TopN
	if topN <= 0 {
		topN = 50
	}
	moves := ai.GenerateMoves(topN) // sorted best first
	m, err := bot.ChooseMoveWithExplorationRand(moves, r.opening.Temperature, r.openingRng.Float64)
	if err != nil {
		log.Err(err).Msg("sampling an opening move; playing the best one")
		return moves[0]
	}
	return m
}

// PlayBestTurn generates the best move for the player and plays it on the board.
func (r *GameRunner) PlayBestTurn(playerIdx int, addToHistory bool) error {
	var bestPlay *move.Move
	var err error
	if r.game.Turn() < r.openingPlies {
		bestPlay = r.sampleOpeningMove(playerIdx)
	} else {
		bestPlay, err = r.genBestMoveForBot(playerIdx)
		if err != nil {
			return fmt.Errorf("player %d: %w", playerIdx, err)
		}
	}
	log.Debug().Int("playerIdx", playerIdx).
		Str("bestPlay", bestPlay.ShortDescription()).Msg("play-best-turn")

	if r.recordMoves {
		// Take a copy, not the pointer. A move generator hands back the same
		// move object every turn -- it fills in one reusable "winner" and
		// returns that -- so storing pointers would leave us holding one move
		// per bot, showing whatever those two objects were last written with,
		// and comparing the halves of a pair would be meaningless.
		recorded := &move.Move{}
		recorded.CopyFrom(bestPlay)
		r.movesPlayed = append(r.movesPlayed, recorded)
	}

	// Grade the inference before the move lands. ExtractLastOppLeave reads the
	// most recent event in the history, which is the opponent's move only until
	// ours is played.
	inferFields := r.inferenceFields(playerIdx)

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
		openingField := ""
		if r.opening.Enabled() {
			openingField = fmt.Sprintf(",%d", r.openingPlies)
		}
		r.logchan <- fmt.Sprintf("%v,%v,%v,%v,%v,%v,%v,%v,%v,%.3f,%v,%v%v%v\n",
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
			inferFields,
			openingField)
	}
	return nil
}

// InferenceLogColumns are the per-turn log columns describing one inference.
// They are present on every row of a run that has an inferring bot, and empty on
// the rows of a bot that does not infer. See inferenceFields.
const InferenceLogColumns = ",inferCount,trueLeave,truePost,truePrior,liftBits,trueRank,inferLeaves,trueMeasured"

// emptyInferenceFields fills those columns in for a row that has no inference
// behind it: eight empty values, so the row is still as wide as the header.
const emptyInferenceFields = ",,,,,,,,"

// inferenceFields returns those columns for this bot's most recent inference, or
// "" for a bot that does not infer.
//
// Autoplay knows the opponent's real rack, so inference can be graded here
// rather than only counted: truePost is the probability the posterior placed on
// the leave the opponent actually held, truePrior is what the tile counts alone
// would have said, and liftBits is log2 of their ratio -- the information
// inference added about the truth. Averaging liftBits over a run gives one
// number to compare tau values or budgets by. See rangefinder.LeaveScore.
func (r *GameRunner) inferenceFields(playerIdx int) string {
	if !r.logInference {
		return "" // no bot here infers, so the log has no such columns
	}
	btp, ok := r.aiplayers[playerIdx].(*bot.BotTurnPlayer)
	if !ok {
		return emptyInferenceFields
	}
	ic := btp.LastInferenceCount()
	if ic < 0 {
		return emptyInferenceFields // this bot has no inferencer at all
	}
	// Every early exit still has to emit the same number of fields, or the log
	// stops being a table.
	unscored := fmt.Sprintf(",%d,,,,,,,", ic)
	trueLeave, err := game.ExtractLastOppLeave(r.game)
	if err != nil {
		return unscored
	}
	score, ok := btp.ScoreLastInference(trueLeave)
	if !ok {
		return unscored
	}
	return fmt.Sprintf(",%d,%s,%.6g,%.6g,%.4f,%d,%d,%v",
		ic,
		tilemapping.MachineWord(trueLeave).UserVisible(r.alphabet),
		score.Posterior, score.Prior, score.LiftBits,
		score.Rank, score.Leaves, score.Measured)
}
