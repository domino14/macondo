package gameanalysis

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/preendgame"
)

// AnalysisConfig holds configuration for game analysis
type AnalysisConfig struct {
	// Early/mid game (>7 tiles in bag)
	SimPlaysEarlyMid int
	SimPliesEarlyMid int
	SimStopEarlyMid  int // 90, 95, 98, 99, or 999

	// Early pre-endgame (2-7 tiles in bag)
	SimPlaysEarlyPreEndgame int
	SimPliesEarlyPreEndgame int
	SimStopEarlyPreEndgame  int

	// Pre-endgame (1 tile in bag) - uses PEG solver
	PEGEarlyCutoff bool

	Threads int

	// Optional: analyze only one player (-1 = both, 0 = player 0, 1 = player 1, or player nickname)
	OnlyPlayer       int
	OnlyPlayerByName string
}

// DefaultAnalysisConfig returns sensible defaults
func DefaultAnalysisConfig() *AnalysisConfig {
	return &AnalysisConfig{
		SimPlaysEarlyMid:        40,
		SimPliesEarlyMid:        5,
		SimStopEarlyMid:         99,
		SimPlaysEarlyPreEndgame: 80,
		SimPliesEarlyPreEndgame: 10,
		SimStopEarlyPreEndgame:  99,
		PEGEarlyCutoff:          true,
		Threads:                 0, // 0 means use default
		OnlyPlayer:              -1,
	}
}

// Analyzer analyzes completed games
type Analyzer struct {
	cfg         *config.Config
	analysisCfg *AnalysisConfig
}

// New creates a new Analyzer
func New(cfg *config.Config, analysisCfg *AnalysisConfig) *Analyzer {
	if analysisCfg == nil {
		analysisCfg = DefaultAnalysisConfig()
	}
	return &Analyzer{
		cfg:         cfg,
		analysisCfg: analysisCfg,
	}
}

// AnalyzeGame analyzes every move in a game and returns the results
func (a *Analyzer) AnalyzeGame(ctx context.Context, history *pb.GameHistory) (*GameAnalysisResult, error) {
	if history == nil {
		return nil, errors.New("game history is nil")
	}

	result := &GameAnalysisResult{
		Turns: make([]*TurnAnalysis, 0),
		PlayerSummaries: [2]*PlayerSummary{
			{
				PlayerName:   history.Players[0].Nickname,
				TurnsPlayed:  0,
				OptimalMoves: 0,
			},
			{
				PlayerName:   history.Players[1].Nickname,
				TurnsPlayed:  0,
				OptimalMoves: 0,
			},
		},
	}

	// Build the game rules from the history
	// Use HistoryToVariant to properly extract board layout, letter distribution, and variant
	boardLayout, ldName, variant := game.HistoryToVariant(history)
	rules, err := game.NewBasicGameRules(
		a.cfg,
		history.Lexicon,
		boardLayout,
		ldName,
		game.CrossScoreAndSet,
		variant,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create game rules: %w", err)
	}

	// Set challenge rule to DOUBLE (not VOID) to allow phonies to be played during replay
	// Phonies will be handled by PHONY_TILES_RETURNED events in the history
	// This matches the behavior of the loadGCG function
	if history.ChallengeRule == pb.ChallengeRule_VOID {
		history.ChallengeRule = pb.ChallengeRule_DOUBLE
	}

	// Determine which player to analyze
	shouldAnalyzePlayer := func(playerIndex int) bool {
		// Check name filter first
		if a.analysisCfg.OnlyPlayerByName != "" {
			return history.Players[playerIndex].Nickname == a.analysisCfg.OnlyPlayerByName
		}
		// Then check player index filter
		if a.analysisCfg.OnlyPlayer == -1 {
			return true // Analyze both players
		}
		return playerIndex == a.analysisCfg.OnlyPlayer
	}

	// Track the previous event for phony detection
	var prevEvent *pb.GameEvent

	// Count total analyzable turns for progress reporting
	totalTurns := 0
	for _, evt := range history.Events {
		if a.isAnalyzableEvent(evt) {
			totalTurns++
		}
	}
	analyzedCount := 0

	// Analyze each turn
	for turnNum, evt := range history.Events {
		// Skip non-analyzable events
		if !a.isAnalyzableEvent(evt) {
			prevEvent = evt
			continue
		}

		playerIndex := int(evt.PlayerIndex)

		// Check if we should analyze this player
		if !shouldAnalyzePlayer(playerIndex) {
			prevEvent = evt
			continue
		}

		// Create a game at this turn
		g, err := game.NewFromHistory(history, rules, turnNum)
		if err != nil {
			return nil, fmt.Errorf("failed to create game at turn %d: %w", turnNum, err)
		}

		// Skip analyzing moves when the game is already over or waiting for final pass
		// This happens for the final pass after someone went out
		if g.Playing() != pb.PlayState_PLAYING {
			log.Debug().
				Int("turn", turnNum).
				Str("playState", g.Playing().String()).
				Msg("skipping turn - game not in playing state")
			prevEvent = evt
			continue
		}

		// Get the move that was played
		playedMove, err := game.MoveFromEvent(evt, g.Alphabet(), g.Board())
		if err != nil {
			return nil, fmt.Errorf("failed to create move from event at turn %d: %w", turnNum, err)
		}

		// Check if previous opponent move was a phony that wasn't challenged
		missedChallenge := false
		if prevEvent != nil && int(prevEvent.PlayerIndex) != playerIndex {
			if a.isPhony(prevEvent, g) {
				missedChallenge = true
			}
		}

		// Check if this move is a phony
		isPhony := a.isPhony(evt, g)
		phonyChallenged := false

		// Check if the next event is a PHONY_TILES_RETURNED for this move
		if turnNum+1 < len(history.Events) {
			nextEvt := history.Events[turnNum+1]
			if nextEvt.Type == pb.GameEvent_PHONY_TILES_RETURNED {
				phonyChallenged = true
				// Substitute with a pass move for analysis since the phony was challenged off
				playedMove = move.NewPassMove(g.RackFor(playerIndex).TilesOn(), g.Alphabet())
			}
		}

		// Analyze the position
		analysis, err := a.analyzeTurn(ctx, g, playedMove, turnNum, playerIndex, history.Players)
		if err != nil {
			log.Warn().Err(err).Int("turn", turnNum).Msg("failed to analyze turn")
			prevEvent = evt
			continue
		}

		// Add phony information
		analysis.IsPhony = isPhony
		analysis.PhonyChallenged = phonyChallenged
		analysis.MissedChallenge = missedChallenge

		result.Turns = append(result.Turns, analysis)

		// Log progress
		analyzedCount++
		log.Info().
			Int("player-turn", analyzedCount).
			Int("total", totalTurns).
			Str("player", analysis.PlayerName).
			Str("move", playedMove.ShortDescription()).
			Msg("analyzed turn")

		// Update player summary
		summary := result.PlayerSummaries[playerIndex]
		summary.TurnsPlayed++
		if analysis.WasOptimal {
			summary.OptimalMoves++
		}

		prevEvent = evt
	}

	// Calculate aggregate statistics
	a.calculatePlayerSummaries(result)

	return result, nil
}

// isAnalyzableEvent returns true if the event represents a move that can be analyzed
func (a *Analyzer) isAnalyzableEvent(evt *pb.GameEvent) bool {
	return evt.Type == pb.GameEvent_TILE_PLACEMENT_MOVE ||
		evt.Type == pb.GameEvent_EXCHANGE ||
		evt.Type == pb.GameEvent_PASS
}

// isPhony checks if a move is a phony by validating the words formed
func (a *Analyzer) isPhony(evt *pb.GameEvent, g *game.Game) bool {
	// Only tile placements can be phonies
	if evt.Type != pb.GameEvent_TILE_PLACEMENT_MOVE {
		return false
	}

	// If words_formed is populated, check each word
	if len(evt.WordsFormed) > 0 {
		lex := g.Lexicon()
		alph := g.Alphabet()
		for _, word := range evt.WordsFormed {
			mw, err := tilemapping.ToMachineWord(word, alph)
			if err != nil {
				continue
			}
			if !lex.HasWord(mw) {
				return true
			}
		}
		return false
	}

	// If words_formed is not populated, we can't determine phony status
	// This shouldn't happen in properly formatted GCG files
	return false
}

// analyzeTurn analyzes a single turn and returns the analysis
func (a *Analyzer) analyzeTurn(ctx context.Context, g *game.Game, playedMove *move.Move,
	turnNum, playerIndex int, players []*pb.PlayerInfo) (*TurnAnalysis, error) {

	tilesInBag := g.Bag().TilesRemaining()
	phase := a.determinePhase(tilesInBag)

	analysis := &TurnAnalysis{
		TurnNumber:  turnNum + 1, // 1-indexed for display
		PlayerIndex: playerIndex,
		PlayerName:  players[playerIndex].Nickname,
		Rack:        g.RackFor(playerIndex).String(),
		Phase:       phase,
		TilesInBag:  tilesInBag,
		PlayedMove:  playedMove,
	}

	var err error
	switch phase {
	case PhaseEarlyMid:
		err = a.analyzeWithSim(ctx, g, analysis, a.analysisCfg.SimPlaysEarlyMid,
			a.analysisCfg.SimPliesEarlyMid, a.analysisCfg.SimStopEarlyMid)
	case PhaseEarlyPreEndgame:
		err = a.analyzeWithSim(ctx, g, analysis, a.analysisCfg.SimPlaysEarlyPreEndgame,
			a.analysisCfg.SimPliesEarlyPreEndgame, a.analysisCfg.SimStopEarlyPreEndgame)
	case PhasePreEndgame:
		err = a.analyzeWithPEG(ctx, g, analysis)
	case PhaseEndgame:
		err = a.analyzeWithEndgame(ctx, g, analysis)
	}

	if err != nil {
		return nil, err
	}

	// Categorize the mistake
	analysis.MistakeCategory = categorizeMistake(analysis)

	return analysis, nil
}

// categorizeMistake categorizes a mistake as Small, Medium, or Large
func categorizeMistake(analysis *TurnAnalysis) string {
	if analysis.WasOptimal {
		return ""
	}

	// Handle endgame spread mistakes
	if analysis.Phase == PhaseEndgame {
		loss := float64(analysis.SpreadLoss)

		// If loss is essentially zero, don't categorize as a mistake
		const epsilon = 0.001
		if loss < epsilon {
			return ""
		}

		// Check for blown endgame: win→loss/tie or tie→loss
		// Calculate absolute final spreads by adding current spread
		optimalAbsoluteFinalSpread := int(analysis.OptimalFinalSpread) + analysis.CurrentSpread
		playedRelativeFinalSpread := analysis.OptimalFinalSpread - analysis.SpreadLoss
		playedAbsoluteFinalSpread := int(playedRelativeFinalSpread) + analysis.CurrentSpread

		if (optimalAbsoluteFinalSpread > 0 && playedAbsoluteFinalSpread <= 0) || // win → loss/tie
			(optimalAbsoluteFinalSpread == 0 && playedAbsoluteFinalSpread < 0) { // tie → loss
			analysis.BlownEndgame = true
			return "Large"
		}

		// Use doubled thresholds for endgame spread: 1-7 small, 8-15 medium, 16+ large
		if loss <= 7 {
			return "Small"
		} else if loss <= 15 {
			return "Medium"
		} else {
			return "Large"
		}
	}

	// Handle PEG spread tiebreak mistakes
	if analysis.Phase == PhasePreEndgame && analysis.SpreadLoss > 0 {
		loss := float64(analysis.SpreadLoss)

		// Use doubled thresholds for PEG spread: 1-7 small, 8-15 medium, 16+ large
		if loss <= 7 {
			return "Small"
		} else if loss <= 15 {
			return "Medium"
		} else {
			return "Large"
		}
	}

	// For sim/PEG win probability, use win probability loss as percentage
	loss := analysis.WinProbLoss * 100

	// If loss is essentially zero, don't categorize as a mistake
	const epsilon = 0.001
	if loss < epsilon {
		return ""
	}

	// Standard thresholds for win% loss: ≤3 small, 4-7 medium, >7 large
	if loss <= 3 {
		return "Small"
	} else if loss <= 7 {
		return "Medium"
	} else {
		return "Large"
	}
}

// mistakePoints returns the point value for a mistake category
func mistakePoints(category string) float64 {
	switch category {
	case "Small":
		return 0.2
	case "Medium":
		return 0.5
	case "Large":
		return 1.0
	default:
		return 0.0
	}
}

// estimateELO returns an estimated ELO rating based on the mistake index
// estimateELO estimates a player's ELO rating based on their mistake index
func estimateELO(mistakeIndex float64) string {
	// Data points: (mistakeIndex, elo)
	dataPoints := []struct {
		mi  float64
		elo int
	}{
		{0.0, 2300}, {0.2, 2250}, {0.5, 2200}, {0.8, 2150},
		{1.2, 2100}, {1.5, 2050}, {1.7, 2000}, {1.9, 1950},
		{2.3, 1900}, {2.6, 1850}, {2.9, 1800}, {3.3, 1750},
		{3.8, 1700}, {4.2, 1650},
	}

	// Handle edge cases
	if mistakeIndex <= 0 {
		return "2300"
	}
	if mistakeIndex > 4.2 {
		// Use formula: 1650 - 125(x - 4.2)
		elo := 1650.0 - 125.0*(mistakeIndex-4.2)
		return fmt.Sprintf("%d", int(elo+0.5)) // Round to nearest int
	}

	// Find bracketing points and interpolate
	for i := 1; i < len(dataPoints); i++ {
		if mistakeIndex <= dataPoints[i].mi {
			lower := dataPoints[i-1]
			upper := dataPoints[i]
			// Linear interpolation
			t := (mistakeIndex - lower.mi) / (upper.mi - lower.mi)
			elo := float64(lower.elo) + t*(float64(upper.elo)-float64(lower.elo))
			return fmt.Sprintf("%d", int(elo+0.5)) // Round to nearest int
		}
	}
	return "1650"
}

// determinePhase determines the game phase based on tiles in bag
func (a *Analyzer) determinePhase(tilesInBag int) GamePhase {
	if tilesInBag > 7 {
		return PhaseEarlyMid
	} else if tilesInBag >= 2 {
		return PhaseEarlyPreEndgame
	} else if tilesInBag == 1 {
		return PhasePreEndgame
	}
	return PhaseEndgame
}

// analyzeWithSim analyzes a turn using Monte Carlo simulation
func (a *Analyzer) analyzeWithSim(ctx context.Context, g *game.Game, analysis *TurnAnalysis,
	numPlays, plies, stopCondition int) error {

	// Create bot turn player for move generation
	botConfig := &bot.BotConfig{Config: *a.cfg}
	botPlayer, err := bot.NewBotTurnPlayerFromGame(g, botConfig, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return fmt.Errorf("failed to create bot player: %w", err)
	}

	// Generate top moves
	plays := botPlayer.GenerateMoves(numPlays)
	if len(plays) == 0 {
		return errors.New("no moves generated")
	}

	// Create simmer
	equityCalc, err := equity.NewCombinedStaticCalculator(
		g.LexiconName(), a.cfg, "", equity.PEGAdjustmentFilename)
	if err != nil {
		return fmt.Errorf("failed to create equity calculator: %w", err)
	}

	simmer := &montecarlo.Simmer{}
	simmer.Init(botPlayer.Game, []equity.EquityCalculator{equityCalc}, equityCalc, a.cfg)

	if a.analysisCfg.Threads > 0 {
		simmer.SetThreads(a.analysisCfg.Threads)
	}

	// Prepare simulation
	err = simmer.PrepareSim(plies, plays)
	if err != nil {
		return fmt.Errorf("failed to prepare simulation: %w", err)
	}

	// Ensure the played move is simmed (avoid pruning it)
	simmer.AvoidPruningMoves([]*move.Move{analysis.PlayedMove})

	// Set stopping condition
	var stop montecarlo.StoppingCondition
	switch stopCondition {
	case 90:
		stop = montecarlo.Stop90
	case 95:
		stop = montecarlo.Stop95
	case 98:
		stop = montecarlo.Stop98
	case 99:
		stop = montecarlo.Stop99
	case 999:
		stop = montecarlo.Stop999
	default:
		stop = montecarlo.Stop99
	}
	simmer.SetStoppingCondition(stop)

	// Run simulation
	err = simmer.Simulate(ctx)
	if err != nil {
		return fmt.Errorf("simulation failed: %w", err)
	}

	// Get results sorted by win probability
	results := simmer.PlaysByWinProb()
	if results == nil {
		return errors.New("simulation produced no results")
	}
	simmedPlays := results.PlaysNoLock()
	if len(simmedPlays) == 0 {
		return errors.New("simulation produced no results")
	}

	// Find optimal move (highest win prob)
	analysis.OptimalMove = simmedPlays[0].Move()
	analysis.OptimalWinProb = simmedPlays[0].WinProb()

	// Find played move in results
	playedFound := false
	for _, result := range simmedPlays {
		if result.Move().Equals(analysis.PlayedMove, false, true) {
			analysis.PlayedWinProb = result.WinProb()
			playedFound = true
			break
		}
	}

	if !playedFound {
		// This shouldn't happen if AvoidPruningMoves worked
		log.Warn().Msg("played move not found in sim results")
		analysis.PlayedWinProb = 0
	}

	analysis.WinProbLoss = analysis.OptimalWinProb - analysis.PlayedWinProb
	analysis.WasOptimal = analysis.OptimalMove.Equals(analysis.PlayedMove, false, true)

	return nil
}

// analyzeWithPEG analyzes a turn using the PEG solver (1 tile in bag)
func (a *Analyzer) analyzeWithPEG(ctx context.Context, g *game.Game, analysis *TurnAnalysis) error {
	// Get the KWG for the lexicon
	gd, err := kwg.GetKWG(a.cfg.WGLConfig(), g.LexiconName())
	if err != nil {
		return fmt.Errorf("failed to get KWG: %w", err)
	}

	// Create PEG solver
	pegSolver := &preendgame.Solver{}
	pegSolver.Init(g, gd)

	// Set options
	pegSolver.SetEarlyCutoffOptim(a.analysisCfg.PEGEarlyCutoff)
	pegSolver.SetAvoidPrune([]*move.Move{analysis.PlayedMove})

	if a.analysisCfg.Threads > 0 {
		pegSolver.SetThreads(a.analysisCfg.Threads)
	}

	// Create timeout context for PEG solving (5 minutes)
	pegCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	// Solve
	plays, err := pegSolver.Solve(pegCtx)
	if err != nil {
		return fmt.Errorf("PEG solve failed: %w", err)
	}

	// Get results
	if len(plays) == 0 {
		return errors.New("PEG solver produced no results")
	}

	// Find optimal move (first in sorted list)
	optimalPlay := plays[0]
	analysis.OptimalMove = optimalPlay.Play

	// Determine total number of possible outcomes
	// Use TotalOutcomes() from the optimal play for accurate denominator
	noutcomes := float32(optimalPlay.TotalOutcomes())
	if noutcomes == 0 {
		// Fallback to bag tiles if no outcomes recorded (shouldn't happen)
		noutcomes = float32(g.Bag().TilesRemaining())
	}

	analysis.OptimalWinProb = float64(optimalPlay.Points / noutcomes)

	// Find played move
	var playedPlay *preendgame.PreEndgamePlay
	for _, play := range plays {
		if play.Play.Equals(analysis.PlayedMove, false, true) {
			playedPlay = play
			break
		}
	}

	if playedPlay == nil {
		return errors.New("played move not found in PEG results")
	}

	analysis.PlayedWinProb = float64(playedPlay.Points / noutcomes)
	analysis.WinProbLoss = analysis.OptimalWinProb - analysis.PlayedWinProb
	analysis.WasOptimal = analysis.OptimalMove.Equals(analysis.PlayedMove, false, true)

	log.Info().
		Str("optimalMove", optimalPlay.Play.ShortDescription()).
		Float64("optimalWinProb", analysis.OptimalWinProb).
		Str("playedMove", playedPlay.Play.ShortDescription()).
		Float64("playedWinProb", analysis.PlayedWinProb).
		Float64("winProbLoss", analysis.WinProbLoss).
		Bool("wasOptimal", analysis.WasOptimal).
		Msg("peg-analysis")

	// When win probabilities are tied (within epsilon), use spread data for comparison
	const epsilon = 1e-6
	if !analysis.WasOptimal && analysis.WinProbLoss < epsilon && analysis.WinProbLoss > -epsilon {
		// Win probabilities are essentially tied - check spread data
		if optimalPlay.HasSpread() && playedPlay.HasSpread() {
			// Calculate average spreads
			optimalAvgSpread := float64(optimalPlay.GetSpread()) / float64(optimalPlay.TotalOutcomes())
			playedAvgSpread := float64(playedPlay.GetSpread()) / float64(playedPlay.TotalOutcomes())

			spreadDiff := optimalAvgSpread - playedAvgSpread

			log.Info().
				Str("optimalMove", optimalPlay.Play.ShortDescription()).
				Int("optimalTotalSpread", optimalPlay.GetSpread()).
				Int("optimalTotalOutcomes", optimalPlay.TotalOutcomes()).
				Float64("optimalAvgSpread", optimalAvgSpread).
				Str("playedMove", playedPlay.Play.ShortDescription()).
				Int("playedTotalSpread", playedPlay.GetSpread()).
				Int("playedTotalOutcomes", playedPlay.TotalOutcomes()).
				Float64("playedAvgSpread", playedAvgSpread).
				Float64("spreadDiff", spreadDiff).
				Msg("peg-spread-tiebreak-analysis")

			if spreadDiff > epsilon {
				// Optimal has better average spread
				analysis.SpreadLoss = int16(spreadDiff + 0.5) // Round to nearest int
				// Keep WinProbLoss at ~0 since win% are tied
			}
		} else {
			log.Info().
				Bool("optimalHasSpread", optimalPlay.HasSpread()).
				Bool("playedHasSpread", playedPlay.HasSpread()).
				Msg("peg-spread-data-missing")
		}
	}

	return nil
}

// analyzeWithEndgame analyzes a turn using the endgame solver (0 tiles in bag)
func (a *Analyzer) analyzeWithEndgame(ctx context.Context, g *game.Game, analysis *TurnAnalysis) error {
	// Create bot turn player for move generation
	botConfig := &bot.BotConfig{Config: *a.cfg}
	botPlayer, err := bot.NewBotTurnPlayerFromGame(g, botConfig, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return fmt.Errorf("failed to create bot player: %w", err)
	}

	// Set backup mode like the shell does
	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(1)
	defer func() {
		g.SetBackupMode(game.InteractiveGameplayMode)
	}()

	// Capture current spread before the move for blown endgame detection
	playerOnTurn := g.PlayerOnTurn()
	analysis.CurrentSpread = g.SpreadFor(playerOnTurn)

	// Create endgame solver
	endgameSolver := &negamax.Solver{}
	err = endgameSolver.Init(botPlayer.MoveGenerator(), botPlayer.Game)
	if err != nil {
		return fmt.Errorf("failed to init endgame solver: %w", err)
	}

	// Set option to also solve the played move
	endgameSolver.SetAlsoSolveMove(analysis.PlayedMove)

	// Create timeout context for endgame solving (3 minutes)
	endgameCtx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()

	// Solve
	plies := 7 // default endgame plies
	bestSpread, principalVariation, err := endgameSolver.Solve(endgameCtx, plies)
	if err != nil {
		return fmt.Errorf("endgame solve failed: %w", err)
	}

	// Get optimal move (first move in principal variation)
	if len(principalVariation) == 0 {
		return errors.New("no principal variation found")
	}
	analysis.OptimalMove = principalVariation[0]
	analysis.OptimalFinalSpread = bestSpread

	// Check if the played move is optimal
	analysis.WasOptimal = analysis.PlayedMove.Equals(analysis.OptimalMove, false, true)

	if analysis.WasOptimal {
		analysis.SpreadLoss = 0
	} else {
		// Find the played move in the variations
		// SetAlsoSolveMove ensures it was solved
		variations := endgameSolver.Variations()
		playedFound := false
		for _, variation := range variations {
			if variation.NumMoves() > 0 && analysis.PlayedMove.Equals(variation.Moves[0], false, true) {
				// SpreadLoss is how much worse this move is
				analysis.SpreadLoss = bestSpread - variation.Score()
				playedFound = true
				break
			}
		}
		if !playedFound {
			log.Warn().Str("move", analysis.PlayedMove.ShortDescription()).
				Msg("played move not found in endgame variations")
			// This shouldn't happen if SetAlsoSolveMove worked correctly
			return fmt.Errorf("played move not found in variations")
		}
	}

	return nil
}

// calculatePlayerSummaries calculates aggregate statistics for each player
func (a *Analyzer) calculatePlayerSummaries(result *GameAnalysisResult) {
	for i := 0; i < 2; i++ {
		summary := result.PlayerSummaries[i]
		if summary.TurnsPlayed == 0 {
			continue
		}

		var totalWinProbLoss float64
		var totalSpreadLoss float64
		winProbCount := 0
		spreadCount := 0

		for _, turn := range result.Turns {
			if turn.PlayerIndex != i {
				continue
			}

			// Count win prob loss for non-endgame phases
			if turn.Phase != PhaseEndgame {
				totalWinProbLoss += turn.WinProbLoss
				winProbCount++
			}

			// Count spread loss for endgame AND PEG with spread tiebreaks
			if turn.Phase == PhaseEndgame || (turn.Phase == PhasePreEndgame && turn.SpreadLoss > 0) {
				totalSpreadLoss += float64(turn.SpreadLoss)
				spreadCount++
			}

			// Add mistake points to the mistake index
			summary.MistakeIndex += mistakePoints(turn.MistakeCategory)
		}

		if winProbCount > 0 {
			summary.AvgWinProbLoss = totalWinProbLoss / float64(winProbCount)
		}
		if spreadCount > 0 {
			summary.AvgSpreadLoss = totalSpreadLoss / float64(spreadCount)
		}

		// Calculate estimated ELO based on mistake index
		summary.EstimatedELO = estimateELO(summary.MistakeIndex)
	}
}
