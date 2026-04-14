package gameanalysis

import (
	"context"
	"errors"
	"fmt"
	"strings"
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

// CurrentAnalysisVersion is incremented when the analysis output format gains new fields.
// The browse table shows ✓ for versions >= 2.
const CurrentAnalysisVersion = 2

// blowoutWinProbThreshold is the win probability threshold below which (or above
// 1 minus which) the game is considered a blowout. Matches the sim autostopper's
// minReasonableWProb — at these extremes win% is meaningless and equity/spread
// is used instead.
const blowoutWinProbThreshold = 0.005

// Spread-loss thresholds for mistake categorization.
const (
	spreadSmall  = 7
	spreadMedium = 15

	// Doubled thresholds used for early/mid blowout positions.
	spreadSmallBlowout  = 15
	spreadMediumBlowout = 30
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

	// UseExposedOppRacks enables using opponent rack information revealed
	// through challenged phonies when analyzing the subsequent turn.
	UseExposedOppRacks bool
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
		UseExposedOppRacks:      true, // Enable by default for more accurate analysis
	}
}

// Analyzer analyzes completed games
type Analyzer struct {
	cfg         *config.Config
	analysisCfg *AnalysisConfig
	version     string
}

// New creates a new Analyzer
func New(cfg *config.Config, analysisCfg *AnalysisConfig, version string) *Analyzer {
	if analysisCfg == nil {
		analysisCfg = DefaultAnalysisConfig()
	}
	return &Analyzer{
		cfg:         cfg,
		analysisCfg: analysisCfg,
		version:     version,
	}
}

// AnalyzeSingleTurnFromHistory analyzes a single turn, building the game rules from history
// This is a convenience wrapper around AnalyzeSingleTurn for shell commands
func (a *Analyzer) AnalyzeSingleTurnFromHistory(ctx context.Context, history *pb.GameHistory, turnNum int) (*TurnAnalysis, error) {
	// Build the game rules from the history
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

	// Set challenge rule to DOUBLE
	if history.ChallengeRule == pb.ChallengeRule_VOID {
		history.ChallengeRule = pb.ChallengeRule_DOUBLE
	}

	return a.AnalyzeSingleTurn(ctx, history, rules, turnNum)
}

// AnalyzeSingleTurn analyzes a single turn in the game
// This can be used standalone or called in a loop by AnalyzeGame
func (a *Analyzer) AnalyzeSingleTurn(ctx context.Context, history *pb.GameHistory, rules *game.GameRules, turnNum int) (*TurnAnalysis, error) {
	if turnNum < 0 || turnNum >= len(history.Events) {
		return nil, fmt.Errorf("turn %d out of range", turnNum)
	}

	evt := history.Events[turnNum]
	if !a.isAnalyzableEvent(evt) {
		return nil, fmt.Errorf("turn %d is not analyzable (type: %s)", turnNum, evt.Type)
	}

	playerIndex := int(evt.PlayerIndex)

	// Create a game at this turn
	g, err := game.NewFromHistory(history, rules, turnNum)
	if err != nil {
		return nil, fmt.Errorf("failed to create game at turn %d: %w", turnNum, err)
	}

	// Skip if game not in playing state
	if g.Playing() != pb.PlayState_PLAYING {
		return nil, fmt.Errorf("game not in playing state at turn %d", turnNum)
	}

	// Get the move that was played
	playedMove, err := game.MoveFromEvent(evt, g.Alphabet(), g.Board())
	if err != nil {
		return nil, fmt.Errorf("failed to create move from event at turn %d: %w", turnNum, err)
	}

	// Check if previous opponent move was a phony that wasn't challenged
	missedChallenge := false
	if turnNum > 0 {
		prevEvent := history.Events[turnNum-1]
		if int(prevEvent.PlayerIndex) != playerIndex {
			if a.isPhony(prevEvent, g) {
				missedChallenge = true
			}
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

	// Check if we have a known opponent rack from a previous challenged phony
	var knownOppRack []tilemapping.MachineLetter
	if a.analysisCfg.UseExposedOppRacks && turnNum > 0 {
		// Look back to see if the previous event was a PHONY_TILES_RETURNED
		prevEvent := history.Events[turnNum-1]
		if prevEvent.Type == pb.GameEvent_PHONY_TILES_RETURNED {
			// The event before that should be the phony play by the opponent
			if turnNum >= 2 {
				phonyPlayEvent := history.Events[turnNum-2]
				if int(phonyPlayEvent.PlayerIndex) != playerIndex {
					// Opponent's phony was challenged off, we know their rack
					knownOppRack = extractExposedTiles(prevEvent.PlayedTiles, g.Alphabet())
					log.Info().
						Str("exposedTiles", prevEvent.PlayedTiles).
						Msg("extracted known opponent rack from challenged phony")
				}
			}
		}
	}

	// Analyze the position (phony flags set before so analyzeWithEndgame can use them)
	// They will be populated on the analysis struct inside analyzeTurn before dispatching.
	analysis, err := a.analyzeTurn(ctx, g, playedMove, isPhony, phonyChallenged, missedChallenge, turnNum, playerIndex, history.Players, knownOppRack)
	if err != nil {
		return nil, err
	}

	// Categorize the mistake (authoritative call — runs after all flags are set)
	analysis.MistakeCategory = categorizeMistake(analysis)

	return analysis, nil
}

// AnalyzeGame analyzes every move in a game and returns the results
func (a *Analyzer) AnalyzeGame(ctx context.Context, history *pb.GameHistory) (*GameAnalysisResult, error) {
	if history == nil {
		return nil, errors.New("game history is nil")
	}

	result := &GameAnalysisResult{
		Turns:           make([]*TurnAnalysis, 0),
		AnalysisVersion: CurrentAnalysisVersion,
		AnalyzerVersion: a.version,
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
			continue
		}

		playerIndex := int(evt.PlayerIndex)

		// Check if we should analyze this player
		if !shouldAnalyzePlayer(playerIndex) {
			continue
		}

		// Analyze this turn using the shared helper function
		analysis, err := a.AnalyzeSingleTurn(ctx, history, rules, turnNum)
		if err != nil {
			log.Warn().Err(err).Int("turn", turnNum).Msg("failed to analyze turn")
			continue
		}

		result.Turns = append(result.Turns, analysis)

		// Log progress
		analyzedCount++
		log.Info().
			Int("player-turn", analyzedCount).
			Int("total", totalTurns).
			Str("player", analysis.PlayerName).
			Str("move", analysis.PlayedMove.ShortDescription()).
			Msg("analyzed turn")

		// Update player summary
		summary := result.PlayerSummaries[playerIndex]
		summary.TurnsPlayed++
		if analysis.WasOptimal {
			summary.OptimalMoves++
		}
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

// extractExposedTiles converts PlayedTiles from PHONY_TILES_RETURNED event
// to MachineLetters, filtering out play-through tiles (0) and converting
// designated blanks to undesignated blanks (since blanks return to rack undesignated).
func extractExposedTiles(playedTiles string, alph *tilemapping.TileMapping) []tilemapping.MachineLetter {
	tiles, err := tilemapping.ToMachineLetters(playedTiles, alph)
	if err != nil {
		log.Err(err).Str("playedTiles", playedTiles).Msg("unable-to-convert-exposed-tiles")
		return nil
	}
	// Filter out play-through tiles and convert blanks to undesignated (rack format)
	var result []tilemapping.MachineLetter
	for _, t := range tiles {
		if t != 0 {
			// Use IntrinsicTileIdx to convert designated blanks (blank-A) to undesignated blank (0)
			result = append(result, t.IntrinsicTileIdx())
		}
	}
	return result
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
	isPhony, phonyChallenged, missedChallenge bool,
	turnNum, playerIndex int, players []*pb.PlayerInfo,
	knownOppRack []tilemapping.MachineLetter) (*TurnAnalysis, error) {

	tilesInBag := g.Bag().TilesRemaining()
	phase := a.determinePhase(tilesInBag)

	analysis := &TurnAnalysis{
		TurnNumber:      turnNum + 1, // 1-indexed for display
		PlayerIndex:     playerIndex,
		PlayerName:      players[playerIndex].Nickname,
		Rack:            g.RackFor(playerIndex).String(),
		Phase:           phase,
		TilesInBag:      tilesInBag,
		PlayedMove:      playedMove,
		IsPhony:         isPhony,
		PhonyChallenged: phonyChallenged,
		MissedChallenge: missedChallenge,
	}

	var err error
	switch phase {
	case PhaseEarlyMid:
		err = a.analyzeWithSim(ctx, g, analysis, a.analysisCfg.SimPlaysEarlyMid,
			a.analysisCfg.SimPliesEarlyMid, a.analysisCfg.SimStopEarlyMid, knownOppRack)
	case PhaseEarlyPreEndgame:
		err = a.analyzeWithSim(ctx, g, analysis, a.analysisCfg.SimPlaysEarlyPreEndgame,
			a.analysisCfg.SimPliesEarlyPreEndgame, a.analysisCfg.SimStopEarlyPreEndgame, knownOppRack)
	case PhasePreEndgame:
		err = a.analyzeWithPEG(ctx, g, analysis, knownOppRack)
	case PhaseEndgame:
		err = a.analyzeWithEndgame(ctx, g, analysis)
	}

	if err != nil {
		return nil, err
	}

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

		// Use standard thresholds for endgame spread.
		if loss <= spreadSmall {
			return "Small"
		} else if loss <= spreadMedium {
			return "Medium"
		} else {
			return "Large"
		}
	}

	// Handle spread/equity tiebreak mistakes for PEG and sim phases.
	// SpreadLoss is set when win probabilities are effectively tied — use
	// spread-based thresholds rather than win% thresholds.
	if analysis.SpreadLoss > 0 {
		loss := float64(analysis.SpreadLoss)

		// In early/mid blowout positions (win% near 0 or 100), the game result
		// is already decided so spread mistakes matter less — use doubled thresholds.
		if analysis.Phase == PhaseEarlyMid &&
			(analysis.OptimalWinProb < blowoutWinProbThreshold ||
				analysis.OptimalWinProb > 1-blowoutWinProbThreshold) {
			if loss <= spreadSmallBlowout {
				return "Small"
			} else if loss <= spreadMediumBlowout {
				return "Medium"
			} else {
				return "Large"
			}
		}

		if loss <= spreadSmall {
			return "Small"
		} else if loss <= spreadMedium {
			return "Medium"
		} else {
			return "Large"
		}
	}

	// For sim/PEG win probability, use win probability loss as percentage
	loss := analysis.WinProbLoss * 100

	// Losses <= 0.25% are noise-dominated and don't count as a mistake.
	const epsilon = 0.25
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
func estimateELO(mistakeIndex float64) float64 {
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
		return 2300.0
	}
	if mistakeIndex > 4.2 {
		// Use formula: 1650 - 125(x - 4.2)
		elo := 1650.0 - 125.0*(mistakeIndex-4.2)
		return elo
	}

	// Find bracketing points and interpolate
	for i := 1; i < len(dataPoints); i++ {
		if mistakeIndex <= dataPoints[i].mi {
			lower := dataPoints[i-1]
			upper := dataPoints[i]
			// Linear interpolation
			t := (mistakeIndex - lower.mi) / (upper.mi - lower.mi)
			elo := float64(lower.elo) + t*(float64(upper.elo)-float64(lower.elo))
			return elo
		}
	}
	return 1650.0
}

// isBingo returns true if the move is a bingo (plays 7 tiles)
func isBingo(m *move.Move) bool {
	if m == nil {
		return false
	}
	return m.BingoPlayed()
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
	numPlays, plies, stopCondition int, knownOppRack []tilemapping.MachineLetter) error {

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
	simmer.TryLoadWMP(a.cfg.WGLConfig(), g.LexiconName())

	if a.analysisCfg.Threads > 0 {
		simmer.SetThreads(a.analysisCfg.Threads)
	}

	// Prepare simulation
	err = simmer.PrepareSim(plies, plays)
	if err != nil {
		return fmt.Errorf("failed to prepare simulation: %w", err)
	}

	// Set known opponent rack if available (MUST be after PrepareSim which clears it)
	if len(knownOppRack) > 0 {
		simmer.SetKnownOppRack(knownOppRack)
		analysis.KnownOppRack = tilemapping.MachineWord(knownOppRack).UserVisible(g.Alphabet())
		log.Info().
			Str("knownOppRack", analysis.KnownOppRack).
			Msg("analyzing with known opponent rack from challenged phony")
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

	// On an empty board, the move generator only produces horizontal plays;
	// vertical first plays are transpositions. We must check transpositions
	// when comparing moves so a vertical opener is matched to its horizontal
	// equivalent in the sim results.
	checkTrans := g.Board().IsEmpty()

	// Find optimal move (highest win prob)
	analysis.OptimalMove = simmedPlays[0].Move()
	analysis.OptimalWinProb = simmedPlays[0].WinProb()

	// Find played move in results
	playedFound := false
	var playedEquity float64
	for _, result := range simmedPlays {
		if result.Move().Equals(analysis.PlayedMove, checkTrans, true) {
			analysis.PlayedWinProb = result.WinProb()
			playedEquity = result.EquityMean()
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
	analysis.WasOptimal = analysis.OptimalMove.Equals(analysis.PlayedMove, checkTrans, true)

	// When all plays are near 0% or 100% win probability, win prob loss is
	// meaningless. Use equity difference as a spread-based tiebreaker instead,
	// mirroring how PEG handles tied win probabilities.
	if !analysis.WasOptimal && playedFound {
		nearZero := analysis.OptimalWinProb < blowoutWinProbThreshold
		nearOne := analysis.OptimalWinProb > 1-blowoutWinProbThreshold
		if nearZero || nearOne {
			optimalEquity := simmedPlays[0].EquityMean()
			equityDiff := optimalEquity - playedEquity
			if equityDiff > 0.5 {
				analysis.SpreadLoss = int16(equityDiff + 0.5)
			}
		}
	}

	// Set bingo flags
	analysis.OptimalIsBingo = isBingo(analysis.OptimalMove)
	analysis.PlayedIsBingo = isBingo(analysis.PlayedMove)
	analysis.MissedBingo = analysis.OptimalIsBingo && !analysis.PlayedIsBingo

	// Extract enriched data: top 5 plays + played move (ignored plays included with flag)
	analysis.TopSimPlays = extractTopSimPlays(simmedPlays, analysis.PlayedMove, checkTrans, 5)

	return nil
}

// extractTopSimPlays returns at most maxTop plays plus ensuring the played move is included.
// Ignored plays are included with IsIgnored=true so the frontend can mark them as pruned.
func extractTopSimPlays(simmedPlays []*montecarlo.SimmedPlay, playedMove *move.Move, checkTrans bool, maxTop int) []*SimPlayResult {
	results := make([]*SimPlayResult, 0, maxTop+1)
	playedIncluded := false
	for _, sp := range simmedPlays {
		isPlayed := sp.Move().Equals(playedMove, checkTrans, true)
		if len(results) < maxTop || (isPlayed && !playedIncluded) {
			results = append(results, simPlayToResult(sp, isPlayed))
		}
		if isPlayed {
			playedIncluded = true
		}
		if len(results) >= maxTop && playedIncluded {
			break
		}
	}
	return results
}

func simPlayToResult(sp *montecarlo.SimmedPlay, isPlayed bool) *SimPlayResult {
	scoreStats := sp.ScoreStatsNoLock()
	bingoStats := sp.BingoStatsNoLock()
	plyCount := len(scoreStats)
	plyResults := make([]*PlyStatResult, plyCount)
	for i := 0; i < plyCount; i++ {
		bingoPct := 0.0
		if i < len(bingoStats) {
			bingoPct = bingoStats[i].Mean()
		}
		plyResults[i] = &PlyStatResult{
			Ply:        i + 1,
			ScoreMean:  scoreStats[i].Mean(),
			ScoreStdev: scoreStats[i].Stdev(),
			BingoPct:   bingoPct,
		}
	}
	return &SimPlayResult{
		MoveDescription: strings.TrimSpace(sp.Move().ShortDescription()),
		Score:           sp.Move().Score(),
		Leave:           sp.Move().LeaveString(),
		IsBingo:         sp.Move().BingoPlayed(),
		WinProb:         sp.WinProb(),
		WinProbStdErr:   sp.WinProbStdErr(),
		Equity:          sp.EquityMean(),
		EquityStdErr:    sp.EquityStdErr(),
		Iterations:      sp.WinProbIterations(),
		PlyStats:        plyResults,
		IsPlayedMove:    isPlayed,
		IsIgnored:       sp.IsIgnored(),
	}
}

// analyzeWithPEG analyzes a turn using the PEG solver (1 tile in bag)
func (a *Analyzer) analyzeWithPEG(ctx context.Context, g *game.Game, analysis *TurnAnalysis,
	knownOppRack []tilemapping.MachineLetter) error {
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

	// Set known opponent rack if available
	if len(knownOppRack) > 0 {
		pegSolver.SetKnownOppRack(knownOppRack)
		analysis.KnownOppRack = tilemapping.MachineWord(knownOppRack).UserVisible(g.Alphabet())
		log.Info().
			Str("knownOppRack", analysis.KnownOppRack).
			Msg("analyzing with known opponent rack from challenged phony")
	}

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

	// Set bingo flags
	analysis.OptimalIsBingo = isBingo(analysis.OptimalMove)
	analysis.PlayedIsBingo = isBingo(analysis.PlayedMove)
	analysis.MissedBingo = analysis.OptimalIsBingo && !analysis.PlayedIsBingo

	// Extract enriched PEG data: top 5 plays + played move
	analysis.TopPEGPlays = extractTopPEGPlays(plays, analysis.PlayedMove, noutcomes, g.Alphabet(), 5)

	return nil
}

// extractTopPEGPlays returns at most maxTop plays plus ensuring the played move is included.
// Ignored plays are included with IsIgnored=true so the frontend can mark them as pruned.
func extractTopPEGPlays(plays []*preendgame.PreEndgamePlay, playedMove *move.Move, noutcomes float32, alph *tilemapping.TileMapping, maxTop int) []*PEGPlayResult {
	results := make([]*PEGPlayResult, 0, maxTop+1)
	playedIncluded := false
	for _, play := range plays {
		isPlayed := play.Play.Equals(playedMove, false, true)
		if len(results) < maxTop || (isPlayed && !playedIncluded) {
			results = append(results, pegPlayToResult(play, noutcomes, alph, isPlayed))
		}
		if isPlayed {
			playedIncluded = true
		}
		if len(results) >= maxTop && playedIncluded {
			break
		}
	}
	return results
}

func pegPlayToResult(play *preendgame.PreEndgamePlay, noutcomes float32, alph *tilemapping.TileMapping, isPlayed bool) *PEGPlayResult {
	winProb := 0.0
	if noutcomes > 0 {
		winProb = float64(play.Points / noutcomes)
	}
	outcomes := play.OutcomesArray()
	outcomeResults := make([]*PEGOutcomeResult, len(outcomes))
	for i, o := range outcomes {
		tilesStr := tilemapping.MachineWord(o.Tiles()).UserVisible(alph)
		var outcomeStr string
		switch o.OutcomeResult() {
		case preendgame.PEGWin:
			outcomeStr = "win"
		case preendgame.PEGDraw:
			outcomeStr = "draw"
		case preendgame.PEGLoss:
			outcomeStr = "loss"
		default:
			outcomeStr = "unknown"
		}
		outcomeResults[i] = &PEGOutcomeResult{
			Tiles:   tilesStr,
			Outcome: outcomeStr,
			Count:   o.Count(),
		}
	}
	avgSpread := 0.0
	if play.HasSpread() && play.TotalOutcomes() > 0 {
		avgSpread = float64(play.GetSpread()) / float64(play.TotalOutcomes())
	}
	return &PEGPlayResult{
		MoveDescription: strings.TrimSpace(play.Play.ShortDescription()),
		Score:           play.Play.Score(),
		Leave:           play.Play.LeaveString(),
		IsBingo:         play.Play.BingoPlayed(),
		WinProb:         winProb,
		Outcomes:        outcomeResults,
		HasSpread:       play.HasSpread(),
		AvgSpread:       avgSpread,
		IsPlayedMove:    isPlayed,
		IsIgnored:       play.Ignore,
	}
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

	// Set bingo flags
	analysis.OptimalIsBingo = isBingo(analysis.OptimalMove)
	analysis.PlayedIsBingo = isBingo(analysis.PlayedMove)
	analysis.MissedBingo = analysis.OptimalIsBingo && !analysis.PlayedIsBingo

	// Extract enriched endgame data: principal variation and other variations
	analysis.PrincipalVariation = pvLineToResult(principalVariation, bestSpread)
	variations := endgameSolver.Variations()
	otherVars := make([]*EndgameVariationResult, 0, len(variations))
	for _, v := range variations {
		if v.NumMoves() == 0 {
			continue
		}
		moves := make([]*move.Move, v.NumMoves())
		for i := 0; i < v.NumMoves(); i++ {
			moves[i] = v.Moves[i]
		}
		otherVars = append(otherVars, pvLineToResult(moves, v.Score()))
	}
	analysis.OtherVariations = otherVars

	return nil
}

// pvLineToResult converts a slice of moves and final spread into an EndgameVariationResult.
func pvLineToResult(moves []*move.Move, finalSpread int16) *EndgameVariationResult {
	moveResults := make([]*EndgameMoveResult, 0, len(moves))
	for i, m := range moves {
		if m == nil {
			break
		}
		moveResults = append(moveResults, &EndgameMoveResult{
			MoveDescription: strings.TrimSpace(m.ShortDescription()),
			Score:           m.Score(),
			MoveNumber:      i + 1,
		})
	}
	return &EndgameVariationResult{
		Moves:       moveResults,
		FinalSpread: finalSpread,
	}
}

// calculatePlayerSummaries calculates aggregate statistics for each player
func (a *Analyzer) calculatePlayerSummaries(result *GameAnalysisResult) {
	for i := 0; i < 2; i++ {
		summary := result.PlayerSummaries[i]
		if summary.TurnsPlayed == 0 {
			continue
		}

		var totalWinProbLoss float64
		winProbCount := 0

		for _, turn := range result.Turns {
			if turn.PlayerIndex != i {
				continue
			}

			// Count win prob loss for non-endgame phases
			if turn.Phase != PhaseEndgame {
				totalWinProbLoss += turn.WinProbLoss
				winProbCount++
			}

			// Add mistake points to the mistake index
			summary.MistakeIndex += mistakePoints(turn.MistakeCategory)

			// Count mistakes by size
			switch turn.MistakeCategory {
			case "Small":
				summary.SmallMistakes++
			case "Medium":
				summary.MediumMistakes++
			case "Large":
				summary.LargeMistakes++
			}

			// Count available and missed bingos
			if turn.OptimalIsBingo {
				summary.AvailableBingos++
				if turn.MissedBingo {
					summary.MissedBingos++
				}
			}
		}

		if winProbCount > 0 {
			summary.AvgWinProbLoss = totalWinProbLoss / float64(winProbCount)
		}

		// Calculate estimated ELO based on mistake index
		summary.EstimatedELO = estimateELO(summary.MistakeIndex)
	}
}
