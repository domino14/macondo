package gameanalysis

// BatchGameResult represents the results for a single game in a batch analysis
type BatchGameResult struct {
	GameID      string              // e.g., "woogles:ABC123", "xt:12345", "/path/to/game.gcg"
	GameInfo    string              // e.g., "Player1 vs Player2"
	LoadError   error               // Error during game loading
	AnalysisErr error               // Error during game analysis
	Result      *GameAnalysisResult // Analysis result if successful
}

// BatchPlayerStats represents aggregate statistics for a player across multiple games
type BatchPlayerStats struct {
	PlayerName      string
	GamesPlayed     int
	TotalTurns      int
	TotalOptimal    int
	TotalSmall      int     // Small mistakes count
	TotalMedium     int     // Medium mistakes count
	TotalLarge      int     // Large mistakes count
	TotalMistakeIdx float64 // Sum of mistake indices
	AvgMistakeIndex float64 // Average mistake index
	AvgEstimatedELO string  // Average estimated ELO
}

// BatchAnalysisResult represents the aggregate results of analyzing multiple games
type BatchAnalysisResult struct {
	Games           []*BatchGameResult
	PlayerStats     map[string]*BatchPlayerStats
	TotalGames      int
	SuccessfulGames int
	FailedGames     int
}

// NewBatchAnalysisResult creates a new BatchAnalysisResult
func NewBatchAnalysisResult() *BatchAnalysisResult {
	return &BatchAnalysisResult{
		Games:       make([]*BatchGameResult, 0),
		PlayerStats: make(map[string]*BatchPlayerStats),
	}
}

// AddGameResult adds a game result to the batch and updates aggregate statistics
func (b *BatchAnalysisResult) AddGameResult(gameResult *BatchGameResult) {
	b.Games = append(b.Games, gameResult)
	b.TotalGames++

	if gameResult.LoadError != nil || gameResult.AnalysisErr != nil {
		b.FailedGames++
		return
	}

	b.SuccessfulGames++

	// Update player stats
	if gameResult.Result != nil {
		for _, playerSummary := range gameResult.Result.PlayerSummaries {
			if playerSummary == nil {
				continue
			}

			stats, exists := b.PlayerStats[playerSummary.PlayerName]
			if !exists {
				stats = &BatchPlayerStats{
					PlayerName: playerSummary.PlayerName,
				}
				b.PlayerStats[playerSummary.PlayerName] = stats
			}

			stats.GamesPlayed++
			stats.TotalTurns += playerSummary.TurnsPlayed
			stats.TotalOptimal += playerSummary.OptimalMoves
			stats.TotalMistakeIdx += playerSummary.MistakeIndex

			// Count mistake categories from turns
			for _, turn := range gameResult.Result.Turns {
				if turn.PlayerName == playerSummary.PlayerName {
					switch turn.MistakeCategory {
					case "Small":
						stats.TotalSmall++
					case "Medium":
						stats.TotalMedium++
					case "Large":
						stats.TotalLarge++
					}
				}
			}
		}
	}
}

// CalculateAverages calculates average statistics for all players
func (b *BatchAnalysisResult) CalculateAverages() {
	for _, stats := range b.PlayerStats {
		if stats.GamesPlayed > 0 {
			stats.AvgMistakeIndex = stats.TotalMistakeIdx / float64(stats.GamesPlayed)
			stats.AvgEstimatedELO = estimateELO(stats.AvgMistakeIndex)
		} else {
			stats.AvgEstimatedELO = "N/A"
		}
	}
}
