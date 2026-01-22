package shell

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/domino14/macondo/gameanalysis"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// GameSource represents a source for loading a game
type GameSource struct {
	Type       string // "woogles", "xt", "file", "web"
	Identifier string // game ID or file path
	Original   string // original source string (for display)
}

// parseGameSource parses a game source string (e.g., "woog:ABC123", "xt:12345", "/path/to/game.gcg", "woogcollection:UUID")
func parseGameSource(source string) (*GameSource, error) {
	if strings.HasPrefix(source, "woogcollection:") {
		return &GameSource{
			Type:       "collection",
			Identifier: strings.TrimPrefix(source, "woogcollection:"),
			Original:   source,
		}, nil
	} else if strings.HasPrefix(source, "woog:") {
		return &GameSource{
			Type:       "woogles",
			Identifier: strings.TrimPrefix(source, "woog:"),
			Original:   source,
		}, nil
	} else if strings.HasPrefix(source, "xt:") {
		return &GameSource{
			Type:       "xt",
			Identifier: strings.TrimPrefix(source, "xt:"),
			Original:   source,
		}, nil
	} else if strings.HasPrefix(source, "http://") || strings.HasPrefix(source, "https://") {
		return &GameSource{
			Type:       "web",
			Identifier: source,
			Original:   source,
		}, nil
	} else {
		// Assume it's a file path
		return &GameSource{
			Type:       "file",
			Identifier: source,
			Original:   source,
		}, nil
	}
}

// WooglesCollectionGame represents a game in a collection
type WooglesCollectionGame struct {
	GameID        string `json:"game_id"`
	ChapterNumber uint32 `json:"chapter_number"`
	ChapterTitle  string `json:"chapter_title"`
}

// WooglesCollection represents a collection from the Woogles API
type WooglesCollection struct {
	UUID        string                   `json:"uuid"`
	Title       string                   `json:"title"`
	Description string                   `json:"description"`
	Games       []WooglesCollectionGame  `json:"games"`
}

// WooglesCollectionResponse represents the API response
type WooglesCollectionResponse struct {
	Collection WooglesCollection `json:"collection"`
}

// fetchWooglesCollection fetches a collection from Woogles and returns the game IDs
func (sc *ShellController) fetchWooglesCollection(collectionID string) ([]string, error) {
	path := "https://woogles.io/api/collections_service.CollectionsService/GetCollection"
	payload := fmt.Sprintf(`{"collection_uuid": "%s"}`, collectionID)
	reader := strings.NewReader(payload)

	resp, err := http.Post(path, "application/json", reader)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("bad status code: %s, body: %s", resp.Status, string(body))
	}

	var response WooglesCollectionResponse
	err = json.Unmarshal(body, &response)
	if err != nil {
		return nil, fmt.Errorf("failed to parse collection response: %w, body: %s", err, string(body))
	}

	gameIDs := make([]string, 0, len(response.Collection.Games))
	for _, game := range response.Collection.Games {
		gameIDs = append(gameIDs, game.GameID)
	}

	return gameIDs, nil
}

// loadGameHistoryFromSource loads a game history from a GameSource
func (sc *ShellController) loadGameHistoryFromSource(source *GameSource) (*pb.GameHistory, error) {
	switch source.Type {
	case "woogles":
		return sc.loadGameHistoryFromWoogles(source.Identifier)
	case "xt":
		return sc.loadGameHistoryFromCrossTables(source.Identifier)
	case "web":
		return sc.loadGameHistoryFromWeb(source.Identifier)
	case "file":
		return sc.loadGameHistoryFromFile(source.Identifier)
	case "collection":
		return nil, fmt.Errorf("collection sources should be expanded before loading")
	default:
		return nil, fmt.Errorf("unknown source type: %s", source.Type)
	}
}

// analyzeBatch analyzes multiple games from various sources
func (sc *ShellController) analyzeBatch(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) == 0 {
		return nil, errors.New("need to provide at least one game source")
	}

	// Parse options
	cfg := gameanalysis.DefaultAnalysisConfig()
	continueOnError := cmd.options.Bool("continue")
	summaryOnly := cmd.options.Bool("summary-only")

	// Check for player filter option
	if playerOpt := cmd.options.String("player"); playerOpt != "" {
		if playerOpt == "0" {
			cfg.OnlyPlayer = 0
		} else if playerOpt == "1" {
			cfg.OnlyPlayer = 1
		} else {
			cfg.OnlyPlayerByName = playerOpt
			cfg.OnlyPlayer = -1
		}
	}

	// Parse game sources
	sources := make([]*GameSource, 0, len(cmd.args))
	for _, arg := range cmd.args {
		source, err := parseGameSource(arg)
		if err != nil {
			return nil, fmt.Errorf("failed to parse source %s: %w", arg, err)
		}

		// If it's a collection, expand it into individual game sources
		if source.Type == "collection" {
			sc.showMessage(fmt.Sprintf("Fetching collection: %s", source.Identifier))
			gameIDs, err := sc.fetchWooglesCollection(source.Identifier)
			if err != nil {
				if !continueOnError {
					return nil, fmt.Errorf("failed to fetch collection %s: %w", source.Identifier, err)
				}
				sc.showMessage(fmt.Sprintf("  Error fetching collection: %v", err))
				continue
			}
			sc.showMessage(fmt.Sprintf("  Found %d games in collection", len(gameIDs)))

			// Add each game from the collection as a woogles source
			for _, gameID := range gameIDs {
				sources = append(sources, &GameSource{
					Type:       "woogles",
					Identifier: gameID,
					Original:   fmt.Sprintf("woog:%s (from woogcollection:%s)", gameID, source.Identifier),
				})
			}
		} else {
			sources = append(sources, source)
		}
	}

	// Create batch result
	batchResult := gameanalysis.NewBatchAnalysisResult()

	// Create analyzer
	analyzer := gameanalysis.New(sc.config, cfg)
	ctx := context.Background()

	// Analyze each game
	for i, source := range sources {
		sc.showMessage(fmt.Sprintf("Analyzing game %d/%d: %s", i+1, len(sources), source.Original))

		gameResult := &gameanalysis.BatchGameResult{
			GameID: source.Original,
		}

		// Load game history
		history, err := sc.loadGameHistoryFromSource(source)
		if err != nil {
			gameResult.LoadError = err
			batchResult.AddGameResult(gameResult)

			if !continueOnError {
				return nil, fmt.Errorf("failed to load game %s: %w", source.Original, err)
			}
			sc.showMessage(fmt.Sprintf("  Error loading: %v", err))
			continue
		}

		// Set game info
		if len(history.Players) >= 2 {
			gameResult.GameInfo = fmt.Sprintf("%s vs %s",
				history.Players[0].Nickname,
				history.Players[1].Nickname)
		}

		// Analyze game
		result, err := analyzer.AnalyzeGame(ctx, history)
		if err != nil {
			gameResult.AnalysisErr = err
			batchResult.AddGameResult(gameResult)

			if !continueOnError {
				return nil, fmt.Errorf("failed to analyze game %s: %w", source.Original, err)
			}
			sc.showMessage(fmt.Sprintf("  Error analyzing: %v", err))
			continue
		}

		gameResult.Result = result
		batchResult.AddGameResult(gameResult)
	}

	// Calculate averages
	batchResult.CalculateAverages()

	// Format output
	output := sc.formatBatchResults(batchResult, summaryOnly)
	return msg(output), nil
}

// formatBatchResults formats the batch analysis results for display
func (sc *ShellController) formatBatchResults(batch *gameanalysis.BatchAnalysisResult, summaryOnly bool) string {
	var sb strings.Builder

	// Show individual game results unless summary-only
	if !summaryOnly {
		for _, gameResult := range batch.Games {
			sb.WriteString(fmt.Sprintf("[%s] %s\n", gameResult.GameID, gameResult.GameInfo))
			sb.WriteString(strings.Repeat("=", 80))
			sb.WriteString("\n")

			if gameResult.LoadError != nil {
				sb.WriteString(fmt.Sprintf("Error loading game: %v\n\n", gameResult.LoadError))
				continue
			}

			if gameResult.AnalysisErr != nil {
				sb.WriteString(fmt.Sprintf("Error analyzing game: %v\n\n", gameResult.AnalysisErr))
				continue
			}

			// Format individual game analysis (reuse existing formatting)
			sb.WriteString(sc.formatGameAnalysisForBatch(gameResult.Result))
			sb.WriteString("\n")
		}
	}

	// Summary section
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n")
	sb.WriteString("BATCH SUMMARY\n")
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n\n")

	sb.WriteString(fmt.Sprintf("Total games: %d\n", batch.TotalGames))
	sb.WriteString(fmt.Sprintf("Successful: %d\n", batch.SuccessfulGames))
	sb.WriteString(fmt.Sprintf("Failed: %d\n\n", batch.FailedGames))

	// Per-game results
	if len(batch.Games) > 0 {
		sb.WriteString("Per-Game Results:\n")
		sb.WriteString(fmt.Sprintf("%-25s  %-15s  %-6s  %-6s  %-6s  %-6s  %-6s\n",
			"Game", "Player", "MI", "Turns", "Small", "Med", "Large"))
		sb.WriteString(strings.Repeat("-", 80))
		sb.WriteString("\n")

		for _, gameResult := range batch.Games {
			if gameResult.Result != nil {
				for _, summary := range gameResult.Result.PlayerSummaries {
					if summary == nil || summary.TurnsPlayed == 0 {
						continue
					}

					// Count mistake categories from turns
					smallCount, mediumCount, largeCount := 0, 0, 0
					for _, turn := range gameResult.Result.Turns {
						if turn.PlayerName == summary.PlayerName {
							switch turn.MistakeCategory {
							case "Small":
								smallCount++
							case "Medium":
								mediumCount++
							case "Large":
								largeCount++
							}
						}
					}

					gameID := gameResult.GameID
					if len(gameID) > 25 {
						gameID = gameID[:22] + "..."
					}

					playerName := summary.PlayerName
					if len(playerName) > 15 {
						playerName = playerName[:12] + "..."
					}

					sb.WriteString(fmt.Sprintf("%-25s  %-15s  %-6.1f  %-6d  %-6d  %-6d  %-6d\n",
						gameID,
						playerName,
						summary.MistakeIndex,
						summary.TurnsPlayed,
						smallCount,
						mediumCount,
						largeCount))
				}
			}
		}
		sb.WriteString("\n")
	}

	// Aggregate by player
	if len(batch.PlayerStats) > 0 {
		sb.WriteString("Aggregate by Player:\n")
		sb.WriteString(fmt.Sprintf("%-15s  %-6s  %-6s  %-8s  %-6s  %-6s  %-6s  %-8s  %-8s\n",
			"Player", "Games", "Turns", "Optimal", "Small", "Medium", "Large", "Avg MI", "Est ELO"))
		sb.WriteString(strings.Repeat("-", 90))
		sb.WriteString("\n")

		// Sort players by name for consistent output
		playerNames := make([]string, 0, len(batch.PlayerStats))
		for name := range batch.PlayerStats {
			playerNames = append(playerNames, name)
		}
		sort.Strings(playerNames)

		for _, name := range playerNames {
			stats := batch.PlayerStats[name]

			// Skip players that weren't analyzed (0 turns)
			if stats.TotalTurns == 0 {
				continue
			}

			playerName := stats.PlayerName
			if len(playerName) > 15 {
				playerName = playerName[:12] + "..."
			}

			sb.WriteString(fmt.Sprintf("%-15s  %-6d  %-6d  %-8d  %-6d  %-6d  %-6d  %-8.1f  %-8s\n",
				playerName,
				stats.GamesPlayed,
				stats.TotalTurns,
				stats.TotalOptimal,
				stats.TotalSmall,
				stats.TotalMedium,
				stats.TotalLarge,
				stats.AvgMistakeIndex,
				stats.AvgEstimatedELO))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// formatGameAnalysisForBatch formats a single game analysis for batch display (simplified)
func (sc *ShellController) formatGameAnalysisForBatch(result *gameanalysis.GameAnalysisResult) string {
	var sb strings.Builder

	// Simplified turn-by-turn table
	sb.WriteString(fmt.Sprintf("%-4s  %-12s  %-8s  %-18s  %-18s  %-6s  %-8s  %-8s\n",
		"Turn", "Player", "Rack", "Played", "Optimal", "Diff", "Phase", "Mistake"))
	sb.WriteString(strings.Repeat("-", 100))
	sb.WriteString("\n")

	for _, turn := range result.Turns {
		player := turn.PlayerName
		if len(player) > 12 {
			player = player[:12]
		}

		rack := turn.Rack
		if len(rack) > 8 {
			rack = rack[:8]
		}

		played := turn.PlayedMove.ShortDescription()
		if len(played) > 18 {
			played = played[:18]
		}

		optimal := turn.OptimalMove.ShortDescription()
		if len(optimal) > 18 {
			optimal = optimal[:18]
		}

		// Format difference
		var diff string
		if turn.Phase == gameanalysis.PhaseEndgame {
			if turn.WasOptimal {
				diff = "+0"
			} else {
				diff = fmt.Sprintf("%+d", turn.SpreadLoss)
			}
		} else if turn.Phase == gameanalysis.PhasePreEndgame && turn.SpreadLoss > 0 {
			diff = fmt.Sprintf("%.1f%% (%+d)", turn.WinProbLoss*100, turn.SpreadLoss)
		} else {
			if turn.WasOptimal {
				diff = "0.0%"
			} else {
				diff = fmt.Sprintf("%.1f%%", turn.WinProbLoss*100)
			}
		}

		mistake := turn.MistakeCategory
		if mistake == "" {
			mistake = "-"
		}

		phaseDisplay := turn.Phase.String()

		sb.WriteString(fmt.Sprintf("%-4d  %-12s  %-8s  %-18s  %-18s  %-6s  %-8s  %-8s\n",
			turn.TurnNumber,
			player,
			rack,
			played,
			optimal,
			diff,
			phaseDisplay,
			mistake))
	}

	sb.WriteString("\n")

	// Player summaries
	sb.WriteString(fmt.Sprintf("%-15s  %-6s  %-8s  %-14s  %-13s\n",
		"Player", "Turns", "Optimal", "Avg Win% Loss", "Mistake Index"))
	sb.WriteString(strings.Repeat("-", 70))
	sb.WriteString("\n")

	for i := 0; i < 2; i++ {
		summary := result.PlayerSummaries[i]
		if summary.TurnsPlayed == 0 {
			continue
		}

		sb.WriteString(fmt.Sprintf("%-15s  %-6d  %-8d  %-14s  %-13s\n",
			summary.PlayerName,
			summary.TurnsPlayed,
			summary.OptimalMoves,
			fmt.Sprintf("%.1f%%", summary.AvgWinProbLoss*100),
			fmt.Sprintf("%.1f", summary.MistakeIndex)))
	}

	sb.WriteString("\n")

	return sb.String()
}
