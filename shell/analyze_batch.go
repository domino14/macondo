package shell

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gameanalysis"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/turnplayer"
	wmppkg "github.com/domino14/macondo/wmp"
)

// GameSource represents a source for loading a game
type GameSource struct {
	Type       string // "woogles", "xt", "file", "dir", "web"
	Identifier string // game ID, file path, or directory path
	Original   string // original source string (for display)
}

// parseGameSource parses a game source string (e.g., "woog:ABC123", "xt:12345", "/path/to/game.gcg", "/path/to/games/", "woogcollection:UUID")
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
		// A path, either a single game or a folder of them. Stat follows
		// symlinks, so a link to a folder is a folder here. A path that can't
		// be stat'ed is left as a file so the loader reports what went wrong.
		path := expandHomePath(source)
		if info, err := os.Stat(path); err == nil && info.IsDir() {
			return &GameSource{
				Type:       "dir",
				Identifier: path,
				Original:   path,
			}, nil
		}
		return &GameSource{
			Type:       "file",
			Identifier: path,
			Original:   path,
		}, nil
	}
}

// gcgFilesInDir returns every .gcg file at or below dir, lexically within each
// directory. Unlike filepath.WalkDir it follows symlinks, since a folder of
// games is as likely to be a link as a real directory and silently finding
// nothing there is worse than the cost of resolving them; already-visited
// directories are remembered so a link back up the tree cannot loop forever.
func gcgFilesInDir(dir string) ([]string, error) {
	var paths []string
	seen := map[string]bool{}

	var walk func(dir string) error
	walk = func(dir string) error {
		// Loop protection keys on the resolved path, so the same directory
		// reached by two different links is only walked once.
		resolved, err := filepath.EvalSymlinks(dir)
		if err != nil {
			return err
		}
		if seen[resolved] {
			return nil
		}
		seen[resolved] = true

		entries, err := os.ReadDir(dir) // sorted by filename
		if err != nil {
			return err
		}
		for _, e := range entries {
			path := filepath.Join(dir, e.Name())
			isDir := e.IsDir()
			if e.Type()&fs.ModeSymlink != 0 {
				info, err := os.Stat(path) // follows the link
				if err != nil {
					continue // dangling link; nothing to analyze
				}
				isDir = info.IsDir()
			}
			if isDir {
				if err := walk(path); err != nil {
					return err
				}
			} else if strings.EqualFold(filepath.Ext(path), ".gcg") {
				paths = append(paths, path)
			}
		}
		return nil
	}

	if err := walk(dir); err != nil {
		return nil, err
	}
	return paths, nil
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

// reuseUndecided is the reason reusableAnalysis gives when it cannot decide
// without the game's player list.
const reuseUndecided = "undecided"

// reusableAnalysis returns the stored analysis if it can stand in for
// re-analyzing the game, or nil and the reason it cannot. players may be nil
// when the game history has not been loaded yet; a reuseUndecided reason means
// the caller should load it and ask again.
//
// The stored row is not proof on its own: it may predate the fields this
// version reports, or have been produced under a -player filter that answers
// a narrower question than the one being asked now.
func reusableAnalysis(stored *gameanalysis.StoredAnalysis, cfg *gameanalysis.AnalysisConfig,
	players []*pb.PlayerInfo) (*gameanalysis.GameAnalysisResult, string) {

	if stored.AnalysisVersion < gameanalysis.CurrentAnalysisVersion {
		return nil, fmt.Sprintf("stored analysis is version %d, current is %d",
			stored.AnalysisVersion, gameanalysis.CurrentAnalysisVersion)
	}
	resultProto := &pb.GameAnalysisResult{}
	if err := protojson.Unmarshal(stored.ResultJSON, resultProto); err != nil {
		return nil, fmt.Sprintf("stored analysis could not be read: %v", err)
	}
	result := gameanalysis.GameAnalysisResultFromProto(resultProto)

	switch result.Covers(cfg, players) {
	case gameanalysis.CoverageComplete:
		return result, ""
	case gameanalysis.CoverageUnknown:
		return nil, reuseUndecided
	default:
		return nil, "stored analysis does not cover the requested player"
	}
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
	batchName := cmd.options.String("batch")
	if batchName == "" {
		batchName = "batch-" + time.Now().Format("2006-01-02-1504")
		sc.showMessage(fmt.Sprintf("No batch name specified; using '%s'", batchName))
	}
	force := cmd.options.Bool("force")
	jsonFile := cmd.options.String("json")

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

	// Open store once for the whole batch
	store, storeErr := sc.getAnalysisStore()
	if storeErr != nil {
		sc.showMessage(fmt.Sprintf("Warning: cannot open analysis store: %v", storeErr))
	}

	// Parse game sources
	sources := make([]*GameSource, 0, len(cmd.args))
	for _, arg := range cmd.args {
		source, err := parseGameSource(arg)
		if err != nil {
			return nil, fmt.Errorf("failed to parse source %s: %w", arg, err)
		}

		// Collections and folders stand for many games; expand them.
		switch source.Type {
		case "collection":
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

		case "dir":
			sc.showMessage(fmt.Sprintf("Scanning folder for .gcg files: %s", source.Identifier))
			paths, err := gcgFilesInDir(source.Identifier)
			if err != nil {
				if !continueOnError {
					return nil, fmt.Errorf("failed to scan folder %s: %w", source.Identifier, err)
				}
				sc.showMessage(fmt.Sprintf("  Error scanning folder: %v", err))
				continue
			}
			if len(paths) == 0 {
				sc.showMessage("  No .gcg files found")
				continue
			}
			sc.showMessage(fmt.Sprintf("  Found %d .gcg files", len(paths)))

			for _, path := range paths {
				sources = append(sources, &GameSource{
					Type:       "file",
					Identifier: path,
					Original:   path,
				})
			}

		default:
			sources = append(sources, source)
		}
	}

	if len(sources) == 0 {
		return nil, errors.New("no games to analyze")
	}

	// Create batch result
	batchResult := gameanalysis.NewBatchAnalysisResult()

	// Create analyzer
	analyzer := gameanalysis.New(sc.config, cfg, sc.macondoVersion)
	ctx := context.Background()

	// Analyze each game
	// batchGameResults holds per-game protojson for the optional JSON export
	type batchExportEntry struct {
		Name   string          `json:"name"`
		Result json.RawMessage `json:"result"`
	}
	var exportEntries []batchExportEntry

	// Without -continue the first failure stops the batch, but whatever was
	// analyzed before it is still worth reporting, so the run unwinds through
	// abortErr rather than returning from inside the loop.
	var abortErr error

	// Nothing in the batch path goes through `load`, so no lexicon has been
	// fetched for these games. Do it once per distinct lexicon, remembering
	// failures so a folder of games in an unavailable lexicon does not retry
	// the download once per file.
	ensuredLexica := map[string]error{}
	ensureLexicon := func(lexicon string) error {
		if err, done := ensuredLexica[lexicon]; done {
			return err
		}
		err := turnplayer.EnsureKWG(lexicon, sc.config.WGLConfig())
		if err != nil {
			err = fmt.Errorf("could not ensure lexicon %s: %w", lexicon, err)
		} else if _, wmpErr := wmppkg.EnsureWMP(sc.config.WGLConfig(), lexicon); wmpErr != nil {
			log.Info().Err(wmpErr).Str("lexicon", lexicon).
				Msg("WMP not available for this lexicon; sim will use the KWG algorithm")
		}
		ensuredLexica[lexicon] = err
		return err
	}

	for i, source := range sources {
		sc.showMessage(fmt.Sprintf("Analyzing game %d/%d: %s", i+1, len(sources), source.Original))

		gameResult := &gameanalysis.BatchGameResult{
			GameID: source.Original,
		}

		// A stored analysis stands in for re-running the game, but only if it
		// answers what this run asks. Coverage can depend on the game's player
		// list, which is not known until the history is loaded, so the check
		// runs again below for the games that need it.
		var stored *gameanalysis.StoredAnalysis
		if store != nil && !force {
			if s, err := store.Get(source.Original); err == nil {
				stored = s
			}
		}
		reuse := func(players []*pb.PlayerInfo) bool {
			if stored == nil {
				return false
			}
			result, reason := reusableAnalysis(stored, cfg, players)
			if reason == reuseUndecided {
				return false // decide after the history is loaded
			}
			if result == nil {
				sc.showMessage(fmt.Sprintf("  Re-analyzing '%s': %s", source.Original, reason))
				stored = nil
				return false
			}
			sc.showMessage(fmt.Sprintf("  Skipping '%s' (already analyzed). Use -force to overwrite.", source.Original))
			gameResult.GameInfo = stored.PlayerInfo
			gameResult.Result = result
			batchResult.AddGameResult(gameResult)
			return true
		}
		if reuse(nil) {
			continue
		}

		// Load game history
		history, err := sc.loadGameHistoryFromSource(source)
		if err != nil {
			gameResult.LoadError = err
			batchResult.AddGameResult(gameResult)

			if !continueOnError {
				abortErr = fmt.Errorf("failed to load game %s: %w", source.Original, err)
				break
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

		// The players are known now, so a reuse decision that depended on
		// them can be made.
		if reuse(history.Players) {
			continue
		}

		// Make sure the lexicon these games were played in is on disk, the
		// way `load` does, then validate racks to catch corrupt games early.
		if history.Lexicon == "" {
			history.Lexicon = sc.config.GetString(config.ConfigDefaultLexicon)
		}
		err = ensureLexicon(history.Lexicon)
		if err == nil {
			boardLayout, ldName, variant := game.HistoryToVariant(history)
			var rules *game.GameRules
			rules, err = game.NewBasicGameRules(sc.config, history.Lexicon, boardLayout, ldName,
				game.CrossScoreAndSet, variant)
			if err == nil {
				if vErr := validateGameHistory(history, rules.LetterDistribution().TileMapping()); vErr != nil {
					err = fmt.Errorf("game history is corrupt: %w", vErr)
				}
			}
		}
		if err != nil {
			gameResult.AnalysisErr = err
			batchResult.AddGameResult(gameResult)

			if !continueOnError {
				abortErr = fmt.Errorf("failed to validate game %s: %w", source.Original, err)
				break
			}
			sc.showMessage(fmt.Sprintf("  Error validating: %v", err))
			continue
		}

		// Analyze game
		result, err := analyzer.AnalyzeGame(ctx, history)
		if err != nil {
			gameResult.AnalysisErr = err
			batchResult.AddGameResult(gameResult)

			if !continueOnError {
				abortErr = fmt.Errorf("failed to analyze game %s: %w (use -continue to skip bad games)",
					source.Original, err)
				break
			}
			sc.showMessage(fmt.Sprintf("  Error analyzing: %v", err))
			continue
		}

		gameResult.Result = result
		batchResult.AddGameResult(gameResult)

		// Save to DB
		if store != nil {
			resultJSON, merr := protojson.Marshal(result.ToProto())
			if merr == nil {
				if serr := store.Save(source.Original, batchName, gameResult.GameInfo,
					history.Lexicon, result.AnalysisVersion, result.AnalyzerVersion, resultJSON); serr != nil {
					sc.showMessage(fmt.Sprintf("  Warning: failed to save to store: %v", serr))
				} else if jsonFile != "" {
					exportEntries = append(exportEntries, batchExportEntry{
						Name:   source.Original,
						Result: json.RawMessage(resultJSON),
					})
				}
			}
		} else if jsonFile != "" {
			if resultJSON, merr := protojson.Marshal(result.ToProto()); merr == nil {
				exportEntries = append(exportEntries, batchExportEntry{
					Name:   source.Original,
					Result: json.RawMessage(resultJSON),
				})
			}
		}
	}

	// Calculate averages
	batchResult.CalculateAverages()

	// JSON export for batch
	if jsonFile != "" {
		exportData, merr := json.MarshalIndent(map[string]interface{}{"games": exportEntries}, "", "  ")
		if merr == nil {
			if werr := os.WriteFile(jsonFile, exportData, 0644); werr != nil {
				sc.showMessage(fmt.Sprintf("Warning: failed to write JSON to %s: %v", jsonFile, werr))
			} else {
				sc.showMessage(fmt.Sprintf("Batch analysis written to %s", jsonFile))
			}
		}
	}

	// Format output
	output := sc.formatBatchResults(batchResult, summaryOnly)
	if abortErr != nil {
		// The shell prints either the response or the error, so show the
		// results of the games that did finish before reporting the failure.
		sc.showMessage(output)
		return nil, abortErr
	}
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
	rowsWritten := 0
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

					sb.WriteString(fmt.Sprintf("%-25s  %-15s  %-6.2f  %-6d  %-6d  %-6d  %-6d\n",
						gameID,
						playerName,
						summary.MistakeIndex,
						summary.TurnsPlayed,
						smallCount,
						mediumCount,
						largeCount))
					rowsWritten++
				}
			}
		}
		if rowsWritten == 0 {
			sb.WriteString("(no turns were analyzed in any game)\n")
		}
		sb.WriteString("\n")
	}

	// Aggregate by player
	if len(batch.PlayerStats) > 0 {
		sb.WriteString("Aggregate by Player:\n")
		sb.WriteString(fmt.Sprintf("%-15s  %-6s  %-6s  %-8s  %-6s  %-6s  %-6s  %-8s  %-8s  %-12s\n",
			"Player", "Games", "Turns", "Optimal", "Small", "Medium", "Large", "Avg MI", "Est ELO", "Bingo Rate"))
		sb.WriteString(strings.Repeat("-", 105))
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

			// Calculate bingo rate
			bingoRate := "-"
			if stats.TotalAvailableBingos > 0 {
				bingosMade := stats.TotalAvailableBingos - stats.TotalMissedBingos
				bingoRate = fmt.Sprintf("%d/%d (%.0f%%)",
					bingosMade, stats.TotalAvailableBingos,
					100.0*float64(bingosMade)/float64(stats.TotalAvailableBingos))
			}

			sb.WriteString(fmt.Sprintf("%-15s  %-6d  %-6d  %-8d  %-6d  %-6d  %-6d  %-8.2f  %-8.0f  %-12s\n",
				playerName,
				stats.GamesPlayed,
				stats.TotalTurns,
				stats.TotalOptimal,
				stats.TotalSmall,
				stats.TotalMedium,
				stats.TotalLarge,
				stats.AvgMistakeIndex,
				stats.AvgEstimatedELO,
				bingoRate))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// formatGameAnalysisForBatch formats a single game analysis for batch display.
func (sc *ShellController) formatGameAnalysisForBatch(result *gameanalysis.GameAnalysisResult) string {
	return formatTurnTable(result) + formatPlayerSummaries(result.PlayerSummaries) + "\n"
}
