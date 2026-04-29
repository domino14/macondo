package shell

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"google.golang.org/protobuf/encoding/protojson"

	"github.com/domino14/macondo/gameanalysis"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// analyze analyzes every move in the loaded game
func (sc *ShellController) analyze(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("no game loaded; please load a game first with 'load'")
	}

	history := sc.game.History()
	if history == nil || len(history.Events) == 0 {
		return nil, errors.New("no game history to analyze")
	}

	// Parse options
	cfg := gameanalysis.DefaultAnalysisConfig()
	jsonFile := cmd.options.String("json")
	force := cmd.options.Bool("force")

	// Check for player filter option
	if playerOpt := cmd.options.String("player"); playerOpt != "" {
		// Try to parse as player number (0 or 1)
		if playerOpt == "0" {
			cfg.OnlyPlayer = 0
		} else if playerOpt == "1" {
			cfg.OnlyPlayer = 1
		} else {
			// Treat as player nickname
			cfg.OnlyPlayerByName = playerOpt
			cfg.OnlyPlayer = -1 // Will be resolved in analyzer
		}
	}

	if err := validateGameHistory(history, sc.game.Alphabet()); err != nil {
		return nil, fmt.Errorf("game history is corrupt: %w", err)
	}

	// Create analyzer
	analyzer := gameanalysis.New(sc.config, cfg, sc.macondoVersion)

	// Show progress message
	sc.showMessage(fmt.Sprintf("Analyzing game: %s vs %s (%d turns)",
		history.Players[0].Nickname,
		history.Players[1].Nickname,
		len(history.Events)))

	// Run analysis
	ctx := context.Background()
	result, err := analyzer.AnalyzeGame(ctx, history)
	if err != nil {
		return nil, fmt.Errorf("analysis failed: %w", err)
	}

	// Serialize to protojson (used for both JSON export and DB save)
	resultProto := result.ToProto()
	resultJSON, err := protojson.Marshal(resultProto)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize analysis: %w", err)
	}

	// JSON export
	if jsonFile != "" {
		if err := os.WriteFile(jsonFile, resultJSON, 0644); err != nil {
			sc.showMessage(fmt.Sprintf("Warning: failed to write JSON to %s: %v", jsonFile, err))
		} else {
			sc.showMessage(fmt.Sprintf("Analysis written to %s", jsonFile))
		}
	}

	// DB save (only when game was loaded from a known source)
	if sc.gameSource != "" {
		store, err := sc.getAnalysisStore()
		if err != nil {
			sc.showMessage(fmt.Sprintf("Warning: cannot open analysis store: %v", err))
		} else {
			if store.Exists(sc.gameSource) && !force {
				sc.showMessage(fmt.Sprintf("Note: Analysis already exists for '%s'. Use -force to overwrite.", sc.gameSource))
			} else {
				playerInfo := fmt.Sprintf("%s vs %s", history.Players[0].Nickname, history.Players[1].Nickname)
				if err := store.Save(sc.gameSource, "", playerInfo, history.Lexicon,
					result.AnalysisVersion, result.AnalyzerVersion, resultJSON); err != nil {
					sc.showMessage(fmt.Sprintf("Warning: failed to save analysis to store: %v", err))
				} else {
					sc.showMessage(fmt.Sprintf("Analysis saved to local store as '%s'", sc.gameSource))
				}
			}
		}
	}

	// Format and return results
	output := sc.formatAnalysisResults(result, history)
	return msg(output), nil
}

// formatAnalysisResults formats the analysis results for display
func (sc *ShellController) formatAnalysisResults(result *gameanalysis.GameAnalysisResult, history *pb.GameHistory) string {
	var sb strings.Builder

	// Header
	sb.WriteString(fmt.Sprintf("Game Analysis: %s vs %s\n",
		history.Players[0].Nickname,
		history.Players[1].Nickname))
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n\n")

	sb.WriteString(formatTurnTable(result))
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n")
	sb.WriteString("Player Summary\n")
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n\n")
	sb.WriteString(formatPlayerSummaries(result.PlayerSummaries))

	sb.WriteString("\n")

	// Add legend
	sb.WriteString("Notes:\n")
	sb.WriteString("  Mistake Categories:\n")
	sb.WriteString("    For win% loss (sim/PEG):\n")
	sb.WriteString("      Small  = ≤3% win loss = 0.2 pts\n")
	sb.WriteString("      Medium = 4-7% win loss = 0.5 pts\n")
	sb.WriteString("      Large  = >7% win loss = 1.0 pts\n")
	sb.WriteString("    For spread loss (PEG tiebreak/endgame):\n")
	sb.WriteString("      Small  = 1-7 pts spread = 0.2 pts\n")
	sb.WriteString("      Medium = 8-15 pts spread = 0.5 pts\n")
	sb.WriteString("      Large  = 16+ pts spread = 1.0 pts\n")
	sb.WriteString("      Exception: Blown endgame (win→tie/loss) = Large regardless of spread\n")
	sb.WriteString("  Mistake Index = Sum of mistake points for all turns\n")
	sb.WriteString("\n")
	sb.WriteString("  Diff Format:\n")
	sb.WriteString("    PEG with tied win%: Shows \"0.0% (+N)\" where N is spread difference\n")
	sb.WriteString("    Endgame: Shows spread difference in points\n")
	sb.WriteString("    Other phases: Shows win probability loss as percentage\n")
	sb.WriteString("\n")
	sb.WriteString("  ⚠️ Phony = Played unchallenged phony\n")
	sb.WriteString("  ❌ Missed challenge = Failed to challenge opponent's phony\n")
	sb.WriteString("  💥 Blown endgame = Mistake changed winning position to loss/tie\n")

	return sb.String()
}

// formatTurnTable renders the turn-by-turn table for a game analysis result.
// Works for both live analysis (PlayedMove/OptimalMove set) and results loaded
// from storage (PlayedMoveStr/OptimalMoveStr set, move pointers nil).
func formatTurnTable(result *gameanalysis.GameAnalysisResult) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("%-4s  %-12s  %-8s  %-22s  %-22s  %-6s  %-8s  %-8s  %s\n",
		"Turn", "Player", "Rack", "Played", "Optimal", "Diff", "Phase", "Mistake", "Note"))
	sb.WriteString(strings.Repeat("-", 125))
	sb.WriteString("\n")

	for _, turn := range result.Turns {
		player := turn.PlayerName
		if len([]rune(player)) > 12 {
			player = string([]rune(player)[:12])
		}
		rack := turn.Rack
		if len([]rune(rack)) > 7 {
			rack = string([]rune(rack)[:7])
		}

		var played, optimal string
		if turn.PlayedMove != nil {
			played = turn.PlayedMove.ShortDescription()
		} else {
			played = turn.PlayedMoveStr
		}
		if turn.OptimalMove != nil {
			optimal = turn.OptimalMove.ShortDescription()
		} else {
			optimal = turn.OptimalMoveStr
		}
		if len([]rune(played)) > 22 {
			played = string([]rune(played)[:22])
		}
		if len([]rune(optimal)) > 22 {
			optimal = string([]rune(optimal)[:22])
		}

		diff := formatDiff(turn.Phase == gameanalysis.PhaseEndgame, turn.WasOptimal, turn.WinProbLoss, int(turn.SpreadLoss))

		phaseDisplay := turn.Phase.String()
		if turn.Phase == gameanalysis.PhaseEarlyMid && turn.TilesInBag <= 50 {
			phaseDisplay = "Middle"
		}

		mistake := turn.MistakeCategory
		if mistake == "" {
			mistake = "-"
		}

		var notes []string
		if turn.IsPhony {
			if turn.PhonyChallenged {
				notes = append(notes, "Phony(off)")
			} else {
				notes = append(notes, "Phony")
			}
		}
		if turn.MissedChallenge {
			notes = append(notes, "MissedChallenge")
		}
		if turn.BlownEndgame {
			notes = append(notes, "BlownEG")
		}
		if turn.MissedBingo {
			notes = append(notes, "MissedBingo")
		}
		note := strings.Join(notes, " ")

		sb.WriteString(fmt.Sprintf("%-4d  %-12s  %-8s  %-22s  %-22s  %-6s  %-8s  %-8s  %s\n",
			turn.TurnNumber, player, rack, played, optimal, diff, phaseDisplay, mistake, note))
	}
	sb.WriteString("\n")

	return sb.String()
}

// formatPlayerSummaries formats player summary statistics (shared between single and batch analysis)
func formatPlayerSummaries(summaries [2]*gameanalysis.PlayerSummary) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("%-15s  %-6s  %-8s  %-14s  %-13s  %-10s  %-12s\n",
		"Player", "Turns", "Optimal", "Avg Win% Loss", "Mistake Index", "Est. ELO", "Bingo Rate"))
	sb.WriteString(strings.Repeat("-", 110))
	sb.WriteString("\n")

	for i := 0; i < 2; i++ {
		summary := summaries[i]
		if summary.TurnsPlayed == 0 {
			continue
		}

		// Calculate bingo rate
		bingoRate := "-"
		if summary.AvailableBingos > 0 {
			bingosMade := summary.AvailableBingos - summary.MissedBingos
			bingoRate = fmt.Sprintf("%d/%d (%.0f%%)",
				bingosMade, summary.AvailableBingos,
				100.0*float64(bingosMade)/float64(summary.AvailableBingos))
		}

		sb.WriteString(fmt.Sprintf("%-15s  %-6d  %-8d  %-14s  %-13s  %-10.0f  %-12s\n",
			summary.PlayerName,
			summary.TurnsPlayed,
			summary.OptimalMoves,
			fmt.Sprintf("%.1f%%", summary.AvgWinProbLoss*100),
			fmt.Sprintf("%.2f", summary.MistakeIndex),
			summary.EstimatedELO,
			bingoRate))
	}

	return sb.String()
}

// analyzeTurn analyzes the current turn in the loaded game
func (sc *ShellController) analyzeTurn(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("no game loaded; please load a game first with 'load'")
	}

	history := sc.game.History()
	if history == nil || len(history.Events) == 0 {
		return nil, errors.New("no game history to analyze")
	}

	// Get current turn number
	turnNumOpt := sc.game.Turn()
	if turnNumOpt >= len(history.Events) {
		return nil, fmt.Errorf("current turn %d is out of range (game has %d events)", turnNumOpt, len(history.Events))
	}

	// Create analyzer
	cfg := gameanalysis.DefaultAnalysisConfig()
	analyzer := gameanalysis.New(sc.config, cfg, sc.macondoVersion)

	// Analyze just this turn
	ctx := context.Background()
	analysis, err := analyzer.AnalyzeSingleTurnFromHistory(ctx, history, turnNumOpt)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze turn %d: %w", turnNumOpt, err)
	}

	// Format and return detailed results for this turn
	output := formatSingleTurnAnalysis(analysis)
	return msg(output), nil
}

// formatSingleTurnAnalysis formats detailed analysis for a single turn
func formatSingleTurnAnalysis(turn *gameanalysis.TurnAnalysis) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Turn %d Analysis\n", turn.TurnNumber))
	sb.WriteString(strings.Repeat("=", 60))
	sb.WriteString("\n\n")

	sb.WriteString(fmt.Sprintf("Player: %s\n", turn.PlayerName))
	sb.WriteString(fmt.Sprintf("Rack: %s\n", turn.Rack))
	sb.WriteString(fmt.Sprintf("Phase: %s (%d tiles in bag)\n", turn.Phase, turn.TilesInBag))
	sb.WriteString("\n")

	// Known opponent rack if available
	if turn.KnownOppRack != "" {
		sb.WriteString(fmt.Sprintf("Known Opponent Rack: %s\n", turn.KnownOppRack))
		sb.WriteString("  (from challenged phony)\n\n")
	}

	// Played move
	sb.WriteString(fmt.Sprintf("Played Move: %s\n", turn.PlayedMove.ShortDescription()))
	sb.WriteString(fmt.Sprintf("  Score: %d points\n", turn.PlayedMove.Score()))
	if turn.PlayedIsBingo {
		sb.WriteString("  BINGO!\n")
	}
	sb.WriteString("\n")

	// Optimal move
	sb.WriteString(fmt.Sprintf("Optimal Move: %s\n", turn.OptimalMove.ShortDescription()))
	sb.WriteString(fmt.Sprintf("  Score: %d points\n", turn.OptimalMove.Score()))
	if turn.OptimalIsBingo {
		sb.WriteString("  BINGO!\n")
	}
	sb.WriteString("\n")

	// Analysis metrics
	if turn.WasOptimal {
		sb.WriteString("✓ Optimal play!\n")
	} else {
		if turn.Phase == gameanalysis.PhaseEndgame {
			sb.WriteString(fmt.Sprintf("Spread Loss: %d points\n", turn.SpreadLoss))
		} else if turn.Phase == gameanalysis.PhasePreEndgame && turn.SpreadLoss > 0 {
			sb.WriteString(fmt.Sprintf("Win Prob Loss: %.2f%%\n", turn.WinProbLoss*100))
			sb.WriteString(fmt.Sprintf("Spread Loss (tiebreak): %d points\n", turn.SpreadLoss))
		} else {
			sb.WriteString(fmt.Sprintf("Win Prob Loss: %.2f%%\n", turn.WinProbLoss*100))
		}

		if turn.MistakeCategory != "" {
			sb.WriteString(fmt.Sprintf("Mistake Category: %s\n", turn.MistakeCategory))
		}
	}

	// Special flags
	if turn.IsPhony {
		sb.WriteString("\n⚠️ PHONY")
		if turn.PhonyChallenged {
			sb.WriteString(" (challenged off)")
		} else {
			sb.WriteString(" (unchallenged)")
		}
		sb.WriteString("\n")
	}

	if turn.MissedChallenge {
		sb.WriteString("\n❌ Missed Challenge - opponent played a phony!\n")
	}

	if turn.BlownEndgame {
		sb.WriteString("\n💥 Blown Endgame - changed winning position to loss/tie\n")
	}

	if turn.MissedBingo {
		sb.WriteString("\n⚠️ Missed Bingo - a bingo was available but not played\n")
	}

	return sb.String()
}

// formatDiff formats the equity/win-probability difference for a turn.
// isEndgame: true if the turn is in the endgame phase.
// wasOptimal: true if the played move was optimal.
// winProbLoss: fractional win-probability difference (0–1).
// spreadLoss: equity/spread loss when win probs are effectively tied.
func formatDiff(isEndgame, wasOptimal bool, winProbLoss float64, spreadLoss int) string {
	if isEndgame {
		if wasOptimal {
			return "+0"
		}
		return fmt.Sprintf("%+d", spreadLoss)
	}
	if !wasOptimal && spreadLoss > 0 {
		return fmt.Sprintf("%.1f%% (%+d)", winProbLoss*100, spreadLoss)
	}
	if wasOptimal {
		return "0.0%"
	}
	return fmt.Sprintf("%.1f%%", winProbLoss*100)
}
