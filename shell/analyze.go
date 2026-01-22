package shell

import (
	"context"
	"errors"
	"fmt"
	"strings"

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

	// Create analyzer
	analyzer := gameanalysis.New(sc.config, cfg)

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

	// Turn-by-turn analysis
	sb.WriteString(fmt.Sprintf("%-4s  %-12s  %-8s  %-18s  %-18s  %-6s  %-8s  %-8s  %s\n",
		"Turn", "Player", "Rack", "Played", "Optimal", "Diff", "Phase", "Mistake", "Note"))
	sb.WriteString(strings.Repeat("-", 125))
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
		var note string

		if turn.Phase == gameanalysis.PhaseEndgame {
			// Endgame: show spread difference
			if turn.WasOptimal {
				diff = "+0"
			} else {
				diff = fmt.Sprintf("%+d", turn.SpreadLoss)
			}
		} else if turn.Phase == gameanalysis.PhasePreEndgame && turn.SpreadLoss > 0 {
			// PEG with spread tiebreak: show both win% and spread
			diff = fmt.Sprintf("%.1f%% (%+d)", turn.WinProbLoss*100, turn.SpreadLoss)
		} else {
			// Sim/PEG: show win probability difference
			if turn.WasOptimal {
				diff = "0.0%"
			} else {
				diff = fmt.Sprintf("%.1f%%", turn.WinProbLoss*100)
			}
		}

		// Add notes for special cases
		if turn.IsPhony {
			if turn.PhonyChallenged {
				note = "❌ Phony (challenged off)"
			} else {
				note = "⚠️ Phony (unchallenged)"
			}
		} else if turn.MissedChallenge {
			note = "❌ Missed challenge"
		} else if turn.BlownEndgame {
			note = "💥 Blown endgame"
		}

		// Format mistake category
		mistake := turn.MistakeCategory
		if mistake == "" {
			mistake = "-"
		}

		// Determine phase display name
		phaseDisplay := turn.Phase.String()
		if turn.Phase == gameanalysis.PhaseEarlyMid && turn.TilesInBag <= 50 {
			phaseDisplay = "Middle"
		}

		sb.WriteString(fmt.Sprintf("%-4d  %-12s  %-8s  %-18s  %-18s  %-6s  %-8s  %-8s  %s\n",
			turn.TurnNumber,
			player,
			rack,
			played,
			optimal,
			diff,
			phaseDisplay,
			mistake,
			note))
	}

	sb.WriteString("\n")
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n")
	sb.WriteString("Player Summary\n")
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n\n")

	// Player summaries
	sb.WriteString(fmt.Sprintf("%-15s  %-6s  %-8s  %-14s  %-14s  %-13s  %-8s\n",
		"Player", "Turns", "Optimal", "Avg Win% Loss", "Avg Spd Loss", "Mistake Index", "Est. ELO"))
	sb.WriteString(strings.Repeat("-", 95))
	sb.WriteString("\n")

	for i := 0; i < 2; i++ {
		summary := result.PlayerSummaries[i]
		if summary.TurnsPlayed == 0 {
			continue
		}

		sb.WriteString(fmt.Sprintf("%-15s  %-6d  %-8d  %-14s  %-14s  %-13s  %-8s\n",
			summary.PlayerName,
			summary.TurnsPlayed,
			summary.OptimalMoves,
			fmt.Sprintf("%.1f%%", summary.AvgWinProbLoss*100),
			fmt.Sprintf("%.1f", summary.AvgSpreadLoss),
			fmt.Sprintf("%.1f", summary.MistakeIndex),
			summary.EstimatedELO))
	}

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
