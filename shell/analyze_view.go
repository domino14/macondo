package shell

import (
	"fmt"
	"strings"

	"google.golang.org/protobuf/encoding/protojson"

	"github.com/domino14/macondo/gameanalysis"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// analyzeView loads and displays one or more stored analyses.
// Usage:
//
//	analyze-view <name> [<name> ...]         view named analyses (combined if multiple)
//	analyze-view -batch <name>               view all analyses in a batch
func (sc *ShellController) analyzeView(cmd *shellcmd) (*Response, error) {
	store, err := sc.getAnalysisStore()
	if err != nil {
		return nil, fmt.Errorf("open analysis store: %w", err)
	}

	batchName := cmd.options.String("batch")

	var stored []gameanalysis.StoredAnalysis

	if batchName != "" {
		// Load all analyses in the batch
		stored, err = store.ListByBatch(batchName)
		if err != nil {
			return nil, fmt.Errorf("list batch: %w", err)
		}
		if len(stored) == 0 {
			return msg(fmt.Sprintf("No analyses found for batch '%s'.", batchName)), nil
		}
	} else {
		// Load named analyses from positional args
		if len(cmd.args) == 0 {
			return msg("Usage: analyze-view <name> [<name> ...]\n       analyze-view -batch <batch-name>"), nil
		}
		for _, name := range cmd.args {
			s, err := store.Get(name)
			if err != nil {
				return nil, fmt.Errorf("load '%s': %w", name, err)
			}
			if s == nil {
				return msg(fmt.Sprintf("No analysis found with name '%s'.", name)), nil
			}
			stored = append(stored, *s)
		}
	}

	if len(stored) == 1 {
		return sc.viewSingleAnalysis(&stored[0])
	}
	return sc.viewCombinedAnalyses(stored)
}

// viewSingleAnalysis displays a single stored analysis.
func (sc *ShellController) viewSingleAnalysis(s *gameanalysis.StoredAnalysis) (*Response, error) {
	resultProto := &pb.GameAnalysisResult{}
	if err := protojson.Unmarshal(s.ResultJSON, resultProto); err != nil {
		return nil, fmt.Errorf("unmarshal result: %w", err)
	}

	output := formatStoredAnalysis(s, resultProto)
	return msg(output), nil
}

// viewCombinedAnalyses combines multiple stored analyses and shows aggregated stats.
func (sc *ShellController) viewCombinedAnalyses(stored []gameanalysis.StoredAnalysis) (*Response, error) {
	batchResult := gameanalysis.NewBatchAnalysisResult()

	for i := range stored {
		s := &stored[i]
		resultProto := &pb.GameAnalysisResult{}
		if err := protojson.Unmarshal(s.ResultJSON, resultProto); err != nil {
			sc.showMessage(fmt.Sprintf("Warning: skipping '%s' (unmarshal error: %v)", s.Name, err))
			continue
		}
		gameResult := &gameanalysis.BatchGameResult{
			GameID:   s.Name,
			GameInfo: s.PlayerInfo,
			Result:   gameanalysis.GameAnalysisResultFromProto(resultProto),
		}
		batchResult.AddGameResult(gameResult)
	}

	batchResult.CalculateAverages()
	output := sc.formatBatchResults(batchResult, false)
	return msg(output), nil
}

// formatStoredAnalysis formats a single stored analysis for display.
func formatStoredAnalysis(s *gameanalysis.StoredAnalysis, result *pb.GameAnalysisResult) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Stored Analysis: %s\n", s.Name))
	sb.WriteString(fmt.Sprintf("Players: %s\n", s.PlayerInfo))
	sb.WriteString(fmt.Sprintf("Lexicon: %s  |  Version: %d  |  Analyzer: %s  |  Date: %s\n",
		s.Lexicon, s.AnalysisVersion, s.AnalyzerVersion,
		s.CreatedAt.Format("2006-01-02 15:04")))
	if s.BatchName != "" {
		sb.WriteString(fmt.Sprintf("Batch: %s\n", s.BatchName))
	}
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n\n")

	// Turn-by-turn table
	if len(result.Turns) > 0 {
		sb.WriteString(fmt.Sprintf("%-4s  %-12s  %-8s  %-22s  %-22s  %-6s  %-8s  %-8s  %s\n",
			"Turn", "Player", "Rack", "Played", "Optimal", "Diff", "Phase", "Mistake", "Note"))
		sb.WriteString(strings.Repeat("-", 120))
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
			played := turn.PlayedMove
			if len([]rune(played)) > 22 {
				played = string([]rune(played)[:22])
			}
			optimal := turn.OptimalMove
			if len([]rune(optimal)) > 22 {
				optimal = string([]rune(optimal)[:22])
			}

			// Diff
			var diff string
			diff = formatDiff(turn.Phase == pb.GamePhase_PHASE_ENDGAME, turn.WasOptimal, turn.WinProbLoss, int(turn.SpreadLoss))

			// Phase name
			phase := protoPhaseShortName(turn.Phase)

			// Mistake
			mistake := mistakeSizeShortName(turn.MistakeSize)

			// Notes
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
				turn.TurnNumber, player, rack, played, optimal, diff, phase, mistake, note))
		}
		sb.WriteString("\n")
	}

	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\nPlayer Summary\n")
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n\n")

	// Player summaries
	sb.WriteString(fmt.Sprintf("%-15s  %-6s  %-8s  %-14s  %-13s  %-10s  %-12s\n",
		"Player", "Turns", "Optimal", "Avg Win% Loss", "Mistake Index", "Est. ELO", "Bingo Rate"))
	sb.WriteString(strings.Repeat("-", 110))
	sb.WriteString("\n")

	for _, ps := range result.PlayerSummaries {
		if ps == nil || ps.TurnsPlayed == 0 {
			continue
		}
		bingoRate := "—"
		if ps.AvailableBingos > 0 {
			made := ps.AvailableBingos - ps.MissedBingos
			bingoRate = fmt.Sprintf("%d/%d (%.0f%%)",
				made, ps.AvailableBingos,
				100.0*float64(made)/float64(ps.AvailableBingos))
		}
		sb.WriteString(fmt.Sprintf("%-15s  %-6d  %-8d  %-14s  %-13s  %-10.0f  %-12s\n",
			ps.PlayerName,
			ps.TurnsPlayed,
			ps.OptimalMoves,
			fmt.Sprintf("%.1f%%", ps.AvgWinProbLoss*100),
			fmt.Sprintf("%.2f", ps.MistakeIndex),
			ps.EstimatedElo,
			bingoRate))
	}

	return sb.String()
}

func protoPhaseShortName(phase pb.GamePhase) string {
	switch phase {
	case pb.GamePhase_PHASE_EARLY_MID:
		return "EarlyMid"
	case pb.GamePhase_PHASE_EARLY_PREENDGAME:
		return "Pre-EG"
	case pb.GamePhase_PHASE_PREENDGAME:
		return "PEG"
	case pb.GamePhase_PHASE_ENDGAME:
		return "Endgame"
	default:
		return "?"
	}
}

func mistakeSizeShortName(size pb.MistakeSize) string {
	switch size {
	case pb.MistakeSize_SMALL:
		return "Small"
	case pb.MistakeSize_MEDIUM:
		return "Medium"
	case pb.MistakeSize_LARGE:
		return "Large"
	default:
		return "-"
	}
}
