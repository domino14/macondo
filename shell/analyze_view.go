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

func unmarshalStoredAnalysis(s *gameanalysis.StoredAnalysis) (*gameanalysis.GameAnalysisResult, error) {
	resultProto := &pb.GameAnalysisResult{}
	if err := protojson.Unmarshal(s.ResultJSON, resultProto); err != nil {
		return nil, fmt.Errorf("unmarshal result: %w", err)
	}
	return gameanalysis.GameAnalysisResultFromProto(resultProto), nil
}

// viewSingleAnalysis displays a single stored analysis.
func (sc *ShellController) viewSingleAnalysis(s *gameanalysis.StoredAnalysis) (*Response, error) {
	result, err := unmarshalStoredAnalysis(s)
	if err != nil {
		return nil, err
	}
	return msg(formatStoredAnalysis(s, result)), nil
}

// viewCombinedAnalyses combines multiple stored analyses and shows aggregated stats.
func (sc *ShellController) viewCombinedAnalyses(stored []gameanalysis.StoredAnalysis) (*Response, error) {
	batchResult := gameanalysis.NewBatchAnalysisResult()

	for i := range stored {
		s := &stored[i]
		result, err := unmarshalStoredAnalysis(s)
		if err != nil {
			sc.showMessage(fmt.Sprintf("Warning: skipping '%s' (unmarshal error: %v)", s.Name, err))
			continue
		}
		batchResult.AddGameResult(&gameanalysis.BatchGameResult{
			GameID:   s.Name,
			GameInfo: s.PlayerInfo,
			Result:   result,
		})
	}

	batchResult.CalculateAverages()
	return msg(sc.formatBatchResults(batchResult, false)), nil
}

// formatStoredAnalysis formats a single stored analysis for display.
func formatStoredAnalysis(s *gameanalysis.StoredAnalysis, result *gameanalysis.GameAnalysisResult) string {
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

	sb.WriteString(formatTurnTable(result))

	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\nPlayer Summary\n")
	sb.WriteString(strings.Repeat("=", 80))
	sb.WriteString("\n\n")
	sb.WriteString(formatPlayerSummaries(result.PlayerSummaries))

	return sb.String()
}
