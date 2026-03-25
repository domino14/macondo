package gameanalysis

import (
	"strings"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// ToProto converts GameAnalysisResult to protobuf message
func (r *GameAnalysisResult) ToProto() *pb.GameAnalysisResult {
	turns := make([]*pb.TurnAnalysis, len(r.Turns))
	for i, turn := range r.Turns {
		turns[i] = turn.ToProto()
	}

	playerSummaries := make([]*pb.PlayerSummary, len(r.PlayerSummaries))
	for i, summary := range r.PlayerSummaries {
		playerSummaries[i] = summary.ToProto()
	}

	return &pb.GameAnalysisResult{
		Turns:           turns,
		PlayerSummaries: playerSummaries,
		AnalysisVersion: int32(r.AnalysisVersion),
		AnalyzerVersion: r.AnalyzerVersion,
	}
}

// ToProto converts TurnAnalysis to protobuf message
func (t *TurnAnalysis) ToProto() *pb.TurnAnalysis {
	return &pb.TurnAnalysis{
		TurnNumber:           int32(t.TurnNumber),
		PlayerIndex:          int32(t.PlayerIndex),
		PlayerName:           t.PlayerName,
		Rack:                 t.Rack,
		Phase:                phaseToProto(t.Phase),
		TilesInBag:           int32(t.TilesInBag),
		PlayedMove:           strings.TrimSpace(t.PlayedMove.ShortDescription()),
		PlayedScore:          int32(t.PlayedMove.Score()),
		OptimalMove:          strings.TrimSpace(t.OptimalMove.ShortDescription()),
		OptimalScore:         int32(t.OptimalMove.Score()),
		WinProbLoss:          t.WinProbLoss,
		SpreadLoss:           int32(t.SpreadLoss),
		WasOptimal:           t.WasOptimal,
		MistakeSize:          mistakeSizeToProto(t.MistakeCategory),
		BlownEndgame:         t.BlownEndgame,
		IsPhony:              t.IsPhony,
		PhonyChallenged:      t.PhonyChallenged,
		MissedChallenge:      t.MissedChallenge,
		OptimalIsBingo:       t.OptimalIsBingo,
		PlayedIsBingo:        t.PlayedIsBingo,
		MissedBingo:          t.MissedBingo,
		KnownOppRack:         t.KnownOppRack,
		TopSimPlays:          simPlaysToProto(t.TopSimPlays),
		TopPegPlays:        pegPlaysToProto(t.TopPEGPlays),
		PrincipalVariation: endgameVarToProto(t.PrincipalVariation),
		OtherVariations:    endgameVarsToProto(t.OtherVariations),
	}
}

func simPlaysToProto(plays []*SimPlayResult) []*pb.SimmedPlayInfo {
	if len(plays) == 0 {
		return nil
	}
	result := make([]*pb.SimmedPlayInfo, len(plays))
	for i, p := range plays {
		plyStats := make([]*pb.PlyStats, len(p.PlyStats))
		for j, ps := range p.PlyStats {
			plyStats[j] = &pb.PlyStats{
				Ply:        int32(ps.Ply),
				ScoreMean:  ps.ScoreMean,
				ScoreStdev: ps.ScoreStdev,
				BingoPct:   ps.BingoPct,
			}
		}
		result[i] = &pb.SimmedPlayInfo{
			MoveDescription: p.MoveDescription,
			Score:           int32(p.Score),
			Leave:           p.Leave,
			IsBingo:         p.IsBingo,
			WinProb:         p.WinProb,
			WinProbStderr:   p.WinProbStdErr,
			Equity:          p.Equity,
			EquityStderr:    p.EquityStdErr,
			Iterations:      int32(p.Iterations),
			PlyStats:        plyStats,
			IsPlayedMove:    p.IsPlayedMove,
			IsIgnored:       p.IsIgnored,
		}
	}
	return result
}

func pegPlaysToProto(plays []*PEGPlayResult) []*pb.PEGPlayInfo {
	if len(plays) == 0 {
		return nil
	}
	result := make([]*pb.PEGPlayInfo, len(plays))
	for i, p := range plays {
		outcomes := make([]*pb.PEGOutcomeInfo, len(p.Outcomes))
		for j, o := range p.Outcomes {
			outcomes[j] = &pb.PEGOutcomeInfo{
				Tiles:   o.Tiles,
				Outcome: pegOutcomeStrToProto(o.Outcome),
				Count:   int32(o.Count),
			}
		}
		result[i] = &pb.PEGPlayInfo{
			MoveDescription: p.MoveDescription,
			Score:           int32(p.Score),
			Leave:           p.Leave,
			IsBingo:         p.IsBingo,
			WinProb:         p.WinProb,
			Outcomes:        outcomes,
			HasSpread:       p.HasSpread,
			AvgSpread:       p.AvgSpread,
			IsPlayedMove:    p.IsPlayedMove,
			IsIgnored:       p.IsIgnored,
		}
	}
	return result
}

func pegOutcomeStrToProto(s string) pb.PEGOutcomeType {
	switch s {
	case "win":
		return pb.PEGOutcomeType_PEG_OUTCOME_WIN
	case "draw":
		return pb.PEGOutcomeType_PEG_OUTCOME_DRAW
	case "loss":
		return pb.PEGOutcomeType_PEG_OUTCOME_LOSS
	default:
		return pb.PEGOutcomeType_PEG_OUTCOME_UNKNOWN
	}
}

func endgameVarToProto(v *EndgameVariationResult) *pb.EndgameVariation {
	if v == nil {
		return nil
	}
	moves := make([]*pb.EndgameMove, len(v.Moves))
	for i, m := range v.Moves {
		moves[i] = &pb.EndgameMove{
			MoveDescription: m.MoveDescription,
			Score:           int32(m.Score),
			MoveNumber:      int32(m.MoveNumber),
		}
	}
	return &pb.EndgameVariation{
		Moves:       moves,
		FinalSpread: int32(v.FinalSpread),
	}
}

func endgameVarsToProto(vars []*EndgameVariationResult) []*pb.EndgameVariation {
	if len(vars) == 0 {
		return nil
	}
	result := make([]*pb.EndgameVariation, len(vars))
	for i, v := range vars {
		result[i] = endgameVarToProto(v)
	}
	return result
}

// ToProto converts PlayerSummary to protobuf message
func (p *PlayerSummary) ToProto() *pb.PlayerSummary {
	return &pb.PlayerSummary{
		PlayerName:       p.PlayerName,
		TurnsPlayed:      int32(p.TurnsPlayed),
		OptimalMoves:     int32(p.OptimalMoves),
		AvgWinProbLoss:   p.AvgWinProbLoss,
		AvgSpreadLoss:    p.AvgSpreadLoss,
		SmallMistakes:    int32(p.SmallMistakes),
		MediumMistakes:   int32(p.MediumMistakes),
		LargeMistakes:    int32(p.LargeMistakes),
		MistakeIndex:     p.MistakeIndex,
		EstimatedElo:     p.EstimatedELO,
		AvailableBingos:  int32(p.AvailableBingos),
		MissedBingos:     int32(p.MissedBingos),
	}
}

// GameAnalysisResultFromProto constructs a GameAnalysisResult from a proto message.
// The returned struct has PlayerSummaries fully populated, and turns populated with
// display-string fields (move descriptions and scores from proto).
// PlayedMove and OptimalMove fields are nil since they require live game state.
func GameAnalysisResultFromProto(p *pb.GameAnalysisResult) *GameAnalysisResult {
	r := &GameAnalysisResult{
		AnalysisVersion: int(p.AnalysisVersion),
		AnalyzerVersion: p.AnalyzerVersion,
	}

	// Populate player summaries (used for batch stats aggregation)
	for i, ps := range p.PlayerSummaries {
		if i >= 2 {
			break
		}
		r.PlayerSummaries[i] = &PlayerSummary{
			PlayerName:      ps.PlayerName,
			TurnsPlayed:     int(ps.TurnsPlayed),
			OptimalMoves:    int(ps.OptimalMoves),
			AvgWinProbLoss:  ps.AvgWinProbLoss,
			AvgSpreadLoss:   ps.AvgSpreadLoss,
			SmallMistakes:   int(ps.SmallMistakes),
			MediumMistakes:  int(ps.MediumMistakes),
			LargeMistakes:   int(ps.LargeMistakes),
			MistakeIndex:    ps.MistakeIndex,
			EstimatedELO:    ps.EstimatedElo,
			AvailableBingos: int(ps.AvailableBingos),
			MissedBingos:    int(ps.MissedBingos),
		}
	}

	return r
}

func phaseToProto(phase GamePhase) pb.GamePhase {
	switch phase {
	case PhaseEarlyMid:
		return pb.GamePhase_PHASE_EARLY_MID
	case PhaseEarlyPreEndgame:
		return pb.GamePhase_PHASE_EARLY_PREENDGAME
	case PhasePreEndgame:
		return pb.GamePhase_PHASE_PREENDGAME
	case PhaseEndgame:
		return pb.GamePhase_PHASE_ENDGAME
	default:
		return pb.GamePhase_PHASE_EARLY_MID
	}
}

func mistakeSizeToProto(category string) pb.MistakeSize {
	switch category {
	case "Small":
		return pb.MistakeSize_SMALL
	case "Medium":
		return pb.MistakeSize_MEDIUM
	case "Large":
		return pb.MistakeSize_LARGE
	default:
		return pb.MistakeSize_NO_MISTAKE
	}
}
