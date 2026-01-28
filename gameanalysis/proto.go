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
	}
}

// ToProto converts TurnAnalysis to protobuf message
func (t *TurnAnalysis) ToProto() *pb.TurnAnalysis {
	return &pb.TurnAnalysis{
		TurnNumber:       int32(t.TurnNumber),
		PlayerIndex:      int32(t.PlayerIndex),
		PlayerName:       t.PlayerName,
		Rack:             t.Rack,
		Phase:            phaseToProto(t.Phase),
		TilesInBag:       int32(t.TilesInBag),
		PlayedMove:       strings.TrimSpace(t.PlayedMove.ShortDescription()),
		PlayedScore:      int32(t.PlayedMove.Score()),
		OptimalMove:      strings.TrimSpace(t.OptimalMove.ShortDescription()),
		OptimalScore:     int32(t.OptimalMove.Score()),
		WinProbLoss:      t.WinProbLoss,
		SpreadLoss:       int32(t.SpreadLoss),
		WasOptimal:       t.WasOptimal,
		MistakeSize:      mistakeSizeToProto(t.MistakeCategory),
		BlownEndgame:     t.BlownEndgame,
		IsPhony:          t.IsPhony,
		PhonyChallenged:  t.PhonyChallenged,
		MissedChallenge:  t.MissedChallenge,
		OptimalIsBingo:   t.OptimalIsBingo,
		PlayedIsBingo:    t.PlayedIsBingo,
		MissedBingo:      t.MissedBingo,
	}
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
