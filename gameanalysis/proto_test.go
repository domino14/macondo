package gameanalysis

import (
	"testing"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/testhelpers"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/stretchr/testify/assert"
)

func TestIsBingo(t *testing.T) {
	alphabet := testhelpers.EnglishAlphabet()

	// 7-tile play is a bingo
	m := move.NewScoringMoveSimple(50, "8D", "PLAYING", "", alphabet)
	assert.True(t, isBingo(m))

	// 6-tile play is NOT a bingo
	m = move.NewScoringMoveSimple(40, "8D", "PLAYED", "", alphabet)
	assert.False(t, isBingo(m))

	// Exchange is NOT a bingo
	tiles := tilemapping.MachineWord{1, 5, 9}
	leave := tilemapping.MachineWord{2, 3, 4, 6}
	m = move.NewExchangeMove(tiles, leave, alphabet)
	assert.False(t, isBingo(m))

	// Pass is NOT a bingo
	rack := tilemapping.MachineWord{1, 2, 3, 4, 5, 6, 7}
	m = move.NewPassMove(rack, alphabet)
	assert.False(t, isBingo(m))

	// nil is NOT a bingo
	assert.False(t, isBingo(nil))
}

func TestPhaseToProto(t *testing.T) {
	assert.Equal(t, pb.GamePhase_PHASE_EARLY_MID, phaseToProto(PhaseEarlyMid))
	assert.Equal(t, pb.GamePhase_PHASE_EARLY_PREENDGAME, phaseToProto(PhaseEarlyPreEndgame))
	assert.Equal(t, pb.GamePhase_PHASE_PREENDGAME, phaseToProto(PhasePreEndgame))
	assert.Equal(t, pb.GamePhase_PHASE_ENDGAME, phaseToProto(PhaseEndgame))
}

func TestMistakeSizeToProto(t *testing.T) {
	assert.Equal(t, pb.MistakeSize_SMALL, mistakeSizeToProto("Small"))
	assert.Equal(t, pb.MistakeSize_MEDIUM, mistakeSizeToProto("Medium"))
	assert.Equal(t, pb.MistakeSize_LARGE, mistakeSizeToProto("Large"))
	assert.Equal(t, pb.MistakeSize_NO_MISTAKE, mistakeSizeToProto(""))
	assert.Equal(t, pb.MistakeSize_NO_MISTAKE, mistakeSizeToProto("Unknown"))
}

func TestToProto(t *testing.T) {
	alphabet := testhelpers.EnglishAlphabet()

	// Create sample moves
	playedMove := move.NewScoringMoveSimple(30, "8D", "PLAYED", "", alphabet)
	optimalMove := move.NewScoringMoveSimple(50, "8D", "PLAYING", "", alphabet)

	// Create a sample analysis result
	result := &GameAnalysisResult{
		Turns: []*TurnAnalysis{
			{
				TurnNumber:      1,
				PlayerIndex:     0,
				PlayerName:      "Alice",
				Rack:            "ABCDEFG",
				Phase:           PhaseEarlyMid,
				TilesInBag:      50,
				PlayedMove:      playedMove,
				OptimalMove:     optimalMove,
				WinProbLoss:     0.05,
				WasOptimal:      false,
				MistakeCategory: "Small",
				OptimalIsBingo:  true,
				PlayedIsBingo:   false,
				MissedBingo:     true,
			},
		},
		PlayerSummaries: [2]*PlayerSummary{
			{
				PlayerName:      "Alice",
				TurnsPlayed:     15,
				OptimalMoves:    12,
				AvgWinProbLoss:  0.03,
				AvgSpreadLoss:   2.5,
				SmallMistakes:   2,
				MediumMistakes:  1,
				LargeMistakes:   0,
				MistakeIndex:    1.5,
				EstimatedELO:    2050.5,
				AvailableBingos: 3,
				MissedBingos:    1,
			},
			{
				PlayerName:   "Bob",
				TurnsPlayed:  15,
				OptimalMoves: 10,
			},
		},
	}

	// Convert to protobuf
	protoResult := result.ToProto()

	// Verify
	assert.NotNil(t, protoResult)
	assert.Len(t, protoResult.Turns, 1)
	assert.Len(t, protoResult.PlayerSummaries, 2)

	// Check turn
	turn := protoResult.Turns[0]
	assert.Equal(t, int32(1), turn.TurnNumber)
	assert.Equal(t, int32(0), turn.PlayerIndex)
	assert.Equal(t, "Alice", turn.PlayerName)
	assert.Equal(t, "ABCDEFG", turn.Rack)
	assert.Equal(t, pb.GamePhase_PHASE_EARLY_MID, turn.Phase)
	assert.Equal(t, int32(50), turn.TilesInBag)
	assert.Equal(t, "8D PLAYED", turn.PlayedMove)
	assert.Equal(t, int32(30), turn.PlayedScore)
	assert.Equal(t, "8D PLAYING", turn.OptimalMove)
	assert.Equal(t, int32(50), turn.OptimalScore)
	assert.Equal(t, 0.05, turn.WinProbLoss)
	assert.False(t, turn.WasOptimal)
	assert.Equal(t, pb.MistakeSize_SMALL, turn.MistakeSize)
	assert.True(t, turn.OptimalIsBingo)
	assert.False(t, turn.PlayedIsBingo)
	assert.True(t, turn.MissedBingo)

	// Check summary
	summary := protoResult.PlayerSummaries[0]
	assert.Equal(t, "Alice", summary.PlayerName)
	assert.Equal(t, int32(15), summary.TurnsPlayed)
	assert.Equal(t, int32(12), summary.OptimalMoves)
	assert.Equal(t, 0.03, summary.AvgWinProbLoss)
	assert.Equal(t, 2.5, summary.AvgSpreadLoss)
	assert.Equal(t, int32(2), summary.SmallMistakes)
	assert.Equal(t, int32(1), summary.MediumMistakes)
	assert.Equal(t, int32(0), summary.LargeMistakes)
	assert.Equal(t, 1.5, summary.MistakeIndex)
	assert.Equal(t, 2050.5, summary.EstimatedElo)
	assert.Equal(t, int32(3), summary.AvailableBingos)
	assert.Equal(t, int32(1), summary.MissedBingos)
}
