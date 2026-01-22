package gameanalysis

import (
	"context"
	"testing"

	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

func TestAnalyzerCreation(t *testing.T) {
	cfg := &config.Config{}
	analysisCfg := DefaultAnalysisConfig()

	analyzer := New(cfg, analysisCfg)
	if analyzer == nil {
		t.Fatal("expected non-nil analyzer")
	}

	if analyzer.cfg != cfg {
		t.Error("analyzer config not set correctly")
	}

	if analyzer.analysisCfg != analysisCfg {
		t.Error("analyzer analysis config not set correctly")
	}
}

func TestDefaultAnalysisConfig(t *testing.T) {
	cfg := DefaultAnalysisConfig()

	if cfg.SimPlaysEarlyMid != 40 {
		t.Errorf("expected SimPlaysEarlyMid=40, got %d", cfg.SimPlaysEarlyMid)
	}

	if cfg.SimPliesEarlyMid != 5 {
		t.Errorf("expected SimPliesEarlyMid=5, got %d", cfg.SimPliesEarlyMid)
	}

	if cfg.SimStopEarlyMid != 99 {
		t.Errorf("expected SimStopEarlyMid=99, got %d", cfg.SimStopEarlyMid)
	}

	if cfg.SimPlaysEarlyPreEndgame != 80 {
		t.Errorf("expected SimPlaysEarlyPreEndgame=80, got %d", cfg.SimPlaysEarlyPreEndgame)
	}

	if cfg.SimPliesEarlyPreEndgame != 10 {
		t.Errorf("expected SimPliesEarlyPreEndgame=10, got %d", cfg.SimPliesEarlyPreEndgame)
	}

	if cfg.PEGEarlyCutoff != true {
		t.Error("expected PEGEarlyCutoff=true")
	}

	if cfg.OnlyPlayer != -1 {
		t.Errorf("expected OnlyPlayer=-1, got %d", cfg.OnlyPlayer)
	}
}

func TestPhaseString(t *testing.T) {
	tests := []struct {
		phase    GamePhase
		expected string
	}{
		{PhaseEarlyMid, "Early"},
		{PhaseEarlyPreEndgame, "EarlyPE"},
		{PhasePreEndgame, "PEG"},
		{PhaseEndgame, "Endgame"},
	}

	for _, tt := range tests {
		if got := tt.phase.String(); got != tt.expected {
			t.Errorf("phase %d: expected %q, got %q", tt.phase, tt.expected, got)
		}
	}
}

func TestDeterminePhase(t *testing.T) {
	cfg := &config.Config{}
	analyzer := New(cfg, DefaultAnalysisConfig())

	tests := []struct {
		tilesInBag int
		expected   GamePhase
	}{
		{10, PhaseEarlyMid},
		{8, PhaseEarlyMid},
		{7, PhaseEarlyPreEndgame},
		{5, PhaseEarlyPreEndgame},
		{2, PhaseEarlyPreEndgame},
		{1, PhasePreEndgame},
		{0, PhaseEndgame},
	}

	for _, tt := range tests {
		got := analyzer.determinePhase(tt.tilesInBag)
		if got != tt.expected {
			t.Errorf("tilesInBag=%d: expected %s, got %s", tt.tilesInBag, tt.expected, got)
		}
	}
}

func TestAnalyzeGame_NilHistory(t *testing.T) {
	cfg := &config.Config{}
	analyzer := New(cfg, DefaultAnalysisConfig())

	ctx := context.Background()
	_, err := analyzer.AnalyzeGame(ctx, nil)
	if err == nil {
		t.Error("expected error for nil history")
	}
}

func TestAnalyzeGame_EmptyHistory(t *testing.T) {
	t.Skip("Skipping test that requires full config setup")

	// This test would require proper config initialization with data paths, etc.
	// In a real test environment, we would set up a proper test config with:
	// - Lexicon data files
	// - Letter distributions
	// - Board layouts
	// For now, we skip this integration test and rely on manual testing.
}

func TestIsAnalyzableEvent(t *testing.T) {
	cfg := &config.Config{}
	analyzer := New(cfg, DefaultAnalysisConfig())

	tests := []struct {
		eventType pb.GameEvent_Type
		expected  bool
	}{
		{pb.GameEvent_TILE_PLACEMENT_MOVE, true},
		{pb.GameEvent_EXCHANGE, true},
		{pb.GameEvent_PASS, true},
		{pb.GameEvent_CHALLENGE, false},
		{pb.GameEvent_END_RACK_PTS, false},
		{pb.GameEvent_PHONY_TILES_RETURNED, false},
	}

	for _, tt := range tests {
		evt := &pb.GameEvent{Type: tt.eventType}
		got := analyzer.isAnalyzableEvent(evt)
		if got != tt.expected {
			t.Errorf("event type %v: expected %v, got %v", tt.eventType, tt.expected, got)
		}
	}
}

func TestCalculatePlayerSummaries(t *testing.T) {
	// Create a simple test case with summaries already populated
	// (as they would be after the main analysis loop)
	result := &GameAnalysisResult{
		Turns: []*TurnAnalysis{
			{
				PlayerIndex:      0,
				Phase:            PhaseEarlyMid,
				WasOptimal:       true,
				WinProbLoss:      0.0,
				OptimalWinProb:   0.5,
				PlayedWinProb:    0.5,
				MistakeCategory:  "",
			},
			{
				PlayerIndex:      0,
				Phase:            PhaseEarlyMid,
				WasOptimal:       false,
				WinProbLoss:      0.1,
				OptimalWinProb:   0.6,
				PlayedWinProb:    0.5,
				MistakeCategory:  "Large",
			},
			{
				PlayerIndex:     1,
				Phase:           PhaseEndgame,
				WasOptimal:      true,
				SpreadLoss:      0,
				MistakeCategory: "",
			},
		},
		PlayerSummaries: [2]*PlayerSummary{
			{PlayerName: "Player 1", TurnsPlayed: 2, OptimalMoves: 1},
			{PlayerName: "Player 2", TurnsPlayed: 1, OptimalMoves: 1},
		},
	}

	cfg := &config.Config{}
	analyzer := New(cfg, DefaultAnalysisConfig())
	analyzer.calculatePlayerSummaries(result)

	// Check that mistake index was calculated correctly
	// Player 0: 1 optimal (0 pts) + 1 non-optimal with 10% loss = Large mistake (1.0 pts)
	expectedMistakeIndex := 1.0
	if result.PlayerSummaries[0].MistakeIndex != expectedMistakeIndex {
		t.Errorf("player 0: expected mistake index %.1f, got %.1f",
			expectedMistakeIndex, result.PlayerSummaries[0].MistakeIndex)
	}

	// Player 1: 1 optimal = 0 mistake points
	expectedMistakeIndex1 := 0.0
	if result.PlayerSummaries[1].MistakeIndex != expectedMistakeIndex1 {
		t.Errorf("player 1: expected mistake index %.1f, got %.1f",
			expectedMistakeIndex1, result.PlayerSummaries[1].MistakeIndex)
	}

	// Check average win prob loss for player 0
	// Player 0 has 2 turns with win prob losses of 0.0 and 0.1
	// Average = 0.05
	expectedAvgLoss := 0.05
	if result.PlayerSummaries[0].AvgWinProbLoss != expectedAvgLoss {
		t.Errorf("player 0: expected avg win prob loss %.2f, got %.2f",
			expectedAvgLoss, result.PlayerSummaries[0].AvgWinProbLoss)
	}
}
