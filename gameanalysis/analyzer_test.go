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

	analyzer := New(cfg, analysisCfg, "")
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

	if cfg.SimPlaysEarlyMid != 100 {
		t.Errorf("expected SimPlaysEarlyMid=100, got %d", cfg.SimPlaysEarlyMid)
	}

	if cfg.SimPliesEarlyMid != 5 {
		t.Errorf("expected SimPliesEarlyMid=5, got %d", cfg.SimPliesEarlyMid)
	}

	if cfg.SimStopEarlyMid != 99 {
		t.Errorf("expected SimStopEarlyMid=99, got %d", cfg.SimStopEarlyMid)
	}

	if cfg.SimPlaysEarlyPreEndgame != 100 {
		t.Errorf("expected SimPlaysEarlyPreEndgame=100, got %d", cfg.SimPlaysEarlyPreEndgame)
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

	if cfg.UseExposedOppRacks != true {
		t.Error("expected UseExposedOppRacks=true")
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
	analyzer := New(cfg, DefaultAnalysisConfig(), "")

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

func TestTiebreakByEquity(t *testing.T) {
	tests := []struct {
		name                          string
		optimalWinProb, playedWinProb float64
		expected                      bool
	}{
		{"ordinary position", 0.62, 0.55, false},
		// The whole field is hopeless: the optimal play is the maximum, so
		// nothing else can be above it either.
		{"optimal hopeless", 0.001, 0.0, true},
		{"both locked up", 0.9999, 0.999, true},
		// Reported in #503: the best play is a lock but the played move still
		// loses 15.8% of the time, which is a real win% difference.
		{"only optimal locked up", 0.9999, 0.842, false},
		{"just inside threshold", 0.9999, 0.9949, false},
	}

	for _, tt := range tests {
		got := tiebreakByEquity(tt.optimalWinProb, tt.playedWinProb)
		if got != tt.expected {
			t.Errorf("%s: tiebreakByEquity(%v, %v) = %v, expected %v",
				tt.name, tt.optimalWinProb, tt.playedWinProb, got, tt.expected)
		}
	}
}

// TestCategorizeMistake_SaturatedTopPlay covers the position from #503: B1 COD
// sims at 100% with +3.1 equity, the played B2 ODIC at 84.2% with -11.4. The
// equity tiebreak must stay off, so the 15.8% win probability loss grades as
// Large rather than the 14.5-point equity gap grading as Medium.
func TestCategorizeMistake_SaturatedTopPlay(t *testing.T) {
	const optimalWinProb, playedWinProb = 0.9999, 0.842
	const optimalEquity, playedEquity = 3.1, -11.4

	analysis := &TurnAnalysis{
		Phase:          PhaseEarlyPreEndgame,
		OptimalWinProb: optimalWinProb,
		PlayedWinProb:  playedWinProb,
		WinProbLoss:    optimalWinProb - playedWinProb,
		WasOptimal:     false,
	}

	// Mirrors what analyzeWithSim does once the sim finishes.
	if tiebreakByEquity(analysis.OptimalWinProb, analysis.PlayedWinProb) {
		if equityDiff := optimalEquity - playedEquity; equityDiff > 0.5 {
			analysis.SpreadLoss = int16(equityDiff + 0.5)
		}
	}

	if analysis.SpreadLoss != 0 {
		t.Errorf("expected no equity tiebreak, got SpreadLoss=%d", analysis.SpreadLoss)
	}
	if got := categorizeMistake(analysis); got != "Large" {
		t.Errorf("expected Large, got %q", got)
	}
}

// TestFinalizeGrade covers #454: a play that ties the engine's move costs
// nothing and is optimal, even though it is a different move.
func TestFinalizeGrade(t *testing.T) {
	tests := []struct {
		name             string
		analysis         TurnAnalysis
		expectedOptimal  bool
		expectedCategory string
	}{
		{
			name:            "played the engine's move",
			analysis:        TurnAnalysis{Phase: PhaseEarlyMid, WasOptimal: true},
			expectedOptimal: true,
		},
		{
			// Several endgame moves can share the best final spread; the solver
			// proved they are worth the same, so a different move is still optimal.
			name:            "endgame ties the best final spread",
			analysis:        TurnAnalysis{Phase: PhaseEndgame, SpreadLoss: 0},
			expectedOptimal: true,
		},
		{
			name: "endgame gives up spread",
			analysis: TurnAnalysis{Phase: PhaseEndgame, SpreadLoss: 3,
				OptimalFinalSpread: 20, CurrentSpread: 50}, // still winning, so not blown
			expectedOptimal:  false,
			expectedCategory: "Small",
		},
		{
			// Below the noise threshold the sim cannot tell the plays apart.
			name:            "sim loss within noise",
			analysis:        TurnAnalysis{Phase: PhaseEarlyPreEndgame, WinProbLoss: 0.001},
			expectedOptimal: true,
		},
		{
			name:             "sim loss above noise",
			analysis:         TurnAnalysis{Phase: PhaseEarlyPreEndgame, WinProbLoss: 0.004},
			expectedOptimal:  false,
			expectedCategory: "Small",
		},
		{
			// Tied on win%, but the equity tiebreak separates them.
			name:             "tied win% but loses spread",
			analysis:         TurnAnalysis{Phase: PhasePreEndgame, WinProbLoss: 0, SpreadLoss: 9},
			expectedOptimal:  false,
			expectedCategory: "Medium",
		},
	}

	for _, tt := range tests {
		analysis := tt.analysis
		finalizeGrade(&analysis)

		if analysis.WasOptimal != tt.expectedOptimal {
			t.Errorf("%s: WasOptimal = %v, expected %v", tt.name, analysis.WasOptimal, tt.expectedOptimal)
		}
		if analysis.MistakeCategory != tt.expectedCategory {
			t.Errorf("%s: MistakeCategory = %q, expected %q",
				tt.name, analysis.MistakeCategory, tt.expectedCategory)
		}
		// Every turn is either optimal or carries a category, never both or neither.
		if analysis.WasOptimal == (analysis.MistakeCategory != "") {
			t.Errorf("%s: optimal=%v and category=%q disagree",
				tt.name, analysis.WasOptimal, analysis.MistakeCategory)
		}
	}
}

func TestAnalyzeGame_NilHistory(t *testing.T) {
	cfg := &config.Config{}
	analyzer := New(cfg, DefaultAnalysisConfig(), "")

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
	analyzer := New(cfg, DefaultAnalysisConfig(), "")

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
	analyzer := New(cfg, DefaultAnalysisConfig(), "")
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
