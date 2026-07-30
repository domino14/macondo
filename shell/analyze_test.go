package shell

import (
	"strings"
	"testing"

	"github.com/domino14/macondo/gameanalysis"
)

func TestFormatDiff(t *testing.T) {
	tests := []struct {
		name        string
		isEndgame   bool
		winProbLoss float64
		spreadLoss  int
		expected    string
	}{
		{name: "no loss", expected: "0.0%"},
		{name: "win prob loss", winProbLoss: 0.043, expected: "4.3%"},
		// #454: a loss too small to be a mistake still gets reported as what it
		// was, rather than rounding into a bare zero.
		{name: "loss inside the noise threshold", winProbLoss: 0.001, expected: "0.1%"},
		{name: "spread tiebreak", winProbLoss: 0, spreadLoss: 9, expected: "0.0% (+9)"},
		{name: "endgame tie", isEndgame: true, expected: "+0"},
		{name: "endgame loss", isEndgame: true, spreadLoss: 3, expected: "+3"},
	}

	for _, tt := range tests {
		got := formatDiff(tt.isEndgame, tt.winProbLoss, tt.spreadLoss)
		if got != tt.expected {
			t.Errorf("%s: formatDiff(%v, %v, %v) = %q, expected %q",
				tt.name, tt.isEndgame, tt.winProbLoss, tt.spreadLoss, got, tt.expected)
		}
	}
}

// TestFormatTurnTable_TieNote checks that a turn that is optimal by a move
// other than the one in the Optimal column says so, since the row would
// otherwise read as though the two columns contradict each other.
func TestFormatTurnTable_TieNote(t *testing.T) {
	result := &gameanalysis.GameAnalysisResult{
		Turns: []*gameanalysis.TurnAnalysis{
			{
				TurnNumber: 20, PlayerName: "porch_microwave", Rack: "FIJKLNU",
				PlayedMoveStr: "13K J.NK", OptimalMoveStr: "14G F.LK",
				Phase: gameanalysis.PhaseEarlyPreEndgame, TilesInBag: 5,
				WinProbLoss: 0.0004, WasOptimal: true,
			},
			{
				TurnNumber: 18, PlayerName: "porch_microwave", Rack: "AEIMMRT",
				PlayedMoveStr: "1I MARMiTE", OptimalMoveStr: "1I MARMiTE",
				Phase: gameanalysis.PhaseEarlyMid, TilesInBag: 20,
				WasOptimal: true,
			},
		},
	}

	lines := strings.Split(strings.TrimSpace(formatTurnTable(result)), "\n")
	var tied, same string
	for _, line := range lines {
		switch {
		case strings.HasPrefix(line, "20 "):
			tied = line
		case strings.HasPrefix(line, "18 "):
			same = line
		}
	}

	if !strings.Contains(tied, "Tie") {
		t.Errorf("expected a Tie note on the turn won by a different move, got %q", tied)
	}
	if strings.Contains(same, "Tie") {
		t.Errorf("did not expect a Tie note when the moves match, got %q", same)
	}
}
