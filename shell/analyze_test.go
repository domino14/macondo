package shell

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/domino14/macondo/gameanalysis"
)

// TestParseGameSource_Folder covers #508: a path is a folder source when it is
// one on disk, and a file source otherwise.
func TestParseGameSource_Folder(t *testing.T) {
	dir := t.TempDir()
	gcg := filepath.Join(dir, "game.gcg")
	if err := os.WriteFile(gcg, nil, 0644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		source       string
		expectedType string
	}{
		{dir, "dir"},
		{gcg, "file"},
		{filepath.Join(dir, "nonexistent.gcg"), "file"}, // let the loader report it
		{"woog:ABC123", "woogles"},
		{"xt:12345", "xt"},
		{"woogcollection:UUID", "collection"},
		{"https://example.com/game.gcg", "web"},
	}

	for _, tt := range tests {
		got, err := parseGameSource(tt.source)
		if err != nil {
			t.Errorf("%s: unexpected error %v", tt.source, err)
			continue
		}
		if got.Type != tt.expectedType {
			t.Errorf("%s: type = %q, expected %q", tt.source, got.Type, tt.expectedType)
		}
	}
}

func TestGCGFilesInDir(t *testing.T) {
	dir := t.TempDir()
	write := func(parts ...string) string {
		path := filepath.Join(append([]string{dir}, parts...)...)
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, nil, 0644); err != nil {
			t.Fatal(err)
		}
		return path
	}

	// Written out of order to check the walk sorts them.
	second := write("b.gcg")
	first := write("a.gcg")
	nested := write("round2", "c.gcg")
	upper := write("round2", "d.GCG") // extension match is case-insensitive
	write("notes.txt")
	write("game.gcg.bak")

	paths, err := gcgFilesInDir(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{first, second, nested, upper}
	if len(paths) != len(expected) {
		t.Fatalf("got %d files %v, expected %d %v", len(paths), paths, len(expected), expected)
	}
	for i, path := range paths {
		if path != expected[i] {
			t.Errorf("file %d = %q, expected %q", i, path, expected[i])
		}
	}

	// An empty folder is not an error; the caller reports it.
	empty, err := gcgFilesInDir(t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error on empty dir: %v", err)
	}
	if len(empty) != 0 {
		t.Errorf("expected no files in an empty dir, got %v", empty)
	}

	if _, err := gcgFilesInDir(filepath.Join(dir, "nope")); err == nil {
		t.Error("expected an error for a missing folder")
	}
}

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
