package shell

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/protobuf/encoding/protojson"

	"github.com/domino14/macondo/gameanalysis"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
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

// Symlinked folders are ordinary folders to the user, so they have to be
// walked; filepath.WalkDir would have skipped them and reported no games.
func TestGCGFilesInDir_Symlinks(t *testing.T) {
	root := t.TempDir()
	real := filepath.Join(root, "real")
	if err := os.MkdirAll(real, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(real, "a.gcg"), nil, 0644); err != nil {
		t.Fatal(err)
	}

	link := filepath.Join(root, "link")
	if err := os.Symlink(real, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	// A link used as the folder argument.
	paths, err := gcgFilesInDir(link)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(paths) != 1 {
		t.Errorf("symlinked folder: got %v, expected one .gcg file", paths)
	}

	// A link nested inside the folder being scanned. "real" and "link" both
	// resolve to the same directory, so the file is reported once.
	paths, err = gcgFilesInDir(root)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(paths) != 1 {
		t.Errorf("folder containing a symlink: got %v, expected one .gcg file", paths)
	}

	// A link pointing at its own parent must not loop forever.
	if err := os.Symlink(root, filepath.Join(real, "up")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if _, err := gcgFilesInDir(root); err != nil {
		t.Fatalf("unexpected error walking a cyclic link: %v", err)
	}

	// A dangling link is skipped rather than failing the whole scan.
	if err := os.Symlink(filepath.Join(root, "gone"), filepath.Join(root, "dead.gcg")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if _, err := gcgFilesInDir(root); err != nil {
		t.Fatalf("unexpected error walking a dangling link: %v", err)
	}
}

// The shell has no OS shell in front of it, so "~/games" arrives verbatim and
// has to be expanded before it can be recognized as a folder.
func TestParseGameSource_Tilde(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home directory: %v", err)
	}
	dir, err := os.MkdirTemp(home, "macondo-gcg-test-")
	if err != nil {
		t.Skipf("cannot create a folder under home: %v", err)
	}
	defer os.RemoveAll(dir)

	rel, err := filepath.Rel(home, dir)
	if err != nil {
		t.Fatal(err)
	}

	got, err := parseGameSource("~/" + rel)
	if err != nil {
		t.Fatal(err)
	}
	if got.Type != "dir" {
		t.Errorf("type = %q, expected %q", got.Type, "dir")
	}
	if got.Identifier != dir {
		t.Errorf("identifier = %q, expected %q", got.Identifier, dir)
	}
}

// A stored analysis is only a substitute for re-running a game when it
// answers the question this run is asking.
func TestReusableAnalysis(t *testing.T) {
	storedFor := func(version int, turns0, turns1 int32) *gameanalysis.StoredAnalysis {
		resultJSON, err := protojson.Marshal(&pb.GameAnalysisResult{
			AnalysisVersion: int32(version),
			PlayerSummaries: []*pb.PlayerSummary{
				{PlayerName: "esmith", TurnsPlayed: turns0},
				{PlayerName: "sammy", TurnsPlayed: turns1},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return &gameanalysis.StoredAnalysis{AnalysisVersion: version, ResultJSON: resultJSON}
	}
	current := gameanalysis.CurrentAnalysisVersion
	players := []*pb.PlayerInfo{
		{Nickname: "esmith", RealName: "Eric Smith"},
		{Nickname: "sammy", RealName: "Sammy Okosagah"},
	}
	bothPlayers := &gameanalysis.AnalysisConfig{OnlyPlayer: -1}
	byFullName := &gameanalysis.AnalysisConfig{OnlyPlayer: -1, OnlyPlayerByName: "Eric Smith"}

	tests := []struct {
		name        string
		stored      *gameanalysis.StoredAnalysis
		cfg         *gameanalysis.AnalysisConfig
		players     []*pb.PlayerInfo
		expectReuse bool
		expectWhy   string
	}{
		{"complete and current", storedFor(current, 12, 11), bothPlayers, nil, true, ""},
		{"stale version", storedFor(current-1, 12, 11), bothPlayers, nil, false, "version"},
		{"analyzed one player only", storedFor(current, 12, 0), bothPlayers, nil, false, "player"},
		{"full name needs the player list", storedFor(current, 12, 0), byFullName, nil, false, reuseUndecided},
		{"full name with the player list", storedFor(current, 12, 0), byFullName, players, true, ""},
		{"unreadable result", &gameanalysis.StoredAnalysis{
			AnalysisVersion: current, ResultJSON: []byte("not json")}, bothPlayers, nil, false, "could not be read"},
	}

	for _, tt := range tests {
		result, why := reusableAnalysis(tt.stored, tt.cfg, tt.players)
		if (result != nil) != tt.expectReuse {
			t.Errorf("%s: reuse = %v, expected %v (reason %q)", tt.name, result != nil, tt.expectReuse, why)
			continue
		}
		if !strings.Contains(why, tt.expectWhy) {
			t.Errorf("%s: reason = %q, expected it to mention %q", tt.name, why, tt.expectWhy)
		}
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
