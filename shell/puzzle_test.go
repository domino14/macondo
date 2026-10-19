package shell

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// TestPuzzleRecordRoundTrip is the contract between `puzzlegen -out` and
// `puzzle open`: whatever the writer puts in a line, the reader gets back. It
// covers the answer and stats in particular, since those travel as protojson
// inside the JSON rather than as plain fields.
func TestPuzzleRecordRoundTrip(t *testing.T) {
	answer := &pb.GameEvent{
		Type:        pb.GameEvent_TILE_PLACEMENT_MOVE,
		Position:    "L8",
		PlayedTiles: "QI",
		Rack:        "EEGIPQZ",
		Score:       44,
	}
	answerJSON, err := pgProtoJSON.Marshal(answer)
	if err != nil {
		t.Fatal(err)
	}
	statsJSON, err := pgProtoJSON.Marshal(&pb.PuzzleStats{
		Score: 44, WordsFormed: 3, TilesPlayed: 2, ScoreAdvantage: 11,
	})
	if err != nil {
		t.Fatal(err)
	}

	want := &pgRecord{
		GameID: "seed:abc", Turn: 16, Seed: "demo", Lexicon: "NWL23",
		LetterDistribution: "english",
		CGP:                "15/15/... EEGIPQZ/ 259/275 0 lex NWL23;",
		Board:              "line one\nline two",
		GCG:                "#character-encoding UTF-8\n>p1: ABC 8H AB +10 10",
		Answer:             answerJSON,
		Tags:               []string{"EQUITY", "NON_BINGO"},
		Stats:              statsJSON,
	}

	line, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got pgRecord
	if err := json.Unmarshal(line, &got); err != nil {
		t.Fatal(err)
	}

	if got.GameID != want.GameID || got.Turn != want.Turn || got.Seed != want.Seed {
		t.Errorf("provenance did not survive: %+v", got)
	}
	if got.CGP != want.CGP {
		t.Errorf("cgp = %q, want %q", got.CGP, want.CGP)
	}
	// The board and the GCG are both multi-line; JSON has to keep them intact
	// or `puzzle open` shows a mangled board.
	if got.Board != want.Board {
		t.Errorf("board = %q, want %q", got.Board, want.Board)
	}
	if got.GCG != want.GCG {
		t.Errorf("gcg = %q, want %q", got.GCG, want.GCG)
	}
	if strings.Join(got.Tags, " ") != "EQUITY NON_BINGO" {
		t.Errorf("tags = %v", got.Tags)
	}

	evt := got.answerEvent()
	if evt == nil {
		t.Fatal("answer did not decode")
	}
	if evt.GetPlayedTiles() != "QI" || evt.GetScore() != 44 || evt.GetRack() != "EEGIPQZ" {
		t.Errorf("answer decoded wrong: %v", evt)
	}
	if s := pgAnswerString(evt); !strings.Contains(s, "L8 QI") {
		t.Errorf("pgAnswerString = %q", s)
	}
}

// TestPgStatsAll checks that stats are reported under the names -filter uses,
// so someone reading `puzzle info` can paste a field straight into a filter.
func TestPgStatsAll(t *testing.T) {
	statsJSON, err := pgProtoJSON.Marshal(&pb.PuzzleStats{
		Score: 44, WordsFormed: 3, MaxCrossWordLength: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	got := pgStatsAll(statsJSON)
	for _, want := range []string{"score=44", "words_formed=3", "max_cross_word_length=2"} {
		if !strings.Contains(got, want) {
			t.Errorf("pgStatsAll = %q, missing %q", got, want)
		}
	}
	// Every name it prints must be one the filter language resolves.
	for _, field := range strings.Fields(got) {
		name, _, _ := strings.Cut(field, "=")
		if _, ok := pgResolveStatField(name); !ok {
			t.Errorf("pgStatsAll printed %q, which -filter does not accept", name)
		}
	}
	// Unset stats stay out rather than showing as a wall of zeroes.
	if strings.Contains(got, "tws_covered") {
		t.Errorf("pgStatsAll included an unset stat: %q", got)
	}
}

func TestPgShortID(t *testing.T) {
	long := "seed:p_pkdfz5gkIDbGqquOffXkC21U_btYfAtcOOmbvFMvc"
	if got := pgShortID(long); len([]rune(got)) != 13 || !strings.HasPrefix(long, strings.TrimSuffix(got, "…")) {
		t.Errorf("pgShortID(%q) = %q", long, got)
	}
	if got := pgShortID("short"); got != "short" {
		t.Errorf("pgShortID abbreviated a short id: %q", got)
	}
}

// TestPgVisibleTags pins which tags may sit above the board. Anything that
// describes the answer's shape has to stay in `puzzle info`, or the header
// solves the puzzle for the reader.
func TestPgVisibleTags(t *testing.T) {
	shown, hidden := pgVisibleTags([]string{
		"EQUITY", "NON_BINGO", "POWER_TILE", "CEL_PLUS_TWOS", "POINTS",
	})
	if strings.Join(shown, " ") != "EQUITY POINTS" {
		t.Errorf("shown = %v, want just the question-type tags", shown)
	}
	if hidden != 3 {
		t.Errorf("hidden = %d, want 3", hidden)
	}

	// Every tag the proto defines must be classified deliberately -- a new one
	// defaults to visible, so this fails until someone decides.
	for name := range pb.PuzzleTag_value {
		switch name {
		case "EQUITY", "POINTS":
			if pgShapeTags[name] {
				t.Errorf("%s should stay visible", name)
			}
		default:
			if !pgShapeTags[name] {
				t.Errorf("%s is not classified; decide whether it gives the answer's shape away", name)
			}
		}
	}
}

// TestPuzzleKeep covers the curation loop: keep a puzzle, refuse to keep it
// twice, take it back out, and end up with a file that is itself a puzzle file.
func TestPuzzleKeep(t *testing.T) {
	dir := t.TempDir()
	keepPath := filepath.Join(dir, "faves.jsonl")

	sc := &ShellController{
		puzzleFile: filepath.Join(dir, "source.jsonl"),
		puzzleSet: []*pgRecord{
			{GameID: "g1", Turn: 4, CGP: "cgp-one", Tags: []string{"EQUITY"}},
			{GameID: "g1", Turn: 9, CGP: "cgp-two", Tags: []string{"BINGO"}},
		},
	}

	if _, err := sc.puzzleKeep(keepPath); err != nil {
		t.Fatalf("keep: %v", err)
	}
	// The destination sticks, so the common case is a bare `puzzle keep`.
	sc.puzzleIdx = 1
	if _, err := sc.puzzleKeep(""); err != nil {
		t.Fatalf("keep without a path: %v", err)
	}
	if len(sc.puzzleKept) != 2 {
		t.Fatalf("kept %d, want 2", len(sc.puzzleKept))
	}

	// Keeping the same puzzle again must not duplicate it.
	if _, err := sc.puzzleKeep(""); err != nil {
		t.Fatalf("re-keep: %v", err)
	}
	if got := countLines(t, keepPath); got != 2 {
		t.Errorf("keep file has %d lines after a duplicate keep, want 2", got)
	}

	// What comes back out is a puzzle file, with the records intact.
	recs := readPuzzleFile(t, keepPath)
	if len(recs) != 2 || recs[0].CGP != "cgp-one" || recs[1].CGP != "cgp-two" {
		t.Fatalf("keep file did not round-trip: %+v", recs)
	}

	if _, err := sc.puzzleUnkeep(); err != nil {
		t.Fatalf("unkeep: %v", err)
	}
	recs = readPuzzleFile(t, keepPath)
	if len(recs) != 1 || recs[0].CGP != "cgp-one" {
		t.Errorf("unkeep removed the wrong record: %+v", recs)
	}
	if _, err := sc.puzzleUnkeep(); err == nil {
		t.Error("unkeep of a puzzle that isn't kept should fail")
	}
}

// TestPuzzleKeepRefusesOpenFile guards the one destructive mistake available:
// appending to the file being browsed would grow it while it is read.
func TestPuzzleKeepRefusesOpenFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "puzzles.jsonl")
	sc := &ShellController{
		puzzleFile: path,
		puzzleSet:  []*pgRecord{{GameID: "g1", Turn: 4}},
	}
	if _, err := sc.puzzleKeep(path); err == nil {
		t.Error("keeping into the open file should be refused")
	}
}

func countLines(t *testing.T, path string) int {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return len(strings.Split(strings.TrimSpace(string(b)), "\n"))
}

func readPuzzleFile(t *testing.T, path string) []*pgRecord {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out []*pgRecord
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		var rec pgRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("line %q: %v", line, err)
		}
		out = append(out, &rec)
	}
	return out
}

// TestPgDefaultKeepFile checks the file `puzzle keep` picks when nobody names
// one: beside the source, named after it, and never the source itself.
func TestPgDefaultKeepFile(t *testing.T) {
	for _, tc := range []struct{ open, want string }{
		{"/tmp/puzzles.jsonl", "/tmp/puzzles-kept.jsonl"},
		{"puzzles.jsonl", "puzzles-kept.jsonl"},
		{"favorites.jsonl", "favorites-kept.jsonl"},
		{"/a/b/set", "/a/b/set-kept"},
		{"", "kept.jsonl"},
		// A second round advances the number instead of stacking suffixes.
		{"puzzles-kept.jsonl", "puzzles-kept2.jsonl"},
		{"puzzles-kept2.jsonl", "puzzles-kept3.jsonl"},
		{"/tmp/puzzles-kept9.jsonl", "/tmp/puzzles-kept10.jsonl"},
	} {
		if got := pgDefaultKeepFile(tc.open); got != tc.want {
			t.Errorf("pgDefaultKeepFile(%q) = %q, want %q", tc.open, got, tc.want)
		}
		if tc.open != "" && pgDefaultKeepFile(tc.open) == tc.open {
			t.Errorf("default keep file collides with the open file: %q", tc.open)
		}
	}
}

// TestPuzzleKeepDefaultsToDerivedFile checks that keeping works with no setup.
func TestPuzzleKeepDefaultsToDerivedFile(t *testing.T) {
	dir := t.TempDir()
	sc := &ShellController{
		puzzleFile: filepath.Join(dir, "source.jsonl"),
		puzzleSet:  []*pgRecord{{GameID: "g1", Turn: 4, CGP: "cgp-one"}},
	}
	if _, err := sc.puzzleKeep(""); err != nil {
		t.Fatalf("keep with no destination named: %v", err)
	}
	want := filepath.Join(dir, "source-kept.jsonl")
	if sc.puzzleKeepFile != want {
		t.Errorf("kept into %q, want %q", sc.puzzleKeepFile, want)
	}
	if recs := readPuzzleFile(t, want); len(recs) != 1 || recs[0].CGP != "cgp-one" {
		t.Errorf("derived keep file did not get the record: %+v", recs)
	}
}

// TestPuzzleUnkeepPrunesOpenFile covers reviewing a collection you built: with
// no keep file named, unkeep takes the puzzle out of the file in front of you,
// and out of the browser with it.
func TestPuzzleUnkeepPrunesOpenFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "favorites.jsonl")
	writePuzzleFile(t, path, []*pgRecord{
		{GameID: "g1", Turn: 4, CGP: "one"},
		{GameID: "g1", Turn: 9, CGP: "two"},
		{GameID: "g2", Turn: 2, CGP: "three"},
	})

	sc := &ShellController{puzzleFile: path, puzzleIdx: 1}
	sc.puzzleSet = readPuzzleFile(t, path)

	// The file rewrite and the browser update, without the redraw that follows
	// them -- redrawing loads the position, which needs a whole game.
	removed, err := pgRemoveFromPuzzleFile(path, pgKeepKey(sc.puzzleSet[1]))
	if err != nil {
		t.Fatal(err)
	}
	if removed != 1 {
		t.Fatalf("removed %d records, want 1", removed)
	}
	if gone := sc.dropCurrentFromBrowser(); gone != 2 {
		t.Errorf("dropped puzzle %d, want 2", gone)
	}

	recs := readPuzzleFile(t, path)
	if len(recs) != 2 || recs[0].CGP != "one" || recs[1].CGP != "three" {
		t.Errorf("file after unkeep: %+v", recs)
	}
	if len(sc.puzzleSet) != 2 {
		t.Errorf("browser still holds %d puzzles, want 2", len(sc.puzzleSet))
	}
}

// TestPuzzleKeepOnACollection: curating a collection again is ordinary, so it
// works, and the name advances the round rather than stacking suffixes.
func TestPuzzleKeepOnACollection(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "puzzles-kept.jsonl")
	sc := &ShellController{
		puzzleFile: path,
		puzzleSet:  []*pgRecord{{GameID: "g1", Turn: 4, CGP: "one"}},
	}
	if _, err := sc.puzzleKeep(""); err != nil {
		t.Fatalf("keeping out of a collection: %v", err)
	}
	want := filepath.Join(dir, "puzzles-kept2.jsonl")
	if sc.puzzleKeepFile != want {
		t.Errorf("kept into %q, want %q", sc.puzzleKeepFile, want)
	}
}

func writePuzzleFile(t *testing.T, path string, recs []*pgRecord) {
	t.Helper()
	var b []byte
	for _, rec := range recs {
		line, err := json.Marshal(rec)
		if err != nil {
			t.Fatal(err)
		}
		b = append(append(b, line...), '\n')
	}
	if err := os.WriteFile(path, b, 0644); err != nil {
		t.Fatal(err)
	}
}
