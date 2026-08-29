package shell

import (
	"encoding/json"
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
