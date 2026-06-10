package shell

import (
	"testing"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

func makePZ(score int32, words int32, tags ...pb.PuzzleTag) *pb.PuzzleCreationResponse {
	return &pb.PuzzleCreationResponse{
		Stats: &pb.PuzzleStats{
			Score:        score,
			WordsFormed:  words,
		},
		Tags: tags,
	}
}

func TestCompileFilterEmpty(t *testing.T) {
	p, err := compileFilter("")
	if err != nil {
		t.Fatalf("compileFilter empty: %v", err)
	}
	if p != nil {
		t.Fatal("expected nil predicate for empty filter")
	}
}

func TestCompileFilterSingleStat(t *testing.T) {
	p, err := compileFilter("score>=30")
	if err != nil {
		t.Fatalf("compileFilter: %v", err)
	}
	if !p(makePZ(30, 1)) {
		t.Error("score=30 should match score>=30")
	}
	if p(makePZ(29, 1)) {
		t.Error("score=29 should not match score>=30")
	}
}

func TestCompileFilterAndPrecedence(t *testing.T) {
	// "score>=30 or words_formed>=3 and tag=BINGO"
	// should parse as: score>=30 OR (words_formed>=3 AND tag=BINGO)
	p, err := compileFilter("score>=30 or words_formed>=3 and tag=BINGO")
	if err != nil {
		t.Fatalf("compileFilter: %v", err)
	}
	// score=35, words=1, no tags — only left side is true
	if !p(makePZ(35, 1)) {
		t.Error("score=35 should match via left OR")
	}
	// score=25, words=3, BINGO — right side true
	if !p(makePZ(25, 3, pb.PuzzleTag_BINGO)) {
		t.Error("words>=3 and BINGO should match via right side")
	}
	// score=25, words=3, no BINGO — right side false, left false
	if p(makePZ(25, 3)) {
		t.Error("words>=3 without BINGO should not match (AND requires both)")
	}
}

func TestCompileFilterParensOverridePrecedence(t *testing.T) {
	// "(score>=30 or words_formed>=3) and tag=BINGO"
	p, err := compileFilter("(score>=30 or words_formed>=3) and tag=BINGO")
	if err != nil {
		t.Fatalf("compileFilter: %v", err)
	}
	// score=35, has BINGO — both conditions true
	if !p(makePZ(35, 1, pb.PuzzleTag_BINGO)) {
		t.Error("score>=30 and BINGO should match")
	}
	// score=35, no BINGO — AND fails
	if p(makePZ(35, 1)) {
		t.Error("score>=30 without BINGO should not match")
	}
	// score=25, words=3, has BINGO — left OR via words_formed
	if !p(makePZ(25, 3, pb.PuzzleTag_BINGO)) {
		t.Error("words>=3 and BINGO should match")
	}
}

func TestCompileFilterNotPrecedence(t *testing.T) {
	// "not tag=BINGO and score>=30" → (not tag=BINGO) and score>=30
	p, err := compileFilter("not tag=BINGO and score>=30")
	if err != nil {
		t.Fatalf("compileFilter: %v", err)
	}
	// score=35, no BINGO → both conditions true
	if !p(makePZ(35, 1)) {
		t.Error("no BINGO and score>=30 should match")
	}
	// score=35, has BINGO → NOT BINGO fails
	if p(makePZ(35, 1, pb.PuzzleTag_BINGO)) {
		t.Error("has BINGO should not match (NOT BINGO)")
	}
	// score=25, no BINGO → score condition fails
	if p(makePZ(25, 1)) {
		t.Error("score<30 should not match")
	}
}

func TestCompileFilterNotGroup(t *testing.T) {
	// "not (tag=BINGO and score>=40)"
	p, err := compileFilter("not (tag=BINGO and score>=40)")
	if err != nil {
		t.Fatalf("compileFilter: %v", err)
	}
	// BINGO, score=50 → negated → false
	if p(makePZ(50, 1, pb.PuzzleTag_BINGO)) {
		t.Error("BINGO with score>=40 should be negated to false")
	}
	// BINGO, score=30 → inner false → negated → true
	if !p(makePZ(30, 1, pb.PuzzleTag_BINGO)) {
		t.Error("BINGO with score<40 → inner false → NOT = true")
	}
	// no BINGO, score=50 → inner false → negated → true
	if !p(makePZ(50, 1)) {
		t.Error("no BINGO, score=50 → inner false → NOT = true")
	}
}

func TestCompileFilterTagPresent(t *testing.T) {
	p, err := compileFilter("tag=BINGO")
	if err != nil {
		t.Fatalf("compileFilter: %v", err)
	}
	if !p(makePZ(30, 1, pb.PuzzleTag_BINGO)) {
		t.Error("BINGO tag should match tag=BINGO")
	}
	if p(makePZ(30, 1, pb.PuzzleTag_EQUITY)) {
		t.Error("EQUITY tag should not match tag=BINGO")
	}
}

func TestCompileFilterTagAbsent(t *testing.T) {
	p, err := compileFilter("tag!=BLANK_BINGO")
	if err != nil {
		t.Fatalf("compileFilter: %v", err)
	}
	if !p(makePZ(30, 1, pb.PuzzleTag_BINGO)) {
		t.Error("no BLANK_BINGO → tag!=BLANK_BINGO should be true")
	}
	if p(makePZ(30, 1, pb.PuzzleTag_BLANK_BINGO)) {
		t.Error("BLANK_BINGO present → tag!=BLANK_BINGO should be false")
	}
}

func TestCompileFilterErrors(t *testing.T) {
	cases := []struct {
		expr        string
		errContains string
	}{
		{"scor>=30", "unknown field"},
		{"tag=BINGOO", "unknown tag"},
		{"tag<BINGO", "not allowed for 'tag'"},
		{"score>=", "expected value"},
		{"(score>=30", "missing closing"},
		{"score>=30)", "unexpected token"},
		{"()", "unexpected"},
		{"not", "unexpected end"},
	}
	for _, tc := range cases {
		_, err := compileFilter(tc.expr)
		if err == nil {
			t.Errorf("compileFilter(%q): expected error containing %q, got nil", tc.expr, tc.errContains)
			continue
		}
		if tc.errContains != "" {
			found := false
			msg := err.Error()
			for i := 0; i+len(tc.errContains) <= len(msg); i++ {
				if msg[i:i+len(tc.errContains)] == tc.errContains {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("compileFilter(%q): error %q, want it to contain %q", tc.expr, err, tc.errContains)
			}
		}
	}
}
