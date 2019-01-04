package anagrammer

import (
	"context"
	"strings"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/lexicon"
)

func TestRacks(t *testing.T) {
	dists := []lexicon.LetterDistribution{
		lexicon.EnglishLetterDistribution(),
		lexicon.SpanishLetterDistribution(),
	}
	for _, dist := range dists {
		for l := 7; l <= 8; l++ {
			for n := 1; n <= 2; n++ {
				for i := 0; i < 10000; i++ {
					rack := genRack(dist, l, n, nil)
					if len(rack) != l {
						t.Errorf("Len rack should have been %v, was %v",
							l, len(rack))
					}
					numBlanks := 0
					for j := 0; j < len(rack); j++ {
						if rack[j] == alphabet.BlankToken {
							numBlanks++
						}
					}
					if numBlanks != n {
						t.Errorf("Should have had exactly %v blanks, got %v",
							n, numBlanks)
					}
				}
			}
		}
	}
}

func TestGenBlanks(t *testing.T) {
	gaddagmaker.GenerateDawg(LexiconDir+"America.txt", true, true)
	d := gaddag.SimpleDawg(gaddag.LoadGaddag("out.dawg"))

	ctx := context.Background()
	bcArgs := &BlankChallengeArgs{
		WordLength: 7, NumQuestions: 25, Lexicon: "America", MaxSolutions: 5,
		Num2Blanks: 6,
	}
	qs, sols, err := GenerateBlanks(ctx, bcArgs, d)
	if err != nil {
		t.Errorf("GenBlanks returned an error: %v", err)
	}
	actualSols := 0
	num2Blanks := 0
	if len(qs) != bcArgs.NumQuestions {
		t.Errorf("Generated %v questions, expected %v", len(qs), bcArgs.NumQuestions)
	}
	for _, q := range qs {
		if strings.Count(q.Q, "?") == 2 {
			num2Blanks++
		}
		if len(q.A) > bcArgs.MaxSolutions {
			t.Errorf("Number of solutions was %v, expected <= %v", len(q.A),
				bcArgs.MaxSolutions)
		}
		actualSols += len(q.A)
	}
	if num2Blanks != bcArgs.Num2Blanks {
		t.Errorf("Expected %v 2-blank questions, got %v", bcArgs.Num2Blanks,
			num2Blanks)
	}
	if actualSols != sols {
		t.Errorf("Expected %v solutions, got %v solutions", sols, actualSols)
	}
}
