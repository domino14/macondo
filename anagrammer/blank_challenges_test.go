package anagrammer

import "testing"
import "github.com/domino14/macondo/lexicon"

func TestRacks(t *testing.T) {
	dists := []lexicon.LetterDistribution{
		lexicon.EnglishLetterDistribution(),
		lexicon.SpanishLetterDistribution(),
	}
	for _, dist := range dists {
		for l := 7; l <= 8; l++ {
			for n := 1; n <= 2; n++ {
				for i := 0; i < 10000; i++ {
					rack := genRack(dist, l, n)
					if len(rack) != l {
						t.Errorf("Len rack should have been %v, was %v",
							l, len(rack))
					}
					numBlanks := 0
					for j := 0; j < len(rack); j++ {
						if rack[j] == BlankCharacter {
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

// func TestGenBlanks(t *testing.T) {
// 	dist := []lexicon.SpanishLetterDistribution()
// 	bcArgs := &BlankChallengeArgs{
// 		WordLength: 7, NumQuestions: 25, Lexicon: "FISE09", MaxSolutions: 5,
// 		Num2Blanks: 1,
// 	}
// 	GenerateBlanks(bcArgs, Dawgs["FISE09"])
// }
