// blank_challenges has utilities for generating racks with blanks
// that have 1 or more solutions.
package anagrammer

import (
	"context"
	"fmt"
	"log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
)

// try tries to generate challenges. It returns an error if it fails
// to generate a challenge with too many or too few answers, or if
// an answer has already been generated.
func try(nBlanks int, dist *alphabet.LetterDistribution, wordLength int,
	dawg *gaddag.SimpleGaddag, maxSolutions int, answerMap map[string]bool) (
	*Question, error) {

	alph := dawg.GetAlphabet()
	rack := alphabet.MachineWord(genRack(dist, wordLength, nBlanks, alph))
	answers := Anagram(rack.UserVisible(alph), dawg, ModeExact)
	if len(answers) == 0 || len(answers) > maxSolutions {
		// Try again!
		return nil, fmt.Errorf("too many or few answers: %v %v",
			len(answers), rack.UserVisible(alph))
	}
	for _, answer := range answers {
		if answerMap[answer] {
			return nil, fmt.Errorf("duplicate answer %v", answer)
		}
	}
	for _, answer := range answers {
		answerMap[answer] = true
	}
	w := alphabet.Word{Word: rack.UserVisible(alph), Dist: dist}
	return &Question{Q: w.MakeAlphagram(), A: answers}, nil
}

// GenerateBlanks - Generate a list of blank word challenges given the
// parameters in args.
func GenerateBlanks(ctx context.Context, args *BlankChallengeArgs,
	dinfo *dawgInfo) ([]*Question, int, error) {

	tries := 0
	// Handle 2-blank challenges at the end.
	// First gen 1-blank challenges.
	answerMap := make(map[string]bool)

	questions := []*Question{}
	qIndex := 0

	defer func() {
		log.Println("[DEBUG] Leaving GenerateBlanks")
	}()
	doIteration := func() (*Question, error) {
		if qIndex < args.NumQuestions-args.Num2Blanks {
			question, err := try(1, dinfo.dist, args.WordLength, dinfo.dawg,
				args.MaxSolutions, answerMap)
			tries++
			return question, err
		} else if qIndex < args.NumQuestions {
			question, err := try(2, dinfo.dist, args.WordLength, dinfo.dawg,
				args.MaxSolutions, answerMap)
			tries++
			return question, err
		}
		return nil, fmt.Errorf("iteration failed?")
	}

	for {
		select {
		case <-ctx.Done():
			return nil, 0, ctx.Err()

		default:
			question, err := doIteration()
			if err != nil {
				log.Printf("[DEBUG] %v", err)
				continue
			}
			questions = append(questions, question)
			qIndex++
			if len(questions) == args.NumQuestions {
				log.Printf("[DEBUG] %v tries", tries)
				return questions, len(answerMap), nil
			}
		}
	}

}

// genRack - Generate a random rack using `dist` and with `blanks` blanks.
func genRack(dist *alphabet.LetterDistribution, wordLength, blanks int,
	alph *alphabet.Alphabet) []alphabet.MachineLetter {

	bag := dist.MakeBag(alph)
	// it's a bag of machine letters.
	rack := make([]alphabet.MachineLetter, wordLength)
	idx := 0
	draw := func(avoidBlanks bool) alphabet.MachineLetter {
		var tiles []alphabet.MachineLetter
		if avoidBlanks {
			for tiles, _ = bag.Draw(1); tiles[0] == alphabet.BlankMachineLetter; {
				tiles, _ = bag.Draw(1)
			}
		} else {
			tiles, _ = bag.Draw(1)
		}
		return tiles[0]
	}
	for idx < wordLength-blanks {
		// Avoid blanks on draw if user specifies a number of blanks.
		rack[idx] = draw(blanks != 0)
		idx++
	}
	for ; idx < wordLength; idx++ {
		rack[idx] = alphabet.BlankMachineLetter
	}
	return rack
}
