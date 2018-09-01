// blank_challenges has utilities for generating racks with blanks
// that have 1 or more solutions.
package anagrammer

import (
	"context"
	"fmt"
	"log"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
)

const BlankCharacter = '?'

// try tries to generate challenges. It returns an error if it fails
// to generate a challenge with too many or too few answers, or if
// an answer has already been generated.
func try(nBlanks int, dist lexicon.LetterDistribution, wordLength int,
	dawg gaddag.SimpleDawg, maxSolutions int, answerMap map[string]bool,
	questionChan chan *Question, errChan chan error) bool {

	rack := genRack(dist, wordLength, nBlanks)
	answers := Anagram(string(rack), dawg, ModeExact)
	if len(answers) == 0 || len(answers) > maxSolutions {
		// Try again!
		errChan <- fmt.Errorf("too many or few answers: %v %v",
			len(answers), string(rack))
		return false
	}
	for _, answer := range answers {
		if answerMap[answer] {
			errChan <- fmt.Errorf("duplicate answer %v", answer)
			return false
		}
	}
	for _, answer := range answers {
		answerMap[answer] = true
	}
	w := lexicon.Word{Word: string(rack), Dist: dist}
	questionChan <- &Question{Q: w.MakeAlphagram(), A: answers}
	return true
}

// GenerateBlanks - Generate a list of blank word challenges given the
// parameters in args.
func GenerateBlanks(ctx context.Context, args *BlankChallengeArgs,
	dawg gaddag.SimpleDawg) ([]*Question, int, error) {

	var dist lexicon.LetterDistribution
	if args.Lexicon == "FISE09" {
		dist = lexicon.SpanishLetterDistribution()
	} else {
		dist = lexicon.EnglishLetterDistribution()
	}
	tries := 0
	// Handle 2-blank challenges at the end.
	// First gen 1-blank challenges.
	answerMap := make(map[string]bool)
	questionChan := make(chan *Question)
	errChan := make(chan error)
	questions := []*Question{}
	qIndex := 0

	go func() {
		for qIndex < args.NumQuestions-args.Num2Blanks {
			success := try(1, dist, args.WordLength, dawg, args.MaxSolutions,
				answerMap, questionChan, errChan)
			tries++
			if !success {
				continue
			}
			qIndex++
		}
		for qIndex < args.NumQuestions {
			success := try(2, dist, args.WordLength, dawg, args.MaxSolutions,
				answerMap, questionChan, errChan)
			tries++
			if !success {
				continue
			}
			qIndex++
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return nil, 0, ctx.Err()

		case q := <-questionChan:
			questions = append(questions, q)
			if len(questions) == args.NumQuestions {
				log.Println(tries, "tries")
				return questions, len(answerMap), nil
			}
		case e := <-errChan:
			log.Printf("[DEBUG] %v", e)
		}

	}

}

// genRack - Generate a random rack using `dist` and with `blanks` blanks.
func genRack(dist lexicon.LetterDistribution, wordLength, blanks int) []rune {
	bag := dist.MakeBag()
	// it's a bag of runes.
	rack := make([]rune, wordLength)
	idx := 0
	draw := func(avoidBlanks bool) rune {
		var tile rune
		if avoidBlanks {
			for tile, _ = bag.Draw(); tile == BlankCharacter; {
				tile, _ = bag.Draw()
			}
		} else {
			tile, _ = bag.Draw()
		}
		return tile
	}
	for idx < wordLength-blanks {
		// Avoid blanks on draw if user specifies a number of blanks.
		rack[idx] = draw(blanks != 0)
		idx++
	}
	for ; idx < wordLength; idx++ {
		rack[idx] = BlankCharacter
	}
	return rack
}
