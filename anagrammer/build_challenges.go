package anagrammer

import (
	"context"
	"fmt"
	"log"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
)

// GenerateBuildChallenge generates a build challenge with given args.
// As an additional condition, letters must anagram exactly to at least
// one word, if that argument is passed in.
func GenerateBuildChallenge(ctx context.Context, args *BuildChallengeArgs,
	dawg gaddag.SimpleDawg) (*Question, int, error) {

	var dist lexicon.LetterDistribution
	if args.Lexicon == "FISE09" {
		dist = lexicon.SpanishLetterDistribution()
	} else {
		dist = lexicon.EnglishLetterDistribution()
	}
	tries := 0

	doIteration := func() (*Question, error) {
		rack := genRack(dist, args.WordLength, 0)
		tries++
		answers := Anagram(string(rack), dawg, ModeExact)
		if len(answers) == 0 && args.RequireLengthSolution {
			return nil, fmt.Errorf("exact required and not found: %v", string(rack))
		}
		answers = Anagram(string(rack), dawg, ModeBuild)
		if len(answers) < args.MinSolutions {
			return nil, fmt.Errorf("total answers fewer than min solutions: %v < %v",
				len(answers), args.MinSolutions)
		}
		meetingCriteria := []string{}
		for _, answer := range answers {
			if len(answer) >= args.MinWordLength {
				meetingCriteria = append(meetingCriteria, answer)
			}
		}
		if len(meetingCriteria) < args.MinSolutions || len(meetingCriteria) > args.MaxSolutions {
			return nil, fmt.Errorf("answers (%v) not match criteria: %v - %v",
				len(meetingCriteria), args.MinSolutions, args.MaxSolutions)
		}
		w := lexicon.Word{Word: string(rack), Dist: dist}
		return &Question{Q: w.MakeAlphagram(), A: meetingCriteria}, nil
	}

	for {
		select {
		case <-ctx.Done():
			log.Printf("[INFO] Could not generate before deadline, exiting.")
			return nil, 0, ctx.Err()
		default:
			question, err := doIteration()
			if err != nil {
				log.Printf("[DEBUG] %v", err)
				continue
			}
			log.Printf("[DEBUG] %v tries", tries)
			return question, len(question.A), nil
		}
	}
}
