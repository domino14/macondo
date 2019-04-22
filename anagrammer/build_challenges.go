package anagrammer

import (
	"context"
	"fmt"
	"log"

	"github.com/domino14/macondo/alphabet"
)

// GenerateBuildChallenge generates a build challenge with given args.
// As an additional condition, letters must anagram exactly to at least
// one word, if that argument is passed in.
func GenerateBuildChallenge(ctx context.Context, args *BuildChallengeArgs,
	dinfo *dawgInfo) (*Question, int, error) {

	tries := 0
	alph := dinfo.dawg.GetAlphabet()

	doIteration := func() (*Question, error) {
		rack := alphabet.MachineWord(genRack(dinfo.dist, args.WordLength, 0, alph))
		tries++
		answers := Anagram(rack.UserVisible(alph), dinfo.dawg, ModeExact)
		if len(answers) == 0 && args.RequireLengthSolution {
			return nil, fmt.Errorf("exact required and not found: %v", rack.UserVisible(alph))
		}
		answers = Anagram(rack.UserVisible(alph), dinfo.dawg, ModeBuild)
		if len(answers) < args.MinSolutions {
			return nil, fmt.Errorf("total answers fewer than min solutions: %v < %v",
				len(answers), args.MinSolutions)
		}
		meetingCriteria := []string{}
		for _, answer := range answers {
			// NB: This might be the only place where we need to use
			// len([]rune(x)) instead of len(x). It's important to use
			// `MachineLetter`s everywhere we can.
			if len([]rune(answer)) >= args.MinWordLength {
				meetingCriteria = append(meetingCriteria, answer)
			}
		}
		if len(meetingCriteria) < args.MinSolutions || len(meetingCriteria) > args.MaxSolutions {
			return nil, fmt.Errorf("answers (%v) not match criteria: %v - %v",
				len(meetingCriteria), args.MinSolutions, args.MaxSolutions)
		}
		w := alphabet.Word{Word: rack.UserVisible(alph), Dist: dinfo.dist}
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
