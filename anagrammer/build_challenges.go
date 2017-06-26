package anagrammer

import (
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
	"log"
)

// Generate a build challenge with given args.
// As an additional condition, letters must anagram exactly to at least
// one word, if that argument is passed in.
func GenerateBuildChallenge(args *BuildChallengeArgs, dawg gaddag.SimpleDawg) (
	*Question, int) {
	var dist lexicon.LetterDistribution
	if args.Lexicon == "FISE" {
		dist = lexicon.SpanishLetterDistribution()
	} else {
		dist = lexicon.EnglishLetterDistribution()
	}
	tries := 0
	var w lexicon.Word
	var answers []string
	var meetingCriteria []string
	success := false
	// Try 10000 times. In very rare cases we might not generate a challenge
	// meeting the criteria, and we'd like to exit cleanly instead of get
	// stuck in an infinite loop.
	for tries < 10000 {
		rack := genRack(dist, args.WordLength, 0)
		tries += 1
		answers = Anagram(string(rack), dawg, ModeExact)
		if len(answers) == 0 && args.RequireLengthSolution {
			continue
		}
		answers = Anagram(string(rack), dawg, ModeBuild)
		if len(answers) < args.MinSolutions {
			continue
		}
		meetingCriteria = []string{}
		for _, answer := range answers {
			if len(answer) >= args.MinWordLength {
				meetingCriteria = append(meetingCriteria, answer)
			}
		}
		if len(meetingCriteria) < args.MinSolutions || len(meetingCriteria) > args.MaxSolutions {
			continue
		}
		w = lexicon.Word{Word: string(rack), Dist: dist}
		success = true
		break
	}
	log.Println("[DEBUG]", tries, "tries")
	if !success {
		return nil, 0
	}
	return &Question{Q: w.MakeAlphagram(), A: meetingCriteria}, len(meetingCriteria)
}
