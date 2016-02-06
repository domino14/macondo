package anagrammer

import (
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
	"log"
)

// Generate a build challenge with given args.
// As an additional condition, letters must anagram exactly to at least
// one word.
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
	for true {
		rack := genRack(dist, args.WordLength, 0)
		tries += 1
		answers = Anagram(string(rack), dawg, ModeExact)
		if len(answers) == 0 {
			continue
		}
		answers = Anagram(string(rack), dawg, ModeBuild)
		if len(answers) < args.MinSolutions || len(answers) > args.MaxSolutions {
			continue
		}
		w = lexicon.Word{Word: string(rack), Dist: dist}
		break
	}
	log.Println("[DEBUG]", tries, "tries")
	return &Question{Q: w.MakeAlphagram(), A: answers}, len(answers)
}
