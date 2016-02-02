package anagrammer

import (
	"github.com/domino14/macondo/gaddag"
)

func Anagram(letters string, dawg gaddag.SimpleDawg) []string {
	answers := []string{}
	answerChan := make(chan string)

	runes := []rune(letters)

	go anagram(runes, dawg, SimpleGaddag(dawg).GetRootNodeIndex(),
		answerChan)

	for answer := range answerChan {
		answers = append(answers, answer)
	}

	return answers
}

func anagram(runes []rune, dawg gaddag.SimpleDawg, nodeIdx uint32,
	answers chan string) {

}
