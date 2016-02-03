package anagrammer

import (
	"github.com/domino14/macondo/gaddag"
	_ "log"
)

const BlankPos = 31

func Anagram(letters string, dawg gaddag.SimpleDawg) []string {
	answers := []string{}
	answerChan := make(chan string)
	runes := []rune(letters)
	gd := gaddag.SimpleGaddag(dawg)
	alphabet := gd.GetAlphabet()
	// 31 maximum letters allowed. rack[31] will be the blank.
	rack := make([]uint8, 32)
	for _, r := range runes {
		if r != '?' {
			rack[alphabet.Val(r)] += 1
		} else {
			rack[BlankPos] += 1
		}
	}
	go func() {
		anagram(rack, gd, gd.GetRootNodeIndex(), answerChan, "", alphabet)
		close(answerChan)
	}()

	for answer := range answerChan {
		answers = append(answers, answer)
	}
	return answers
}

func anagramHelper(letterCode byte, dawg gaddag.SimpleGaddag,
	alphabet *gaddag.Alphabet, nodeIdx uint32, answers chan string,
	rack []uint8, answerSoFar string) {

	var nextNodeIdx uint32
	var nextLetter rune
	letter := alphabet.Letter(letterCode)
	//log.Println(prefix, "Rack is now", rack, "letter", string(letter))
	if dawg.InLetterSet(letter, nodeIdx) {
		//	log.Println(prefix, string(letter), "in letter set")
		answers <- answerSoFar + string(letter)
	}

	numArcs := dawg.NumArcs(nodeIdx)
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, nextLetter = dawg.ArcToIdxLetter(nodeIdx + uint32(i))
		if letter == nextLetter {
			anagram(rack, dawg, nextNodeIdx, answers,
				answerSoFar+string(letter), alphabet)
		}
	}
}

func anagram(rack []uint8, dawg gaddag.SimpleGaddag, nodeIdx uint32,
	answers chan string, answerSoFar string, alphabet *gaddag.Alphabet) {

	for idx, val := range rack {
		if val == 0 {
			continue
		}
		rack[idx] -= 1
		anagramHelper(byte(idx), dawg, alphabet, nodeIdx, answers, rack,
			answerSoFar)
		rack[idx] += 1
	}
}
