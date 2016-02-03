package anagrammer

import (
	"github.com/domino14/macondo/gaddag"
	"log"
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
	log.Println("Calling with rack", rack)
	go anagram(rack, gd, gd.GetRootNodeIndex(), answerChan, "", alphabet)

	for answer := range answerChan {
		log.Println(answer)
		answers = append(answers, answer)
	}
	log.Println("Peacing out")
	return answers
}

func anagram(rack []uint8, dawg gaddag.SimpleGaddag, nodeIdx uint32,
	answers chan string, answerSoFar string, alphabet *gaddag.Alphabet) {

	var nextNodeIdx uint32
	var nextLetter rune
	prefix := ""
	for i := 0; i < len([]rune(answerSoFar)); i++ {
		prefix = prefix + " "
	}
	noTile := true
	for idx, val := range rack {
		if val == 0 {
			continue
		}
		noTile = false
		rack[idx] -= 1

		letter := alphabet.Letter(byte(idx))
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
		rack[idx] += 1
	}
	if noTile {
		//close(answers)
	}
}
