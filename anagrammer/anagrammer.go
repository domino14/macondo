package anagrammer

import (
	"github.com/domino14/macondo/gaddag"
	_ "log"
	"strings"
)

const BlankPos = 31

type AnagramMode int

const (
	ModeBuild AnagramMode = iota
	ModeExact
)

func Anagram(letters string, dawg gaddag.SimpleDawg, mode AnagramMode) []string {
	letters = strings.ToUpper(letters)
	answers := make(map[string]bool)
	answerChan := make(chan string)
	runes := []rune(letters)
	gd := gaddag.SimpleGaddag(dawg)
	alphabet := gd.GetAlphabet()
	// 31 maximum letters allowed. rack[31] will be the blank.
	rack := make([]uint8, 32)
	for _, r := range runes {
		if r != '?' {
			idx, err := alphabet.Val(r)
			if err == nil {
				rack[idx] += 1
			}
		} else {
			rack[BlankPos] += 1
		}
	}
	go func() {
		anagram(rack, gd, gd.GetRootNodeIndex(), answerChan, "", alphabet,
			mode, len(runes))
		close(answerChan)
	}()
	// Use a map to throw away duplicate answers (can happen with blanks)
	for answer := range answerChan {
		answers[answer] = true
	}
	// Turn the answers map into a string array.
	answerStrings := make([]string, len(answers))
	i := 0
	for k := range answers {
		answerStrings[i] = k
		i++
	}
	return answerStrings
}

func anagramHelper(letter rune, dawg gaddag.SimpleGaddag,
	alphabet *gaddag.Alphabet, nodeIdx uint32, answers chan string,
	rack []uint8, answerSoFar string, mode AnagramMode, numTotalLetters int) {

	var nextNodeIdx uint32
	var nextLetter rune

	if dawg.InLetterSet(letter, nodeIdx) {
		if mode == ModeBuild || (mode == ModeExact &&
			len([]rune(answerSoFar+string(letter))) == numTotalLetters) {
			answers <- answerSoFar + string(letter)
		}
	}

	numArcs := dawg.NumArcs(nodeIdx)
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, nextLetter = dawg.ArcToIdxLetter(nodeIdx + uint32(i))
		if letter == nextLetter {
			anagram(rack, dawg, nextNodeIdx, answers,
				answerSoFar+string(letter), alphabet, mode,
				numTotalLetters)
		}
	}
}

func anagram(rack []uint8, dawg gaddag.SimpleGaddag, nodeIdx uint32,
	answers chan string, answerSoFar string, alphabet *gaddag.Alphabet,
	mode AnagramMode, numTotalLetters int) {

	for idx, val := range rack {
		if val == 0 {
			continue
		}
		rack[idx] -= 1
		if idx == BlankPos {
			nlet := alphabet.NumLetters()
			for i := byte(0); i < nlet; i++ {
				letter := alphabet.Letter(i)
				anagramHelper(letter, dawg, alphabet, nodeIdx, answers, rack,
					answerSoFar, mode, numTotalLetters)
			}
		} else {
			letter := alphabet.Letter(byte(idx))
			anagramHelper(letter, dawg, alphabet, nodeIdx, answers, rack,
				answerSoFar, mode, numTotalLetters)
		}

		rack[idx] += 1
	}
}
