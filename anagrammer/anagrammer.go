// Package anagrammer uses a DAWG instead of a GADDAG to simplify the
// algorithm and make it potentially faster - we don't need a GADDAG
// to generate anagrams/subanagrams.
//
// This package generates anagrams and subanagrams and has an RPC
// interface.
package anagrammer

import (
	"github.com/domino14/macondo/gaddag"
	"log"
	"os"
	"strings"
)

func LoadDawgs(dawgPath string) {
	// Load the DAWGs into memory.
	lexica := []string{"America", "CSW15", "FISE09"}
	Dawgs = make(map[string]gaddag.SimpleDawg)
	for _, lex := range lexica {
		filename := dawgPath + lex + ".dawg"
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			log.Println("[ERROR] File", filename, "did not exist. Continuing...")
			continue
		}

		Dawgs[lex] = gaddag.SimpleDawg(gaddag.LoadGaddag(filename))
	}
}

const BlankPos = 31

type AnagramMode int

const (
	ModeBuild AnagramMode = iota
	ModeExact
)

var Dawgs map[string]gaddag.SimpleDawg

type AnagramStruct struct {
	answerChan chan string
	mode       AnagramMode
	numLetters int
}

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

	ahs := &AnagramStruct{
		answerChan: answerChan,
		mode:       mode,
		numLetters: len(runes),
	}

	go func() {
		anagram(ahs, gd, gd.GetRootNodeIndex(), alphabet, "", rack)
		close(ahs.answerChan)
	}()
	// Use a map to throw away duplicate answers (can happen with blanks)
	for answer := range ahs.answerChan {
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

func anagramHelper(letter rune, gd gaddag.SimpleGaddag, ahs *AnagramStruct,
	nodeIdx uint32, alphabet *gaddag.Alphabet, answerSoFar string,
	rack []uint8) {

	var nextNodeIdx uint32
	var nextLetter rune

	if gd.InLetterSet(letter, nodeIdx) {
		toCheck := answerSoFar + string(letter)
		if ahs.mode == ModeBuild || (ahs.mode == ModeExact &&
			len([]rune(toCheck)) == ahs.numLetters) {
			ahs.answerChan <- toCheck
		}
	}

	numArcs := gd.NumArcs(nodeIdx)
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, nextLetter = gd.ArcToIdxLetter(nodeIdx + uint32(i))
		if letter == nextLetter {
			anagram(ahs, gd, nextNodeIdx, alphabet, answerSoFar+string(letter),
				rack)
		}
	}
}

func anagram(ahs *AnagramStruct, gd gaddag.SimpleGaddag, nodeIdx uint32,
	alphabet *gaddag.Alphabet, answerSoFar string, rack []uint8) {
	for idx, val := range rack {
		if val == 0 {
			continue
		}
		rack[idx] -= 1
		if idx == BlankPos {
			nlet := alphabet.NumLetters()
			for i := byte(0); i < nlet; i++ {
				letter := alphabet.Letter(i)
				anagramHelper(letter, gd, ahs, nodeIdx, alphabet, answerSoFar,
					rack)
			}
		} else {
			letter := alphabet.Letter(byte(idx))
			anagramHelper(letter, gd, ahs, nodeIdx, alphabet, answerSoFar,
				rack)
		}

		rack[idx] += 1
	}
}
