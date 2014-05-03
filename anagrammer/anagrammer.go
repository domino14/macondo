// Package anagrammer implements an anagrammer based on a GADDAG data
// structure.
// I am basing this (and the movegen in the future) on my ujamaa program
// at github.com/domino14/ujamaa
// However, in order to simplify data structures and (hopefully) speed things
// up, we replace ARC* with just the index of the node that the ARC would
// point to in all the algorithms.
package anagrammer

import (
	"fmt"
	"github.com/domino14/gorilla/gaddag"
	"github.com/domino14/gorilla/movegen"
	"strings"
)

const (
	modeBuild   = iota
	modeAnagram = iota
)

var answerSet map[string]bool

// anagramGen This is a simplified version of the "Gen" function from
// the original GADDAG paper. Once we build out the actual GADDAG move
// generator, we should replace this function with the GADDAG Gen.
func anagramGen(gaddagData []uint32, pos int8, word string, rack []uint8,
	nodeIdx uint32, mode uint8) {
	var i, k uint8
	if !movegen.LettersRemain(rack) {
		return
	}
	// For all letters except the blank
	for i = 0; i < 26; i++ {
		if rack[i] > 0 {
			// Letter i + 'A' is on this rack. Temporarily remove it.
			rack[i]--
			anagramGoOn(gaddagData, pos, i+'A', word, rack,
				movegen.NextNode(gaddagData, nodeIdx, i+'A'), nodeIdx, mode)
			// Re-add letter
			rack[i]++
		}
	}
	// Check if there is a blank.
	if rack[26] > 0 {
		// For each blank:
		for k = 0; k < 26; k++ {
			rack[26]--
			anagramGoOn(gaddagData, pos, k+'A', word, rack,
				movegen.NextNode(gaddagData, nodeIdx, k+'A'), nodeIdx, mode)
			rack[26]++
		}
	}
}

// anagramGoOn This is a simplified version of the "GoOn" function from
// the original GADDAG paper. Once we build out the actual GADDAG move
// generator, we should replace this function with the GADDAG GoOn.
func anagramGoOn(gaddagData []uint32, pos int8, L byte, word string,
	rack []uint8, newNodeIdx uint32, oldNodeIdx uint32, mode uint8) {
	if pos <= 0 {
		word := string(L) + word
		if gaddag.ContainsLetter(gaddagData, oldNodeIdx, L) {
			if mode == modeBuild || (mode == modeAnagram &&
				movegen.LettersRemain(rack)) {
				addPlay(word)
			}
		}
		if newNodeIdx != 0 {
			anagramGen(gaddagData, pos-1, word, rack, newNodeIdx, mode)
			newNodeIdx = movegen.NextNode(gaddagData, newNodeIdx,
				gaddag.SeparationToken)
			// Now shift direction.
			if newNodeIdx != 0 {
				anagramGen(gaddagData, 1, word, rack, newNodeIdx, mode)
			}
		}
	} else if pos > 0 {
		word := word + string(L)
		if gaddag.ContainsLetter(gaddagData, oldNodeIdx, L) {
			if mode == modeBuild || (mode == modeAnagram &&
				movegen.LettersRemain(rack)) {
				addPlay(word)
			}
		}
		if newNodeIdx != 0 {
			anagramGen(gaddagData, pos+1, word, rack, newNodeIdx, mode)
		}
	}
}

// turnStringIntoRack Turns a given rack into a uint8 slice of 27 integers,
// one for each letter of the alphabet (blank is the 27th).
func turnStringIntoRack(str string) []uint8 {
	rack := make([]uint8, 27)
	str = strings.ToUpper(str)
	for _, c := range str {
		if c == '?' {
			rack[26]++
		} else {
			rack[c-'A']++
		}
	}
	return rack
}

func addPlay(word string) {
	answerSet[word] = true
}

// Anagram anagrams or builds the passed in string.
func Anagram(gaddagData []uint32, str string, mode string) {
	answerSet = make(map[string]bool)
	rack := turnStringIntoRack(str)
	initWord := ""
	switch mode {
	case "anagram":
		anagramGen(gaddagData, 0, initWord, rack, 0, modeAnagram)
	case "build":
		anagramGen(gaddagData, 0, initWord, rack, 0, modeBuild)
	}
	fmt.Println(answerSet)
}
