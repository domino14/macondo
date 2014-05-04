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
	"time"
)

const (
	ModeBuild   = iota
	ModeAnagram = iota
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
	for i = 0; i < movegen.NumTotalLetters-1; i++ {
		if rack[i] > 0 {
			// Letter i + 'A' is on this rack. Temporarily remove it.
			rack[i]--
			anagramGoOn(gaddagData, pos, i+'A', word, rack,
				movegen.NextNodeIdx(gaddagData, nodeIdx, i+'A'), nodeIdx, mode)
			// Re-add letter
			rack[i]++
		}
	}
	// Check if there is a blank.
	if rack[movegen.BlankPosition] > 0 {
		// For each letter that the blank could be:
		for k = 0; k < movegen.NumTotalLetters-1; k++ {
			rack[movegen.BlankPosition]--
			anagramGoOn(gaddagData, pos, k+'A', word, rack,
				movegen.NextNodeIdx(gaddagData, nodeIdx, k+'A'), nodeIdx, mode)
			rack[movegen.BlankPosition]++
		}
	}
}

// anagramGoOn This is a simplified version of the "GoOn" function from
// the original GADDAG paper. Once we build out the actual GADDAG move
// generator, we should replace this function with the GADDAG GoOn.
func anagramGoOn(gaddagData []uint32, pos int8, L byte, word string,
	rack []uint8, newNodeIdx uint32, oldNodeIdx uint32, mode uint8) {
	if pos <= 0 {
		wordC := string(L) + word
		if gaddag.ContainsLetter(gaddagData, oldNodeIdx, L) {
			if mode == ModeBuild || (mode == ModeAnagram &&
				!movegen.LettersRemain(rack)) {
				addPlay(wordC)
			}
		}
		if newNodeIdx != 0 {
			anagramGen(gaddagData, pos-1, wordC, rack, newNodeIdx, mode)
			newNodeIdx = movegen.NextNodeIdx(gaddagData, newNodeIdx,
				gaddag.SeparationToken)
			// Now shift direction.
			if newNodeIdx != 0 {
				anagramGen(gaddagData, 1, wordC, rack, newNodeIdx, mode)
			}
		}
	} else if pos > 0 {
		wordC := word + string(L)
		if gaddag.ContainsLetter(gaddagData, oldNodeIdx, L) {
			if mode == ModeBuild || (mode == ModeAnagram &&
				!movegen.LettersRemain(rack)) {
				addPlay(wordC)
			}
		}
		if newNodeIdx != 0 {
			anagramGen(gaddagData, pos+1, wordC, rack, newNodeIdx, mode)
		}
	}
}

// turnStringIntoRack Turns a given rack into a uint8 slice of 27 integers,
// one for each letter of the alphabet (blank is the 27th).
func turnStringIntoRack(str string) []uint8 {
	rack := make([]uint8, movegen.NumTotalLetters)
	str = strings.ToUpper(str)
	for _, c := range str {
		if c == '_' {
			rack[movegen.BlankPosition]++
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
func Anagram(gaddagData []uint32, str string, mode uint8) {
	answerSet = make(map[string]bool)
	rack := turnStringIntoRack(str)
	initWord := ""
	t0 := time.Now()
	anagramGen(gaddagData, 0, initWord, rack, 0, mode)
	t1 := time.Now()
	fmt.Println(answerSet)
	fmt.Println(len(answerSet), "answers")
	fmt.Printf("The call took %v to run.\n", t1.Sub(t0))

}
