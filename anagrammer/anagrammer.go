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
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/movegen"
	"strings"
	"time"
)

const (
	ModeBuild   = iota
	ModeAnagram = iota
)

type Rack struct {
	rack  []uint8
	count uint8
}

var answerSet map[string]bool

// anagram This function is used for anagramming or building. We want to use
// this instead of the gen/goOn because it is much faster.
func anagram(g gaddag.SimpleGaddag, nodeIdx uint32, prefix []byte,
	rack *Rack, mode uint8) {
	// Go through all arc children.
	numArcs := g[nodeIdx] >> gaddag.NumArcsBitLoc
	for i := uint32(1); i <= numArcs; i++ {
		nextNodeIdx, letter := g.ArcToIdxLetter(nodeIdx + i)
		rackPos := letter - 'A'
		if letter == gaddag.SeparationToken {
			break
		}
		if rack.rack[rackPos] <= 0 {
			continue
		}
		rack.rack[rackPos]--
		rack.count--
		newPrefix := []byte{letter}
		newPrefix = append(newPrefix, prefix...)
		// The following bitwise & checks to see if this letter is in the
		// node's LetterSet.
		if g[nodeIdx]&(1<<rackPos) != 0 {
			// We just made a word.
			if mode == ModeAnagram && rack.count == 0 {
				addPlay(newPrefix)
			} else if mode == ModeBuild {
				addPlay(newPrefix)
			}
		}
		anagram(g, nextNodeIdx, newPrefix, rack, mode)
		// Restore letter to rack.
		rack.rack[rackPos]++
		rack.count++
	}

}

// anagramGen This is a simplified version of the "Gen" function from
// the original GADDAG paper. Once we build out the actual GADDAG move
// generator, we should replace this function with the GADDAG Gen.
// func anagramGen(gaddagData []uint32, pos int8, word []byte, rack *Rack,
// 	nodeIdx uint32, mode uint8) {
// 	var i, k uint8
// 	var letter byte
// 	if rack.count == 0 {
// 		return
// 	}
// 	// For all letters except the blank
// 	for i = 0; i < movegen.NumTotalLetters-1; i++ {
// 		if rack.rack[i] > 0 {
// 			// Letter i + 'A' is on this rack. Temporarily remove it.
// 			rack.rack[i]--
// 			rack.count--
// 			letter = i + 'A'
// 			anagramGoOn(gaddagData, pos, letter, word, rack,
// 				movegen.NextNodeIdx(gaddagData, nodeIdx, letter), nodeIdx, mode)
// 			// Re-add letter
// 			rack.rack[i]++
// 			rack.count++
// 		}
// 	}
// 	// Check if there is a blank.
// 	if rack.rack[movegen.BlankPosition] > 0 {
// 		// For each letter that the blank could be:
// 		for k = 0; k < movegen.NumTotalLetters-1; k++ {
// 			rack.rack[movegen.BlankPosition]--
// 			rack.count--
// 			letter = k + 'A'
// 			anagramGoOn(gaddagData, pos, letter, word, rack,
// 				movegen.NextNodeIdx(gaddagData, nodeIdx, letter), nodeIdx, mode)
// 			rack.rack[movegen.BlankPosition]++
// 			rack.count++
// 		}
// 	}
// }

// anagramGoOn This is a simplified version of the "GoOn" function from
// the original GADDAG paper. Once we build out the actual GADDAG move
// generator, we should replace this function with the GADDAG GoOn.
// func anagramGoOn(gaddagData []uint32, pos int8, L byte, word []byte,
// 	rack *Rack, newNodeIdx uint32, oldNodeIdx uint32, mode uint8) {
// 	if pos <= 0 {
// 		// word <- L | word
// 		word = append([]byte{L}, word...)

// 		if gaddag.ContainsLetter(gaddagData, oldNodeIdx, L) {
// 			if mode == ModeBuild || (mode == ModeAnagram && rack.count == 0) {
// 				addPlay(word)
// 			}
// 		}
// 		if newNodeIdx != 0 {
// 			anagramGen(gaddagData, pos-1, word, rack, newNodeIdx, mode)
// 			newNodeIdx = movegen.NextNodeIdx(gaddagData, newNodeIdx,
// 				gaddag.SeparationToken)
// 			// Now shift direction.
// 			if newNodeIdx != 0 {
// 				anagramGen(gaddagData, 1, word, rack, newNodeIdx, mode)
// 			}
// 		}
// 	} else if pos > 0 {
// 		// word <- word | L
// 		word = append(word, L)
// 		if gaddag.ContainsLetter(gaddagData, oldNodeIdx, L) {
// 			if mode == ModeBuild || (mode == ModeAnagram && rack.count == 0) {
// 				addPlay(word)
// 			}
// 		}
// 		if newNodeIdx != 0 {
// 			anagramGen(gaddagData, pos+1, word, rack, newNodeIdx, mode)
// 		}
// 	}
// }

// turnStringIntoRack Turns a given rack into a uint8 slice of 27 integers,
// one for each letter of the alphabet (blank is the 27th).
func turnStringIntoRack(str string) Rack {
	fmt.Println("Turning", str, "into rack")
	rack := make([]uint8, movegen.NumTotalLetters)
	str = strings.ToUpper(str)
	ct := 0
	for _, c := range str {
		if c == '_' {
			rack[movegen.BlankPosition]++
		} else {
			rack[c-'A']++
		}
		ct++
	}
	r := Rack{rack, uint8(ct)}
	fmt.Println(r)
	return r
}

func addPlay(word []byte) {
	answerSet[string(word)] = true
}

// Anagram anagrams or builds the passed in string.
func Anagram(gaddagData []uint32, str string, mode uint8) {
	answerSet = make(map[string]bool)
	rack := turnStringIntoRack(str)
	t0 := time.Now()
	anagram(gaddagData, 0, []byte(nil), &rack, mode)
	t1 := time.Now()
	fmt.Println(answerSet)
	fmt.Println(len(answerSet), "answers")
	fmt.Printf("The call took %v to run.\n", t1.Sub(t0))

}
