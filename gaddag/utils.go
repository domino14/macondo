// Utility functions for doing cool things with gaddags.
package gaddag

import (
	"log"
	"strings"

	"github.com/domino14/macondo/alphabet"
)

// prepare the str for finding by inserting a ^
func prepareForFind(str string) []rune {
	str = strings.ToUpper(str)
	runes := []rune(str)
	runes = append(runes[:1], append([]rune{alphabet.SeparationToken}, runes[1:]...)...)
	return runes
}

// FindPrefix returns a boolean indicating if the prefix is in the GADDAG.
// This function is meant to be exported.
func FindPrefix(g SimpleGaddag, prefix string) bool {
	return findPartialWord(g, g.GetRootNodeIndex(), prepareForFind(prefix), 0)
}

func FindWord(g SimpleGaddag, word string) bool {
	if g.Nodes == nil {
		return false
	}
	found, _ := findWord(g, g.GetRootNodeIndex(), prepareForFind(word), 0)
	return found
}

// Finds a word in a SimpleGaddag that is a DAWG. The word can just be
// found verbatim.
func FindWordDawg(g SimpleGaddag, word string) bool {
	if g.Nodes == nil {
		return false
	}
	found, _ := findWord(g, g.GetRootNodeIndex(), []rune(word), 0)
	return found
}

const (
	BackHooks      = 0
	FrontHooks     = 1
	BackInnerHook  = 2
	FrontInnerHook = 3
)

// FindInnerHook finds whether the word has a back or front "inner hook",
// that is, whether if you remove the back or front letter, you can still
// make a word.
func FindInnerHook(g SimpleGaddag, word string, hookType int) bool {
	runes := []rune(word)
	if hookType == BackInnerHook {
		word = string(runes[:len(runes)-1])
	} else if hookType == FrontInnerHook {
		word = string(runes[1:])
	}
	found := FindWord(g, word)
	return found
}

func FindHooks(g SimpleGaddag, word string, hookType int) []rune {
	var runes []rune
	if hookType == BackHooks {
		runes = prepareForFind(word)
	} else if hookType == FrontHooks {
		runes = reverse(word)
	}
	found, nodeIdx := findWord(g, g.GetRootNodeIndex(), runes, 0)
	if !found {
		return nil
	}
	hooks := []rune{}
	numArcs := g.NumArcs(nodeIdx)
	// Look for the last letter as an arc.
	found = false
	var nextNodeIdx uint32
	var mletter alphabet.MachineLetter
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, mletter = g.ArcToIdxLetter(nodeIdx + uint32(i))
		if g.alphabet.Letter(mletter) == runes[len(runes)-1] {
			found = true
			break
		}

	}
	if !found {
		return hooks // Empty - no hooks.
	}
	// nextNodeIdx's letter set is all the hooks.
	return g.LetterSetAsRunes(nextNodeIdx)
}

func reverse(word string) []rune {
	// reverses the string after turning into a rune array.
	word = strings.ToUpper(word)
	runes := []rune(word)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return runes
}

// findPartialWord returns a boolean indicating if the given partial word is
// in the GADDAG. The partial word has already been flipped by FindPrefix
// above.
func findPartialWord(g SimpleGaddag, nodeIdx uint32, partialWord []rune,
	curPrefixIdx uint8) bool {
	var numArcs, i byte
	var mletter alphabet.MachineLetter
	var nextNodeIdx uint32
	if curPrefixIdx == uint8(len(partialWord)) {
		// If we're here, we're going to get an index error - we will
		// assume we found the word since we have not returned false
		// in earlier iterations.
		return true
	}

	numArcs = g.NumArcs(nodeIdx)
	found := false
	for i = byte(1); i <= numArcs; i++ {
		nextNodeIdx, mletter = g.ArcToIdxLetter(nodeIdx + uint32(i))
		if g.alphabet.Letter(mletter) == partialWord[curPrefixIdx] {
			found = true
			break
		}
	}
	if !found {
		if curPrefixIdx == uint8(len(partialWord)-1) {
			// If we didn't find the prefix, it could be a word and thus be
			// in the letter set.
			ml, err := g.alphabet.Val(partialWord[curPrefixIdx])
			if err != nil {
				panic("1.Unexpected err: " + err.Error())
			}
			return g.InLetterSet(ml, nodeIdx)
		}
		return false
	}
	curPrefixIdx++
	return findPartialWord(g, nextNodeIdx, partialWord, curPrefixIdx)
}

func findWord(g SimpleGaddag, nodeIdx uint32, word []rune, curIdx uint8) (
	bool, uint32) {
	var numArcs, i byte
	var letter alphabet.MachineLetter
	var nextNodeIdx uint32

	if curIdx == uint8(len(word)-1) {
		// log.Println("checking letter set last Letter", string(letter),
		// 	"nodeIdx", nodeIdx, "word", string(word))
		ml, err := g.alphabet.Val(word[curIdx])
		if err != nil {
			log.Printf("[ERROR] %v", err)
			return false, 0
		}
		return g.InLetterSet(ml, nodeIdx), nodeIdx
	}

	numArcs = g.NumArcs(nodeIdx)
	found := false
	for i = byte(1); i <= numArcs; i++ {
		nextNodeIdx, letter = g.ArcToIdxLetter(nodeIdx + uint32(i))
		curml, err := g.alphabet.Val(word[curIdx])
		if err != nil {
			log.Printf("[ERROR] %v", err)
			return false, 0
		}
		if letter == curml {
			// log.Println("Letter", string(letter), "this node idx", nodeIdx,
			// 	"next node idx", nextNodeIdx, "word", string(word))
			found = true
			break
		}
	}

	if !found {
		return false, 0
	}
	curIdx++
	return findWord(g, nextNodeIdx, word, curIdx)
}
