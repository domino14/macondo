// Utility functions for doing cool things with gaddags.
package gaddag

import "strings"
import _ "log"

// prepare the str for finding by inserting a ^
func prepareForFind(str string) []rune {
	str = strings.ToUpper(str)
	runes := []rune(str)
	runes = append(runes[:1], append([]rune{'^'}, runes[1:]...)...)
	return runes
}

// FindPrefix returns a boolean indicating if the prefix is in the GADDAG.
// This function is meant to be exported.
func FindPrefix(g SimpleGaddag, prefix string) bool {
	// XXX: Too repetitive, should get and save alphabet only once in the
	// g structure.
	alphabet := g.GetAlphabet()

	return findPartialWord(g, g.GetRootNodeIndex(),
		prepareForFind(prefix), 0, alphabet)
}

func FindWord(g SimpleGaddag, word string) bool {
	// XXX: See above note on alphabet.
	alphabet := g.GetAlphabet()
	found, _ := findWord(g, g.GetRootNodeIndex(), prepareForFind(word), 0, alphabet)
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
	alphabet := g.GetAlphabet()

	var runes []rune
	if hookType == BackHooks {
		runes = prepareForFind(word)
	} else if hookType == FrontHooks {
		runes = reverse(word)
	}
	found, nodeIdx := findWord(g, g.GetRootNodeIndex(), runes, 0, alphabet)
	if !found {
		return nil
	}
	hooks := []rune{}
	numArcs := byte(g[nodeIdx] >> NumArcsBitLoc)
	// Look for the last letter as an arc.
	found = false
	var nextNodeIdx uint32
	var letter rune
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, letter = g.ArcToIdxLetter(nodeIdx + uint32(i))
		if letter == runes[len(runes)-1] {
			found = true
			break
		}

	}
	if !found {
		return hooks // Empty - no hooks.
	}
	// nextNodeIdx's letter set is all the hooks.
	return g.LetterSetAsRunes(nextNodeIdx, alphabet)
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
	curPrefixIdx uint8, alphabet *Alphabet) bool {
	var numArcs, i byte
	var letter rune
	var nextNodeIdx uint32
	if curPrefixIdx == uint8(len(partialWord)) {
		// If we're here, we're going to get an index error - we will
		// assume we found the word since we have not returned false
		// in earlier iterations.
		return true
	}

	numArcs = byte(g[nodeIdx] >> NumArcsBitLoc)
	found := false
	for i = byte(1); i <= numArcs; i++ {
		nextNodeIdx, letter = g.ArcToIdxLetter(nodeIdx + uint32(i))
		if letter == partialWord[curPrefixIdx] {
			found = true
			break
		}
	}
	if !found {
		if curPrefixIdx == uint8(len(partialWord)-1) {
			// If we didn't find the prefix, it could be a word and thus be
			// in the letter set.
			return g.InLetterSet(partialWord[curPrefixIdx], nodeIdx, alphabet)
		} else {
			return false
		}
	}
	curPrefixIdx++
	return findPartialWord(g, nextNodeIdx, partialWord, curPrefixIdx, alphabet)
}

func findWord(g SimpleGaddag, nodeIdx uint32, word []rune, curIdx uint8,
	alphabet *Alphabet) (bool, uint32) {
	var numArcs, i byte
	var letter rune
	var nextNodeIdx uint32

	if curIdx == uint8(len(word)-1) {
		// log.Println("checking letter set last Letter", string(letter),
		// 	"nodeIdx", nodeIdx, "word", string(word))
		return g.InLetterSet(word[curIdx], nodeIdx, alphabet), nodeIdx
	}

	numArcs = byte(g[nodeIdx] >> NumArcsBitLoc)
	found := false
	for i = byte(1); i <= numArcs; i++ {
		nextNodeIdx, letter = g.ArcToIdxLetter(nodeIdx + uint32(i))
		if letter == word[curIdx] {
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
	return findWord(g, nextNodeIdx, word, curIdx, alphabet)
}
