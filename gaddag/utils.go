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
	return findWord(g, g.GetRootNodeIndex(), prepareForFind(word), 0, alphabet)
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
		nextNodeIdx, letter = g.ArcToIdxLetter(nodeIdx+uint32(i), alphabet)
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
	alphabet *Alphabet) bool {
	var numArcs, i byte
	var letter rune
	var nextNodeIdx uint32

	numArcs = byte(g[nodeIdx] >> NumArcsBitLoc)
	found := false
	if curIdx < uint8(len(word)-1) {
		for i = byte(1); i <= numArcs; i++ {
			nextNodeIdx, letter = g.ArcToIdxLetter(nodeIdx+uint32(i), alphabet)
			if letter == word[curIdx] {
				found = true
				break
			}
		}
	} else {
		return g.InLetterSet(word[curIdx], nodeIdx, alphabet)
	}
	if !found {
		return false
	}
	curIdx++
	return findWord(g, nextNodeIdx, word, curIdx, alphabet)
}
