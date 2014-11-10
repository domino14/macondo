// Utility functions for doing cool things with gaddags.
package gaddag

//import "fmt"

// FindPrefix returns a boolean indicating if the prefix is in the GADDAG.
// This function is meant to be exported.
// We should just look for first_letter^rest_of_word
func FindPrefix(g SimpleGaddag, prefix string) bool {
	newString := string(prefix[0]) + string(SeparationToken) + prefix[1:]
	return findPartialWord(g, 0, newString, 0)
}

func FindWord(g SimpleGaddag, word string) bool {
	newWord := string(word[0]) + string(SeparationToken) + word[1:]
	return findWord(g, 0, newWord, 0)
}

// findPartialWord returns a boolean indicating if the given partial word is
// in the GADDAG. The partial word has already been flipped by FindPrefix
// above.
func findPartialWord(g SimpleGaddag, nodeIdx uint32, partialWord string,
	curPrefixIdx uint8) bool {
	var numArcs, letter, i byte
	var nextNodeIdx, letterSet uint32

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
			letterSet = g[nodeIdx] & ((1 << NumArcsBitLoc) - 1)
			if letterSet&(1<<(partialWord[curPrefixIdx]-'A')) != 0 {
				return true
			}
			return false
		} else {
			return false
		}
	}
	curPrefixIdx++
	return findPartialWord(g, nextNodeIdx, partialWord, curPrefixIdx)
}

func findWord(g SimpleGaddag, nodeIdx uint32, word string, curIdx uint8) bool {
	var numArcs, letter, i byte
	var nextNodeIdx, letterSet uint32

	numArcs = byte(g[nodeIdx] >> NumArcsBitLoc)
	found := false
	if curIdx < uint8(len(word)-1) {
		for i = byte(1); i <= numArcs; i++ {
			nextNodeIdx, letter = g.ArcToIdxLetter(nodeIdx + uint32(i))
			if letter == word[curIdx] {
				found = true
				break
			}
		}
	} else {
		letterSet = g[nodeIdx] & ((1 << NumArcsBitLoc) - 1)
		if letterSet&(1<<(word[curIdx]-'A')) != 0 {
			return true
		}
		return false
	}
	if !found {
		return false
	}
	curIdx++
	return findWord(g, nextNodeIdx, word, curIdx)
}
