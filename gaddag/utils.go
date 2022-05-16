// Utility functions for doing cool things with gaddags.
package gaddag

import (
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/rs/zerolog/log"
)

// A GenericDawg interface allows dawgs and gaddags to be passed to the same
// functions.

type GenericDawgType string

const (
	TypeGaddag GenericDawgType = "gaddag"
	TypeDawg   GenericDawgType = "dawg"
)

type GenericDawg interface {
	GetLetterSet(nodeIdx uint32) alphabet.LetterSet
	InLetterSet(letter alphabet.MachineLetter, nodeIdx uint32) bool
	GetAlphabet() *alphabet.Alphabet
	LexiconName() string
	Nodes() []uint32
	NumArcs(nodeIdx uint32) byte
	ArcToIdxLetter(arcIdx uint32) (uint32, alphabet.MachineLetter)
	GetRootNodeIndex() uint32
	NextNodeIdx(uint32, alphabet.MachineLetter) uint32
	Type() GenericDawgType
}

// FindWord finds a word in a SimpleDawg
func FindWord(d *SimpleDawg, word string) bool {
	found, _ := findWord(d, d.GetRootNodeIndex(), word, 0)
	return found
}

// FindMachineWord finds a word in a dawg or gaddag.
func FindMachineWord(d GenericDawg, word alphabet.MachineWord) bool {
	var found bool
	if d.Type() == TypeGaddag {
		// Flip the word if it's a gaddag, due to how gaddags encode their words.
		wordCopy := make([]alphabet.MachineLetter, len(word))
		copy(wordCopy, word)
		for i, j := 0, len(word)-1; i < j; i, j = i+1, j-1 {
			wordCopy[i], wordCopy[j] = wordCopy[j], wordCopy[i]
		}
		found, _ = findMachineWord(d, d.GetRootNodeIndex(), wordCopy, 0)

	} else {
		found, _ = findMachineWord(d, d.GetRootNodeIndex(), word, 0)
	}

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
func FindInnerHook(d *SimpleDawg, word string, hookType int) bool {
	runes := []rune(word)
	if hookType == BackInnerHook {
		word = string(runes[:len(runes)-1])
	} else if hookType == FrontInnerHook {
		word = string(runes[1:])
	}
	found := FindWord(d, word)
	return found
}

func FindHooks(d *SimpleDawg, word string, hookType int) []rune {
	var runes []rune
	if hookType == BackHooks {
		runes = []rune(word)
	} else if hookType == FrontHooks {
		// Assumes that the passed in dawg is a reverse dawg!
		runes = reverse(word)
	}
	found, nodeIdx := findWord(d, d.GetRootNodeIndex(), string(runes), 0)
	if !found {
		return nil
	}
	hooks := []rune{}
	numArcs := d.NumArcs(nodeIdx)
	// Look for the last letter as an arc.
	found = false
	var nextNodeIdx uint32
	var mletter alphabet.MachineLetter
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, mletter = d.ArcToIdxLetter(nodeIdx + uint32(i))
		if d.alphabet.Letter(mletter) == runes[len(runes)-1] {
			found = true
			break
		}

	}
	if !found {
		return hooks // Empty - no hooks.
	}
	// nextNodeIdx's letter set is all the hooks.
	return d.LetterSetAsRunes(nextNodeIdx)
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

// FindPrefix finds a partial word in the Dawg.
func FindPrefix(d *SimpleDawg, prefix string) bool {
	return findPartialWord(d, d.GetRootNodeIndex(), []rune(strings.ToUpper(prefix)), 0)
}

// findPartialWord returns a boolean indicating if the given partial word is
// in the dawg.
// above.
func findPartialWord(d *SimpleDawg, nodeIdx uint32, partialWord []rune,
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

	numArcs = d.NumArcs(nodeIdx)
	found := false
	for i = byte(1); i <= numArcs; i++ {
		nextNodeIdx, mletter = d.ArcToIdxLetter(nodeIdx + uint32(i))
		if d.alphabet.Letter(mletter) == partialWord[curPrefixIdx] {
			found = true
			break
		}
	}
	if !found {
		if curPrefixIdx == uint8(len(partialWord)-1) {
			// If we didn't find the prefix, it could be a word and thus be
			// in the letter set.
			ml, err := d.alphabet.Val(partialWord[curPrefixIdx])
			if err != nil {
				panic("1.Unexpected err: " + err.Error())
			}
			return d.InLetterSet(ml, nodeIdx)
		}
		return false
	}
	curPrefixIdx++
	return findPartialWord(d, nextNodeIdx, partialWord, curPrefixIdx)
}

func findWord(d GenericDawg, nodeIdx uint32, word string, curIdx uint8) (
	bool, uint32) {

	mw, err := alphabet.ToMachineWord(word, d.GetAlphabet())
	if err != nil {
		log.Err(err).Msg("convert-to-mw")
		return false, 0
	}
	return findMachineWord(d, nodeIdx, mw, curIdx)
}

func findMachineWord(d GenericDawg, nodeIdx uint32, word alphabet.MachineWord, curIdx uint8) (
	bool, uint32) {

	var numArcs, i byte
	var letter alphabet.MachineLetter
	var nextNodeIdx uint32

	if curIdx == uint8(len(word)-1) {
		// log.Println("checking letter set last Letter", string(letter),
		// 	"nodeIdx", nodeIdx, "word", string(word))
		ml := word[curIdx]
		return d.InLetterSet(ml, nodeIdx), nodeIdx
	}

	numArcs = d.NumArcs(nodeIdx)
	found := false
	for i = byte(1); i <= numArcs; i++ {
		nextNodeIdx, letter = d.ArcToIdxLetter(nodeIdx + uint32(i))
		curml := word[curIdx]
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
	return findMachineWord(d, nextNodeIdx, word, curIdx)
}
