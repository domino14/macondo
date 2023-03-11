package kwg

import (
	"github.com/domino14/macondo/tilemapping"
	"github.com/rs/zerolog/log"
)

// FindWord finds a word in a KWG
func FindWord(d *KWG, word string) bool {
	found, _ := findWord(d, d.GetRootNodeIndex(), word, 0)
	return found
}

// FindMachineWord finds a word in a dawg or gaddag.
func FindMachineWord(d *KWG, word tilemapping.MachineWord) bool {
	var found bool

	found, _ = findMachineWord(d, d.GetRootNodeIndex(), word, 0)

	return found
}

func findWord(d *KWG, nodeIdx uint32, word string, curIdx uint8) (
	bool, uint32) {

	mw, err := tilemapping.ToMachineWord(word, d.GetAlphabet())
	if err != nil {
		log.Err(err).Msg("convert-to-mw")
		return false, 0
	}
	return findMachineWord(d, nodeIdx, mw, curIdx)
}

func findMachineWord(d *KWG, nodeIdx uint32, word tilemapping.MachineWord, curIdx uint8) (
	bool, uint32) {

	var i byte
	var letter tilemapping.MachineLetter
	var nextNodeIdx uint32

	if curIdx == uint8(len(word)-1) {
		// log.Println("checking letter set last Letter", string(letter),
		// 	"nodeIdx", nodeIdx, "word", string(word))
		ml := word[curIdx]
		return d.InLetterSet(ml, nodeIdx), nodeIdx
	}

	found := false
	for i = byte(1); ; i++ {
		nextNodeIdx = d.ArcIndex(nodeIdx + uint32(i))
		letter = tilemapping.MachineLetter(d.Tile(nodeIdx + uint32(i)))

		curml := word[curIdx]
		if letter == curml {
			// log.Println("Letter", string(letter), "this node idx", nodeIdx,
			// 	"next node idx", nextNodeIdx, "word", string(word))
			found = true
			break
		}
		if d.IsEnd(nodeIdx + uint32(i)) {
			break
		}
	}

	if !found {
		return false, 0
	}
	curIdx++
	return findMachineWord(d, nextNodeIdx, word, curIdx)
}
