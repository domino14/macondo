package kwg

import (
	"errors"
	"fmt"

	"github.com/domino14/macondo/tilemapping"
)

// zero value works. not threadsafe.
type KWGAnagrammer struct {
	ans         tilemapping.MachineWord
	freq        []uint8
	blanks      uint8
	queryLength int
}

func (da KWGAnagrammer) commonInit(kwg *KWG) {
	alph := kwg.GetAlphabet()
	numLetters := alph.NumLetters()
	if cap(da.freq) < int(numLetters)+1 {
		da.freq = make([]uint8, numLetters+1)
	} else {
		da.freq = da.freq[:numLetters+1]
		for i := range da.freq {
			da.freq[i] = 0
		}
	}
	da.blanks = 0
	da.ans = da.ans[:0]
}

func (da *KWGAnagrammer) InitForString(kwg *KWG, tiles string) error {
	da.commonInit(kwg)
	da.queryLength = 0
	alph := kwg.GetAlphabet()
	vals := alph.Vals()
	for _, r := range tiles {
		da.queryLength++ // count number of runes, not number of bytes
		if r == tilemapping.BlankToken {
			da.blanks++
		} else if val, ok := vals[r]; ok {
			da.freq[val]++
		} else {
			return fmt.Errorf("invalid rune %v", r)
		}
	}
	return nil
}

func (da *KWGAnagrammer) InitForMachineWord(kwg *KWG, machineTiles tilemapping.MachineWord) error {
	da.commonInit(kwg)
	da.queryLength = len(machineTiles)
	alph := kwg.GetAlphabet()
	numLetters := alph.NumLetters()
	for _, v := range machineTiles {
		if v == 0 {
			da.blanks++
		} else if uint8(v) < numLetters {
			da.freq[v]++
		} else {
			return fmt.Errorf("invalid byte %v", v)
		}
	}
	return nil
}

// f must not modify the given slice. if f returns error, abort iteration.
func (da *KWGAnagrammer) iterate(kwg *KWG, nodeIdx uint32, minLen int, minExact int, f func(tilemapping.MachineWord) error) error {
	alph := kwg.GetAlphabet()
	numLetters := alph.NumLetters()
	letterSet := kwg.GetLetterSet(nodeIdx)
	// numArcs := kwg.NumArcs(nodeIdx)
	j := tilemapping.MachineLetter(0)
	if kwg.IsEnd(nodeIdx) {
		return nil // ??
	}
	for i := byte(1); ; i++ {
		nextNodeIdx := kwg.ArcIndex(nodeIdx + uint32(i))
		nextLetter := kwg.Tile(nodeIdx + uint32(i))
		if uint8(nextLetter) >= numLetters {
			continue
		}
		for ; j <= tilemapping.MachineLetter(nextLetter); j++ {
			if letterSet&(1<<j) != 0 {
				if da.freq[j] > 0 {
					da.freq[j]--
					da.ans = append(da.ans, j)
					if minLen <= 1 && minExact <= 1 {
						if err := f(da.ans); err != nil {
							return err
						}
					}
					da.ans = da.ans[:len(da.ans)-1]
					da.freq[j]++
				} else if da.blanks > 0 {
					da.blanks--
					da.ans = append(da.ans, j)
					if minLen <= 1 && minExact <= 0 {
						if err := f(da.ans); err != nil {
							return err
						}
					}
					da.ans = da.ans[:len(da.ans)-1]
					da.blanks++
				}
			}
		}
		if da.freq[nextLetter] > 0 {
			da.freq[nextLetter]--
			da.ans = append(da.ans, tilemapping.MachineLetter(nextLetter))
			if err := da.iterate(kwg, nextNodeIdx, minLen-1, minExact-1, f); err != nil {
				return err
			}
			da.ans = da.ans[:len(da.ans)-1]
			da.freq[nextLetter]++
		} else if da.blanks > 0 {
			da.blanks--
			da.ans = append(da.ans, tilemapping.MachineLetter(nextLetter))
			if err := da.iterate(kwg, nextNodeIdx, minLen-1, minExact, f); err != nil {
				return err
			}
			da.ans = da.ans[:len(da.ans)-1]
			da.blanks++
		}
		if kwg.IsEnd(nodeIdx + uint32(i)) {
			break
		}
	}
	for ; uint8(j) < numLetters; j++ {
		if letterSet&(1<<j) != 0 {
			if da.freq[j] > 0 {
				da.freq[j]--
				da.ans = append(da.ans, j)
				if minLen <= 1 && minExact <= 1 {
					if err := f(da.ans); err != nil {
						return err
					}
				}
				da.ans = da.ans[:len(da.ans)-1]
				da.freq[j]++
			} else if da.blanks > 0 {
				da.blanks--
				da.ans = append(da.ans, j)
				if minLen <= 1 && minExact <= 0 {
					if err := f(da.ans); err != nil {
						return err
					}
				}
				da.ans = da.ans[:len(da.ans)-1]
				da.blanks++
			}
		}
	}
	return nil
}

func (da *KWGAnagrammer) Anagram(dawg *KWG, f func(tilemapping.MachineWord) error) error {
	return da.iterate(dawg, dawg.GetRootNodeIndex(), da.queryLength, 0, f)
}

func (da *KWGAnagrammer) Subanagram(dawg *KWG, f func(tilemapping.MachineWord) error) error {
	return da.iterate(dawg, dawg.GetRootNodeIndex(), 1, 0, f)
}

func (da *KWGAnagrammer) Superanagram(dawg *KWG, f func(tilemapping.MachineWord) error) error {
	minExact := da.queryLength - int(da.blanks)
	blanks := da.blanks
	da.blanks = 255
	err := da.iterate(dawg, dawg.GetRootNodeIndex(), da.queryLength, minExact, f)
	da.blanks = blanks
	return err
}

var errHasAnagram = errors.New("has anagram")
var errHasBlanks = errors.New("has blanks")

func foundAnagram(tilemapping.MachineWord) error {
	return errHasAnagram
}

// checks if a word with no blanks has any valid anagrams.
func (da *KWGAnagrammer) IsValidJumble(dawg *KWG, word tilemapping.MachineWord) (bool, error) {
	if err := da.InitForMachineWord(dawg, word); err != nil {
		return false, err
	} else if da.blanks > 0 {
		return false, errHasBlanks
	}
	err := da.Anagram(dawg, foundAnagram)
	if err == nil {
		return false, nil
	} else if err == errHasAnagram {
		return true, nil
	} else {
		return false, err
	}
}
