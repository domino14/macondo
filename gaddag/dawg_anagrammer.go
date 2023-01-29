package gaddag

import (
	"errors"
	"fmt"

	"github.com/domino14/macondo/alphabet"
)

// a "rangeBlank" is a blank that can only be a subset of letters.
type rangeBlank struct {
	inuse       bool
	letterRange []alphabet.MachineLetter
}

// zero value works. not threadsafe.
type DawgAnagrammer struct {
	ans               alphabet.MachineWord
	freq              []uint8
	blanks            uint8
	queryLength       int
	rangeBlanksForUse uint8
	rangeBlanks       []rangeBlank
}

func (da *DawgAnagrammer) commonInit(dawg GenericDawg) {
	alph := dawg.GetAlphabet()
	numLetters := alph.NumLetters()
	if cap(da.freq) < int(numLetters) {
		da.freq = make([]uint8, numLetters)
	} else {
		da.freq = da.freq[:numLetters]
		for i := range da.freq {
			da.freq[i] = 0
		}
	}
	da.blanks = 0
	da.ans = da.ans[:0]
}

func (da *DawgAnagrammer) InitForString(dawg GenericDawg, tiles string) error {
	da.commonInit(dawg)
	da.queryLength = 0
	alph := dawg.GetAlphabet()
	vals := alph.Vals()

	bracketedLetters := []alphabet.MachineLetter{}
	parsingBracket := false
	rb := []rangeBlank{}

	for _, r := range tiles {
		if r == '[' {
			// Basically treat as a blank that can only be a subset of all
			// letters.
			if parsingBracket {
				return errors.New("badly formed search string")
			}
			parsingBracket = true
			bracketedLetters = []alphabet.MachineLetter{}
			continue
		}
		if r == ']' {
			if !parsingBracket {
				return errors.New("badly formed search string")
			}
			parsingBracket = false
			rb = append(rb, rangeBlank{inuse: false, letterRange: bracketedLetters})
			da.queryLength++
			continue
		}
		if parsingBracket {
			val, ok := vals[r]
			if !ok {
				return fmt.Errorf("bracketing - invalid rune %v", r)
			}
			bracketedLetters = append(bracketedLetters, val)
			continue
		}

		da.queryLength++ // count number of runes, not number of bytes
		if r == alphabet.BlankToken {
			da.blanks++
		} else if val, ok := vals[r]; ok {
			da.freq[val]++
		} else {
			return fmt.Errorf("invalid rune %v", r)
		}
	}

	da.rangeBlanks = rb
	da.rangeBlanksForUse = uint8(len(rb))
	fmt.Println("range blanks are", rb)
	fmt.Println("queryLength", da.queryLength)
	fmt.Println("freq", da.freq)
	fmt.Println("blanks", da.blanks)
	fmt.Println(da.rangeBlanksForUse)
	return nil
}

func (da *DawgAnagrammer) InitForMachineWord(dawg GenericDawg, machineTiles alphabet.MachineWord) error {
	da.commonInit(dawg)
	da.queryLength = len(machineTiles)
	alph := dawg.GetAlphabet()
	numLetters := alph.NumLetters()
	for _, v := range machineTiles {
		if v == alphabet.BlankMachineLetter {
			da.blanks++
		} else if uint8(v) < numLetters {
			da.freq[v]++
		} else {
			return fmt.Errorf("invalid byte %v", v)
		}
	}
	return nil
}

func letterInRange(ml alphabet.MachineLetter, letterRange []alphabet.MachineLetter) bool {
	for _, m := range letterRange {
		if m == ml {
			return true
		}
	}
	return false
}

func (da *DawgAnagrammer) findFirstMatchingRangeBlank(ml alphabet.MachineLetter) int {

	for idx, rb := range da.rangeBlanks {
		if !rb.inuse {
			if letterInRange(ml, rb.letterRange) {
				return idx
			}
		}
	}
	return -1

}

// f must not modify the given slice. if f returns error, abort iteration.
func (da *DawgAnagrammer) iterate(dawg GenericDawg, nodeIdx uint32, minLen int, minExact int, f func(alphabet.MachineWord) error) error {
	alph := dawg.GetAlphabet()
	numLetters := alph.NumLetters()
	letterSet := dawg.GetLetterSet(nodeIdx)
	numArcs := dawg.NumArcs(nodeIdx)
	j := alphabet.MachineLetter(0)
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, nextLetter := dawg.ArcToIdxLetter(nodeIdx + uint32(i))
		if uint8(nextLetter) >= numLetters {
			continue
		}
		for ; j <= nextLetter; j++ {
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
				} else if da.rangeBlanksForUse > 0 {
					rbi := da.findFirstMatchingRangeBlank(j)

					if rbi != -1 {
						da.rangeBlanksForUse--
						da.rangeBlanks[rbi].inuse = true
						da.ans = append(da.ans, j)
						if minLen <= 1 && minExact <= 0 {
							if err := f(da.ans); err != nil {
								return err
							}
						}
						da.ans = da.ans[:len(da.ans)-1]
						da.rangeBlanksForUse++
						da.rangeBlanks[rbi].inuse = false
					}
				}
			}
		}
		if da.freq[nextLetter] > 0 {
			da.freq[nextLetter]--
			da.ans = append(da.ans, nextLetter)
			if err := da.iterate(dawg, nextNodeIdx, minLen-1, minExact-1, f); err != nil {
				return err
			}
			da.ans = da.ans[:len(da.ans)-1]
			da.freq[nextLetter]++
		} else if da.blanks > 0 {
			da.blanks--
			da.ans = append(da.ans, nextLetter)
			if err := da.iterate(dawg, nextNodeIdx, minLen-1, minExact, f); err != nil {
				return err
			}
			da.ans = da.ans[:len(da.ans)-1]
			da.blanks++
		} else if da.rangeBlanksForUse > 0 {
			rbi := da.findFirstMatchingRangeBlank(nextLetter)
			if rbi != -1 {
				da.rangeBlanksForUse--
				da.rangeBlanks[rbi].inuse = true
				da.ans = append(da.ans, nextLetter)
				if err := da.iterate(dawg, nextNodeIdx, minLen-1, minExact, f); err != nil {
					return err
				}
				da.ans = da.ans[:len(da.ans)-1]
				da.rangeBlanksForUse++
				da.rangeBlanks[rbi].inuse = false
			}
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
			} else if da.rangeBlanksForUse > 0 {
				rbi := da.findFirstMatchingRangeBlank(j)
				if rbi != -1 {
					da.rangeBlanksForUse--
					da.rangeBlanks[rbi].inuse = true
					da.ans = append(da.ans, j)
					if minLen <= 1 && minExact <= 0 {
						if err := f(da.ans); err != nil {
							return err
						}
					}
					da.ans = da.ans[:len(da.ans)-1]
					da.rangeBlanksForUse++
					da.rangeBlanks[rbi].inuse = false
				}
			}
		}
	}
	return nil
}

func (da *DawgAnagrammer) Anagram(dawg GenericDawg, f func(alphabet.MachineWord) error) error {
	return da.iterate(dawg, dawg.GetRootNodeIndex(), da.queryLength, 0, f)
}

func (da *DawgAnagrammer) Subanagram(dawg GenericDawg, f func(alphabet.MachineWord) error) error {
	return da.iterate(dawg, dawg.GetRootNodeIndex(), 1, 0, f)
}

func (da *DawgAnagrammer) Superanagram(dawg GenericDawg, f func(alphabet.MachineWord) error) error {
	minExact := da.queryLength - int(da.blanks)
	blanks := da.blanks
	da.blanks = 255
	err := da.iterate(dawg, dawg.GetRootNodeIndex(), da.queryLength, minExact, f)
	da.blanks = blanks
	return err
}

var errHasAnagram = errors.New("has anagram")
var errHasBlanks = errors.New("has blanks")

func foundAnagram(alphabet.MachineWord) error {
	return errHasAnagram
}

// checks if a word with no blanks has any valid anagrams.
func (da *DawgAnagrammer) IsValidJumble(dawg GenericDawg, word alphabet.MachineWord) (bool, error) {
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
