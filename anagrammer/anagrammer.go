// Package anagrammer uses a DAWG instead of a GADDAG to simplify the
// algorithm and make it potentially faster - we don't need a GADDAG
// to generate anagrams/subanagrams.
//
// This package generates anagrams and subanagrams and has an RPC
// interface.
package anagrammer

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/rs/zerolog/log"
)

type dawgInfo struct {
	dawg *gaddag.SimpleDawg
	dist *alphabet.LetterDistribution
}

var Dawgs map[string]*dawgInfo

func (di *dawgInfo) GetDawg() *gaddag.SimpleDawg {
	return di.dawg
}

func (di *dawgInfo) GetDist() *alphabet.LetterDistribution {
	return di.dist
}

func LoadDawgs(dawgPath string) {
	// Load the DAWGs into memory.
	var dawgs []string
	err := filepath.Walk(dawgPath, func(path string, info os.FileInfo, err error) error {
		if filepath.Ext(path) != ".dawg" {
			return nil
		}
		if strings.Contains(path, "-r.dawg") {
			return nil
		}
		dawgs = append(dawgs, path)
		return nil
	})
	if err != nil {
		panic(err)
	}
	log.Debug().Msgf("Found files in directory: %v", dawgs)
	Dawgs = make(map[string]*dawgInfo)
	for _, filename := range dawgs {
		dawg, _ := gaddag.LoadDawg(filename)
		lex := dawg.LexiconName()
		Dawgs[lex] = &dawgInfo{}
		Dawgs[lex].dawg = dawg
		if strings.Contains(lex, "FISE") {
			Dawgs[lex].dist = alphabet.SpanishLetterDistribution()
		} else if strings.Contains(lex, "OSPS") {
			Dawgs[lex].dist = alphabet.PolishLetterDistribution()
		} else {
			Dawgs[lex].dist = alphabet.EnglishLetterDistribution()
		}
	}
}

const BlankPos = alphabet.MaxAlphabetSize

type AnagramMode int

const (
	ModeBuild AnagramMode = iota
	ModeExact
	ModePattern
)

type AnagramStruct struct {
	answerList []string
	mode       AnagramMode
	numLetters int
}

func Anagram(letters string, d *gaddag.SimpleDawg, mode AnagramMode) []string {

	letters = strings.ToUpper(letters)
	answerList := []string{}
	runes := []rune(letters)
	alph := d.GetAlphabet()
	rack := alphabet.RackFromString(letters, alph)

	ahs := &AnagramStruct{
		answerList: answerList,
		mode:       mode,
		numLetters: len(runes),
	}
	stopChan := make(chan struct{})

	go func() {
		anagram(ahs, d, d.GetRootNodeIndex(), "", rack)
		close(stopChan)
	}()
	<-stopChan

	return dedupeAndTransformAnswers(ahs.answerList, alph)
	//return ahs.answerList
}

func dedupeAndTransformAnswers(answerList []string, alph *alphabet.Alphabet) []string {
	// Use a map to throw away duplicate answers (can happen with blanks)
	// This seems to be significantly faster than allowing the anagramming
	// goroutine to write directly to a map.
	empty := struct{}{}
	answers := make(map[string]struct{})
	for _, answer := range answerList {
		answers[alphabet.MachineWord(answer).UserVisible(alph)] = empty
	}

	// Turn the answers map into a string array.
	answerStrings := make([]string, len(answers))
	i := 0
	for k := range answers {
		answerStrings[i] = k
		i++
	}
	return answerStrings
}

func anagramHelper(letter alphabet.MachineLetter, d *gaddag.SimpleDawg,
	ahs *AnagramStruct, nodeIdx uint32, answerSoFar string, rack *alphabet.Rack) {

	var nextNodeIdx uint32
	var nextLetter alphabet.MachineLetter

	if d.InLetterSet(letter, nodeIdx) {
		toCheck := answerSoFar + string(letter)
		if ahs.mode == ModeBuild || (ahs.mode == ModeExact &&
			len(toCheck) == ahs.numLetters) {

			ahs.answerList = append(ahs.answerList, toCheck)
		}
	}

	numArcs := d.NumArcs(nodeIdx)
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, nextLetter = d.ArcToIdxLetter(nodeIdx + uint32(i))
		if letter == nextLetter {
			anagram(ahs, d, nextNodeIdx, answerSoFar+string(letter), rack)
		}
	}
}

func anagram(ahs *AnagramStruct, d *gaddag.SimpleDawg, nodeIdx uint32,
	answerSoFar string, rack *alphabet.Rack) {

	for idx, val := range rack.LetArr {
		if val == 0 {
			continue
		}
		rack.LetArr[idx]--
		if idx == BlankPos {
			nlet := alphabet.MachineLetter(d.GetAlphabet().NumLetters())
			for i := alphabet.MachineLetter(0); i < nlet; i++ {
				anagramHelper(i, d, ahs, nodeIdx, answerSoFar, rack)
			}
		} else {
			letter := alphabet.MachineLetter(idx)
			anagramHelper(letter, d, ahs, nodeIdx, answerSoFar, rack)
		}

		rack.LetArr[idx]++
	}
}
