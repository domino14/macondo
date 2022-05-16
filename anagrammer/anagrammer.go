// Package anagrammer uses a DAWG instead of a GADDAG to simplify the
// algorithm and make it potentially faster - we don't need a GADDAG
// to generate anagrams/subanagrams.
//
// This package generates anagrams and subanagrams and has an RPC
// interface.
package anagrammer

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
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

func LoadDawgs(cfg *config.Config, dawgPath string) {
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
		var err error
		if strings.HasPrefix(lex, "FISE") {
			Dawgs[lex].dist, err = alphabet.SpanishLetterDistribution(cfg)
		} else if strings.HasPrefix(lex, "OSPS") {
			Dawgs[lex].dist, err = alphabet.PolishLetterDistribution(cfg)
		} else if strings.HasPrefix(lex, "RD") {
			Dawgs[lex].dist, err = alphabet.GermanLetterDistribution(cfg)
		} else if strings.HasPrefix(lex, "NSF") {
			Dawgs[lex].dist, err = alphabet.NorwegianLetterDistribution(cfg)
		} else if strings.HasPrefix(lex, "FRA") {
			Dawgs[lex].dist, err = alphabet.FrenchLetterDistribution(cfg)
		} else {
			Dawgs[lex].dist, err = alphabet.EnglishLetterDistribution(cfg)
		}
		if err != nil {
			panic(err)
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

type rangeBlank struct {
	count       int
	letterRange []alphabet.MachineLetter
}

// RackWrapper wraps an alphabet.Rack and adds helper data structures
// to make it usable for anagramming.
type RackWrapper struct {
	rack        *alphabet.Rack
	rangeBlanks []rangeBlank
	numLetters  int
}

func makeRack(letters string, alph *alphabet.Alphabet) (*RackWrapper, error) {
	bracketedLetters := []alphabet.MachineLetter{}
	parsingBracket := false

	rack := alphabet.NewRack(alph)

	convertedLetters := []alphabet.MachineLetter{}
	rb := []rangeBlank{}
	numLetters := 0
	for _, s := range letters {
		if s == alphabet.BlankToken {
			convertedLetters = append(convertedLetters, alphabet.BlankMachineLetter)
			numLetters++
			continue
		}

		if s == '[' {
			// Basically treat as a blank that can only be a subset of all
			// letters.
			if parsingBracket {
				return nil, errors.New("Badly formed search string")
			}
			parsingBracket = true
			bracketedLetters = []alphabet.MachineLetter{}
			continue
		}
		if s == ']' {
			if !parsingBracket {
				return nil, errors.New("Badly formed search string")
			}
			parsingBracket = false
			rb = append(rb, rangeBlank{1, bracketedLetters})
			numLetters++
			continue

		}
		// Otherwise it's just a letter.
		ml, err := alph.Val(s)
		if err != nil {
			// Ignore this error, but log it.
			log.Error().Msgf("Ignored error: %v", err)
			continue
		}
		if parsingBracket {
			bracketedLetters = append(bracketedLetters, ml)
			continue
		}
		numLetters++
		convertedLetters = append(convertedLetters, ml)
	}
	if parsingBracket {
		return nil, errors.New("Badly formed search string")
	}
	rack.Set(convertedLetters)

	return &RackWrapper{
		rack:        rack,
		rangeBlanks: rb,
		numLetters:  numLetters,
	}, nil
}

func Anagram(letters string, d *gaddag.SimpleDawg, mode AnagramMode) []string {

	letters = strings.ToUpper(letters)
	answerList := []string{}
	alph := d.GetAlphabet()

	rw, err := makeRack(letters, alph)
	if err != nil {
		log.Error().Msgf("Anagram error: %v", err)
		return []string{}
	}

	ahs := &AnagramStruct{
		answerList: answerList,
		mode:       mode,
		numLetters: rw.numLetters,
	}
	stopChan := make(chan struct{})

	go func() {
		anagram(ahs, d, d.GetRootNodeIndex(), "", rw)
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

// XXX: utf8.RuneCountInString is slow, but necessary to support unicode tiles.
func anagramHelper(letter alphabet.MachineLetter, d *gaddag.SimpleDawg,
	ahs *AnagramStruct, nodeIdx uint32, answerSoFar string, rw *RackWrapper) {

	// log.Debug().Msgf("Anagram helper called with %v %v", letter, answerSoFar)
	var nextNodeIdx uint32
	var nextLetter alphabet.MachineLetter

	if d.InLetterSet(letter, nodeIdx) {
		toCheck := answerSoFar + string(letter)
		if ahs.mode == ModeBuild || (ahs.mode == ModeExact &&
			utf8.RuneCountInString(toCheck) == ahs.numLetters) {

			// log.Debug().Msgf("Appending word %v", toCheck)
			ahs.answerList = append(ahs.answerList, toCheck)
		}
	}

	numArcs := d.NumArcs(nodeIdx)
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, nextLetter = d.ArcToIdxLetter(nodeIdx + uint32(i))
		if letter == nextLetter {
			anagram(ahs, d, nextNodeIdx, answerSoFar+string(letter), rw)
		}
	}
}

func anagram(ahs *AnagramStruct, d *gaddag.SimpleDawg, nodeIdx uint32,
	answerSoFar string, rw *RackWrapper) {

	for idx, val := range rw.rack.LetArr {
		if val == 0 {
			continue
		}
		rw.rack.LetArr[idx]--
		if idx == BlankPos {
			// log.Debug().Msgf("Blank is NOT range")

			nlet := alphabet.MachineLetter(d.GetAlphabet().NumLetters())
			for i := alphabet.MachineLetter(0); i < nlet; i++ {
				anagramHelper(i, d, ahs, nodeIdx, answerSoFar, rw)
			}

		} else {
			letter := alphabet.MachineLetter(idx)
			// log.Debug().Msgf("Found regular letter %v", letter)
			anagramHelper(letter, d, ahs, nodeIdx, answerSoFar, rw)
		}

		rw.rack.LetArr[idx]++
	}
	for idx := range rw.rangeBlanks {
		// log.Debug().Msgf("whichblank %v Blank is range, range is %v",
		// 	rw.whichBlank, blank.letterRange)
		if rw.rangeBlanks[idx].count == 0 {
			continue
		}
		rw.rangeBlanks[idx].count--

		for _, ml := range rw.rangeBlanks[idx].letterRange {
			// log.Debug().Msgf("Making blank %v a %v", rw.whichBlank, ml)
			anagramHelper(ml, d, ahs, nodeIdx, answerSoFar, rw)
		}
		rw.rangeBlanks[idx].count++
	}
}
