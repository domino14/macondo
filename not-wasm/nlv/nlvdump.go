package main

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/cache"
	"github.com/rs/zerolog/log"
)

func dumpNewLeaves(filename string, alphabetName string) {
	var alph *alphabet.Alphabet
	if alphabetName == "English" {
		// XXX: Support other alphabets in the future in other ways.
		alph = alphabet.EnglishAlphabet()
	}
	_ = alph

	fmt.Println(filename)
	file, err := cache.Open(filename)
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}

	defer file.Close()

	var lenLeaves uint32
	err = binary.Read(file, binary.BigEndian, &lenLeaves)
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}

	leaveFloats := make([]float32, lenLeaves)
	err = binary.Read(file, binary.BigEndian, &leaveFloats)
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}

	var maxLength uint32
	err = binary.Read(file, binary.BigEndian, &maxLength)
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}

	buf := make([]byte, maxLength*lenLeaves)
	err = binary.Read(file, binary.BigEndian, &buf)
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}

	machineLetters := make([]alphabet.MachineLetter, maxLength)
	lenLeavesInt := int(lenLeaves)
	maxLengthInt := int(maxLength)
	for i := 0; i < lenLeavesInt; i++ {
		k := i * maxLengthInt
		actualLength := maxLengthInt
		for j := 0; j < maxLengthInt; j++ {
			if buf[k+j] == 0xff {
				actualLength = j
				break
			}
			machineLetters[j] = alphabet.MachineLetter(buf[k+j])
		}
		machineWord := alphabet.MachineWord(machineLetters[:actualLength])
		userVisibleString := machineWord.UserVisible(alph)
		fmt.Printf("%s,%v\n", userVisibleString, leaveFloats[i])
		if i == 0 {
			continue
		}
		if !(bytes.Compare(buf[k-maxLengthInt:k], buf[k:k+maxLengthInt]) < 0) {
			panic("what")
		}
	}
}

func main() {
	filename := flag.String("filename", "", "filename of the leaves .nlv")

	flag.Parse()
	dumpNewLeaves(*filename, "English")
}
