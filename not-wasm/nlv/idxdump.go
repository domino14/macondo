package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"

	"github.com/alecthomas/mph"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/cache"
	"github.com/rs/zerolog/log"
)

func dumpMPH(filename string, alphabetName string) {
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

	h, err := mph.Read(file)
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}

	var machineLetters []alphabet.MachineLetter
	for itr := h.Iterate(); itr != nil; itr = itr.Next() {
		k, v := itr.Get()
		machineLetters = machineLetters[:0]
		for j := 0; j < len(k); j++ {
			machineLetters = append(machineLetters, alphabet.MachineLetter(k[j]))
		}
		machineWord := alphabet.MachineWord(machineLetters)
		userVisibleString := machineWord.UserVisible(alph)
		fmt.Printf("%s,%v\n", userVisibleString, math.Float32frombits(binary.BigEndian.Uint32(v)))
	}
}

func main() {
	filename := flag.String("filename", "", "filename of the leaves .idx")

	flag.Parse()
	dumpMPH(*filename, "English")
}
